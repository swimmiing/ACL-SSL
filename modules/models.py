import torch
import torch.nn.functional as F
from torch import nn

import yaml
import argparse

from modules.BEATs.BEATs import BEATs, BEATsConfig
from modules.AudioToken.embedder import FGAEmbedder
from modules.CLIPSeg.clipseg_for_audio import CLIPSeg
from modules.mask_utils import ImageMasker, FeatureMasker
from transformers import AutoTokenizer


class ACL(nn.Module):
    def __init__(self, conf_file: str, device: str):
        """
        Audio-Grounded Contrastive Learning (ACL) model.

        Args:
            conf_file (str): Path to the configuration file.
            device (str): Device to move the model to.
        """
        super(ACL, self).__init__()

        # Get configuration
        with open(conf_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.args = argparse.Namespace()
            self.args.model = argparse.Namespace(**config['model'])
            self.args.clip_embedding_dim = config['clip_conf'][self.args.model.clip]['embedding_dim']
            self.args.clip_name = config['clip_conf'][self.args.model.clip]['name']
            self.pretrain = argparse.Namespace(**config['pretrain'])
            self.args.audio_proj = argparse.Namespace(**config['fga_conf'][self.args.model.audio_proj])

        # Init audio encoder
        checkpoint = torch.load(self.pretrain.audio_backbone)
        cfg = BEATsConfig(checkpoint['cfg'])
        self.audio_backbone = BEATs(cfg)

        # Text Tokenizer for placeholder prompt
        self.tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")

        # Init audio projection layer
        self.audio_proj = FGAEmbedder(input_size=self.args.audio_proj.input_size * 3,
                                      output_size=self.args.audio_proj.output_size)

        # Init audio-visual grounder (Grounder: CLIPSeg)
        self.av_grounder = CLIPSeg.from_pretrained("CIDAS/clipseg-rd64-refined")

        # Init maskers
        self.masker_i = ImageMasker(10.0, 14.0, 1.0)
        self.masker_f = FeatureMasker(0.5, 0.07)

        # Load weights
        self.audio_backbone.load_state_dict(checkpoint['model'])
        self.audio_backbone.predictor = None

        if self.pretrain.audio_proj is not None:
            self.audio_proj.load_state_dict(torch.load(self.pretrain.audio_embedder))

        # Set device
        self.device = device
        self.audio_backbone.to(device=self.device)
        self.av_grounder.to(device=self.device)
        self.audio_proj.to(device=self.device)
        self.masker_i.to(self.device)
        self.masker_f.to(self.device)

    def get_placeholder_token(self, prompt_text: str):
        """
        Get placeholder token from prompt text

        Args:
            prompt_text (str): prompt text without '{}'

        Returns:
            CLIPTokenizerFast result with prompt text
        """
        placeholder_token = self.tokenizer(prompt_text, return_tensors="pt").data['input_ids']
        placeholder_token = F.pad(placeholder_token, (0, 77 - placeholder_token.shape[-1])).to(self.device)
        return placeholder_token

    def train(self, bool: bool = True):
        """
        Set the module in training mode.

        Args:
            bool (bool): If True, set the module in training mode.
        """
        super().train(bool)
        self.av_grounder.requires_grad_(False)
        self.audio_backbone.requires_grad_(False)

    def encode_audio(self, audio: torch.Tensor, placeholder_token: torch.Tensor, pos: int,
                     prompt_size: int) -> torch.Tensor:
        """
        Encode audio input into audio-driven embedding (Audio-Driven Embedder)

        Args:
            audio (torch.Tensor): Input audio tensor.
            placeholder_token (torch.Tensor): Placeholder token for CLIP Text encoder.
            pos (int): Position of audio token.
            prompt_size (int): Size of the placeholder prompt.

        Returns:
            torch.Tensor: Audio-driven embeddings.
        """
        audio_feat = self.audio_backbone.extract_features(audio)[1]
        audio_token_emb = self.audio_proj(audio_feat).unsqueeze(1)
        audio_driven_embedding = self.av_grounder.encode_audio(placeholder_token, audio_token_emb, pos,
                                                               prompt_size + audio_token_emb.shape[1])

        return audio_driven_embedding

    def encode_vision(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode visual input and generate visual embeddings.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Visual embeddings.
        """
        vision_outputs = self.av_grounder.clip.vision_model(pixel_values=image,
                                                            output_attentions=None,
                                                            output_hidden_states=True,
                                                            return_dict=True)
        pooled_output = self.av_grounder.clip.visual_projection(vision_outputs[1])

        return pooled_output

    def forward_decoder(self, image: torch.Tensor, embedding: torch.Tensor, resolution: int = 224) -> torch.Tensor:
        """
        Forward pass of audio-visual grounder

        Args:
            image (torch.Tensor): Input image tensor.
            embedding (torch.Tensor): Condition embedding tensor for grounder.
            resolution (int): Resolution of the output.
            ignore_indices (list): List of indices to ignore.

        Returns:
            torch.Tensor: Logits from the decoder.
        """
        # step 1: forward the query images through the frozen CLIP vision encoder
        vision_outputs = self.av_grounder.clip.vision_model(pixel_values=image,
                                                            output_attentions=None,
                                                            output_hidden_states=True,
                                                            return_dict=True)

        hidden_states = vision_outputs.hidden_states
        # we add +1 here as the hidden states also include the initial embeddings
        activations = [hidden_states[i + 1] for i in self.av_grounder.extract_layers]

        # step 2: compute conditional embeddings, either from text, images or an own provided embedding
        # Audio injected embedding from input argument

        # step 3: forward both the pooled output and the activations through the lightweight decoder to predict masks
        decoder_outputs = self.av_grounder.decoder(
            activations,
            embedding,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )
        logits = decoder_outputs.logits

        if logits.ndim == 2:
            logits = logits.unsqueeze(0).unsqueeze(1)
        else:
            logits = logits.unsqueeze(1)

        B, c, h, w = image.shape
        if (h, w) != (resolution, resolution):
            logits = F.interpolate(logits, resolution, mode='bicubic')

        return logits

    def forward_module(self, image: torch.Tensor, embedding: torch.Tensor, resolution: int = 224,
                       force_comb: bool = False) -> torch.Tensor:
        """
        Forward pass through the module.

        Args:
            image (torch.Tensor): Input image tensor.
            embedding (torch.Tensor): Condition embedding tensor for grounder.
            resolution (int): Resolution of the output tensor.
            force_comb (bool): If True, force to get logits with all combination audio and image.

        Returns:
            torch.Tensor: Logits from the decoder.
        """
        # N image, 1 embedding case -> [B_i, h, w]
        if embedding.shape[0] != image.shape[0] and embedding.shape[0] == 1:
            embeddings = embedding.repeat(image.shape[0], 1)
            logits = self.forward_decoder(image, embeddings, resolution)

        # N image, M embedding case -> [B_i, B_e, h, w]
        elif embedding.shape[0] != image.shape[0] and embedding.shape[0] != 1 and image.shape[0] != 1 or force_comb:
            logit_list = []
            for i in range(embedding.shape[0]):
                embeddings = embedding[i].unsqueeze(0).repeat(image.shape[0], 1)
                logit_list.append(self.forward_decoder(image, embeddings, resolution))
            logits = torch.cat(logit_list, dim=1)

        # N image, N embedding or 1 image, N embedding -> [B_e, h, w]
        else:
            logits = self.forward_decoder(image, embedding, resolution)

        return logits

    def encode_masked_vision(self, image: torch.Tensor, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Encode masked visual feature both image-level and feature-level.

        Args:
            image (torch.Tensor): Input image tensor.
            embedding (torch.Tensor): Condition embedding tensor for grounder.

        Returns:
            tuple[torch.Tensor, torch.Tensor, float, float]: Feature masked embeddings, masked image embeddings, positive area, negative area.
        """
        B, c, h, w = image.shape
        maskclip_feat = self.av_grounder.get_pixels(image)  # v^D: [B, c, h, w]
        clipseg_mask = self.forward_module(image, embedding, h, force_comb=True)  # M^G: [B, B, H, W]

        # Area
        area_matrix = self.masker_i(clipseg_mask).mean((2, 3))
        positive_area = area_matrix.diagonal().mean()
        negative_area = area_matrix.mean() - positive_area / B

        # Feature level masker
        feature_mask = F.interpolate(self.masker_f(clipseg_mask), maskclip_feat.shape[2])

        # Image level masker
        ind = torch.arange(B).to(image.device)
        image_mask = self.masker_i(clipseg_mask[ind, ind].unsqueeze(1))  # Positive pair only
        feature_masked_emb = torch.einsum('bchw,bnhw->bnc', maskclip_feat, feature_mask) / (feature_mask.sum() + 1e-6)

        # step 1: forward the query images through the frozen CLIP vision encoder
        masked_vision_outputs = self.av_grounder.clip.vision_model(pixel_values=image * image_mask,
                                                                   output_attentions=None,
                                                                   output_hidden_states=True,
                                                                   return_dict=True)
        masked_image_emb = self.av_grounder.clip.visual_projection(masked_vision_outputs[1])

        return feature_masked_emb, masked_image_emb, positive_area, negative_area

    def forward(self, image: torch.Tensor, embedding: torch.Tensor, resolution: int = 224) -> dict:
        """
        Forward pass of ACL model.

        Args:
            image (torch.Tensor): Input image tensor.
            embedding (torch.Tensor): Condition embedding tensor for grounder.
            resolution (int): Resolution of the output tensor.

        Returns:
            dict: Output dictionary containing relevant tensors.
        """
        if self.training:
            # seg_logit = self.forward_module(image, embedding, resolution)
            v_f, v_i, p_area, n_area = self.encode_masked_vision(image, embedding)
            out_dict = {'v_f': v_f, 'v_i': v_i, 'p_area': p_area, 'n_area': n_area}

        else:
            seg_logit = self.forward_module(image, embedding, resolution)
            heatmap = self.masker_i(seg_logit, infer=True)
            out_dict = {'heatmap': heatmap}

        return out_dict

    def save(self, model_dir: str):
        """
        Save model parameters to a file. (Only trainable parts)

        Args:
            model_dir (str): Directory to save the model.
        """
        ckp = {'audio_proj': self.audio_proj.state_dict(), 'masker_i': self.masker_i.state_dict()}
        torch.save(ckp, model_dir)

    def load(self, model_dir: str):
        """
        Load model parameters from a file. (Only trainable parts)

        Args:
            model_dir (str): Directory to load the model from.
        """
        ckp = torch.load(model_dir)
        self.audio_proj.load_state_dict(ckp['audio_proj'])
        self.masker_i.load_state_dict(ckp['masker_i'])
