import transformers
import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Tuple, Union, Optional
import numpy as np
from transformers.models.clipseg.modeling_clipseg import _expand_mask


class CLIPSeg(transformers.CLIPSegForImageSegmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """
        Encode textual input and return the text embeddings.

        Args:
            text (torch.Tensor): Input text tensor.

        Returns:
            torch.Tensor: Text embeddings.
        """
        tokens = text
        if text.ndim == 3:
            tokens = torch.squeeze(text, dim=1)
        non_zero_index = torch.nonzero(tokens.sum(axis=0) == 0)[0]
        input_ids = tokens[:, :non_zero_index]
        attention_mask = (input_ids > 0).to(tokens.dtype)
        input_ids += torch.max(input_ids) * (1 - attention_mask)
        conditional_embeddings = self.clip.get_text_features(input_ids, attention_mask=attention_mask,
                                                             position_ids=None)

        return conditional_embeddings

    def similarity(self, image: torch.Tensor, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculate the similarity score between an image and a list of embeddings.

        Args:
            image (torch.Tensor): Input image tensor of shape (B, C, H, W).
            embeddings (List[torch.Tensor]): List of N embedding tensors of shape (dim,).

        Returns:
            torch.Tensor: Similarity scores of shape (B, N) for each batch.
        """
        B, c, h, w = image.shape
        if (h, w) != (352, 352):
            vision_outputs = self.clip.vision_model(pixel_values=F.interpolate(image, 352, mode='bicubic'),
                                                    output_attentions=False,
                                                    output_hidden_states=False,
                                                    return_dict=False)
            img_embedding = self.clip.visual_projection(vision_outputs[1])
        else:
            vision_outputs = self.clip.vision_model(pixel_values=image,
                                                    output_attentions=False,
                                                    output_hidden_states=False,
                                                    return_dict=False)
            img_embedding = self.clip.visual_projection(vision_outputs[1])

        paired_embedding = torch.cat(embeddings, dim=0)
        paired_embedding = paired_embedding.repeat(B, 1)  # Batch-wise replication of embeddings
        paired_embedding = paired_embedding.view(B, -1, img_embedding.size(-1))

        result = torch.matmul(F.normalize(paired_embedding, dim=-1), F.normalize(img_embedding, dim=-1).unsqueeze(-1))
        result = result.squeeze(-1).view(B, -1)
        return F.softmax(result, dim=-1)

    def encode_audio(self, placeholder_token: torch.Tensor, audio_token: torch.Tensor, pos: int,
                     length: int) -> torch.Tensor:
        """
        Encode audio token into the audio-driven embeddings. (Audio-Driven Embedder)

        Args:
            placeholder_token (torch.Tensor): Placeholder text token tensor.
            audio_token (torch.Tensor): Audio token tensor.
            pos (int): Position index for audio token.
            length (int): Length of the input token.

        Returns:
            torch.Tensor: Audio-driven embeddings.

        Reference:
            "Can CLIP Help Sound Source Localization?" WACV 2024
            - https://arxiv.org/abs/2311.04066
        """
        tokens = placeholder_token
        if placeholder_token.ndim == 3:
            tokens = torch.squeeze(placeholder_token, dim=1)

        inputs_embeds = self.clip.text_model.embeddings.token_embedding(tokens).type(
            self.dtype)  # [batch_size, n_ctx, d_model]
        inputs_embeds = torch.cat((inputs_embeds[:, :pos, :], audio_token, inputs_embeds[:, pos:, :]),
                                  dim=1)  # Inject Audio token
        inputs_embeds = inputs_embeds[:, :length, :]

        bsz, seq_len, _ = inputs_embeds.shape
        attention_mask = torch.ones((bsz, seq_len)).to(placeholder_token.device)
        position_ids = torch.arange(length).unsqueeze(0).to(placeholder_token.device)

        position_embeddings = self.clip.text_model.embeddings.position_embedding(position_ids)
        hidden_states = inputs_embeds + position_embeddings

        bsz, seq_len, _ = inputs_embeds.shape
        # CLIPSeg's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIPSeg/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clipseg/model.py#L324
        causal_attention_mask = self.clip.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.clip.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.clip.text_model.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[:, -1, :]
        audio_driven_embeddings = self.clip.text_projection(pooled_output)
        return audio_driven_embeddings

    def get_pixels(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features (pixel-level) from the CLIP image encoder.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Spatial visual features (pixel-level).
        """
        vision_outputs = self.clip.vision_model(pixel_values=image,
                                                output_attentions=None,
                                                output_hidden_states=True,
                                                return_dict=True)
        last_layer = self.clip.vision_model.encoder.layers[-1]

        hidden_states = vision_outputs.hidden_states[-2]
        residual = hidden_states

        hidden_states = last_layer.layer_norm1(hidden_states)

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        # query_states = last_layer.self_attn.q_proj(hidden_states) * last_layer.self_attn.scale
        # key_states = last_layer.self_attn.k_proj(hidden_states)
        value_states = last_layer.self_attn.v_proj(hidden_states)

        value_states = last_layer.self_attn.out_proj(value_states)

        value_states += residual

        residual = value_states
        value_states = last_layer.layer_norm2(value_states)
        value_states = last_layer.mlp(value_states)
        value_states += residual

        value_states = self.clip.vision_model.post_layernorm(value_states)
        output = self.clip.visual_projection(value_states)

        width = int(np.sqrt(tgt_len - 1))
        output = output[:, 1:]
        if output.ndim == 2:
            output = output.unsqueeze(0)

        output = output.permute(0, 2, 1)
        output = output.reshape(bsz, self.clip.visual_projection.out_features, width, width)

        return output
