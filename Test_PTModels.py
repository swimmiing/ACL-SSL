import os

from VGGSS.VGGSS_Dataset import VGGSSDataset, ExtendVGGSSDataset
from Flickr.Flickr_Dataset import FlickrDataset, ExtendFlickrDataset
from AVSBench.AVSBench_Dataset import AVSBenchDataset

from Eval import eval_vggss_agg, eval_flickr_agg, eval_avsbench_agg, eval_exvggss_agg, eval_exflickr_agg

from modules.models import *
from modules.arg_utils import int_or_int_list_or_none
from typing import Union, List, Any


@torch.no_grad()
def main(
        model_name: str,
        exp_name: str,
        epochs: Union[int, List[Union[int, None]]],
        data_path_dict: dict,
        save_path: str) -> None:
    """
    Main function for evaluating sound source localization model.

    Args:
        model_name (str): The name of the model, corresponding to the model config file in './config/model'.
        exp_name (str): The postfix for saving the experiment.
        epochs (Union[int, List[Union[int, None]]]): List of epochs to evaluate.
        data_path_dict (dict): The directory for dataset.
        save_path (str): The directory for saving evaluation results.
    """

    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')

    model_exp_name = f'{model_name}_{exp_name}' if exp_name != "" else model_name

    print(f"Exp_name: {model_exp_name}")

    for epoch in epochs:
        # Get model
        model_conf_file = f'./config/model/{model_name}.yaml'
        model = ACL(model_conf_file, device)
        model.train(False)

        # Load model
        postfix = str(epoch) if epoch is not None else 'best'
        model_dir = os.path.join(save_path, 'Train_record', model_exp_name, f'Param_{postfix}.pth')
        model.load(model_dir)

        # Set directory
        viz_dir_template = os.path.join(save_path, 'Visual_results', '{}', model_exp_name, f'epoch{postfix}')
        tensorboard_path = os.path.join(save_path, 'Train_record', model_exp_name)

        # Get dataloader
        exvggss_dataset = ExtendVGGSSDataset(data_path_dict['vggss'], input_resolution=352)
        exvggss_dataloader = torch.utils.data.DataLoader(exvggss_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                         pin_memory=True, drop_last=False)

        exflickr_dataset = ExtendFlickrDataset(data_path_dict['flickr'], input_resolution=352)
        exflickr_dataloader = torch.utils.data.DataLoader(exflickr_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                          pin_memory=True, drop_last=False)

        flickr_dataset = FlickrDataset(data_path_dict['flickr'], 'flickr_test', is_train=False, input_resolution=352)
        flickr_dataloader = torch.utils.data.DataLoader(flickr_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                        pin_memory=True, drop_last=False)

        vggss_dataset = VGGSSDataset(data_path_dict['vggss'], 'vggss_test', is_train=False, input_resolution=352)
        vggss_dataloader = torch.utils.data.DataLoader(vggss_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                       pin_memory=True, drop_last=False)

        avss4_dataset = AVSBenchDataset(data_path_dict['avs'], 'avs1_s4_test', is_train=False, input_resolution=352)
        avss4_dataloader = torch.utils.data.DataLoader(avss4_dataset, batch_size=5, shuffle=False, num_workers=1,
                                                       pin_memory=True, drop_last=False)

        avsms3_dataset = AVSBenchDataset(data_path_dict['avs'], 'avs1_ms3_test', is_train=False, input_resolution=352)
        avsms3_dataloader = torch.utils.data.DataLoader(avsms3_dataset, batch_size=5, shuffle=False, num_workers=1,
                                                        pin_memory=True, drop_last=False)

        # Evaluate
        eval_exflickr_agg(model, exflickr_dataloader, viz_dir_template.format('exflickr'))
        eval_exvggss_agg(model, exvggss_dataloader, viz_dir_template.format('exvggss'))
        eval_flickr_agg(model, flickr_dataloader, viz_dir_template.format('flickr'), tensorboard_path=tensorboard_path)
        eval_vggss_agg(model, vggss_dataloader, viz_dir_template.format('vggss'), tensorboard_path=tensorboard_path)
        eval_avsbench_agg(model, avss4_dataloader, viz_dir_template.format('s4'), tensorboard_path=tensorboard_path)
        eval_avsbench_agg(model, avsms3_dataloader, viz_dir_template.format('ms3'), tensorboard_path=tensorboard_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ACL_ViT16', help='Use model config file name')
    parser.add_argument('--exp_name', type=str, default='aclifa_2gpu', help='postfix for save experiment')
    parser.add_argument('--epochs', type=int_or_int_list_or_none, default=[None], help='epochs ([None] for released)')
    parser.add_argument('--vggss_path', type=str, default='', help='VGGSS dataset directory')
    parser.add_argument('--flickr_path', type=str, default='', help='Flickr dataset directory')
    parser.add_argument('--avs_path', type=str, default='', help='AVSBench dataset directory')
    parser.add_argument('--save_path', type=str, default='', help='Checkpoints directory')
    args = parser.parse_args()

    data_dict = {'vggss': args.vggss_path,
                 'flickr': args.flickr_path,
                 'avs': args.avs_path}

    # Run example
    main(args.model_name, args.exp_name, args.epochs, data_dict, args.save_path)
