import torch
import os
import sys

import time
import datetime
import yaml
import shutil
import argparse

from tqdm import tqdm
from util import get_prompt_template, fix_seed, seed_worker
from VGGSS.VGGSS_Dataset import VGGSSDataset, ExtendVGGSSDataset
from Flickr.Flickr_Dataset import FlickrDataset, ExtendFlickrDataset
from AVSBench.AVSBench_Dataset import AVSBenchDataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from importlib import import_module
from Eval import eval_vggss_agg, eval_avsbench_agg, eval_flickr_agg, eval_exvggss_agg, eval_exflickr_agg
from contextlib import nullcontext

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


def main(model_name, exp_name, train_config_name, data_path_dict, save_path):
    """
    Main function for training an image compression model.

    Args:
        model_name (str): The name of the compression model, corresponding to the model config file in './config/model'.
        exp_name (str): The postfix for saving the experiment.
        train_config_name (str): The name of the training configuration, corresponding to the files in './config/train'.
        data_path_dict (dict): The directory for dataset.
        save_path (str): The directory where training results will be saved.

    Returns:
        None
    """
    USE_CUDA = torch.cuda.is_available()

    # Check the number of GPUs for training
    num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    use_ddp = True if num_gpus > 1 else False

    rank = 0 if not use_ddp else None

    if use_ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=9000))
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        world_size = dist.get_world_size()
        print(f'World size: {world_size}') if rank == 0 else None

    device = torch.cuda.current_device() if USE_CUDA else torch.device('cpu')
    print(f'Device: {device} is used\n')

    model_exp_name = f'{model_name}_{exp_name}' if exp_name != "" else model_name

    ''' Set logging dir '''
    tensorboard_path = os.path.join(save_path, 'Train_record', model_exp_name, "tensorboard")

    ''' Get train configure '''
    train_conf_file = f'./config/train/{train_config_name}.yaml'
    with open(train_conf_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args = argparse.Namespace(**config['common'])
        args.optim = config['optim_conf'][config['optimizer']]
        if rank == 0:
            print(vars(args))

    ''' Fix random seed'''
    fix_seed(args.seed)

    ''' Tensorboard '''
    writer = SummaryWriter(tensorboard_path)
    print(f"\nSave dir: {os.path.join(save_path, 'Train_record', model_exp_name)}\n") if rank == 0 else None

    ''' Get model '''
    model_conf_file = f'./config/model/{model_name}.yaml'
    model = getattr(import_module('modules.models'), config['model'])(model_conf_file, device)
    if rank == 0:
        print(f"Model '{model.__class__.__name__}' with configure file '{model_name}' is loaded")
        print(f"Loaded model details: {vars(model.args.model)}\n")

    training_consumed_sec = 0

    ''' Get dataloader '''
    if args.train_data == 'vggss':
        # Get Train Dataloader (VGGSS)
        train_dataset = VGGSSDataset(data_path_dict['vggss'], 'vggss_144k', is_train=True,
                                     input_resolution=args.input_resolution)
    elif args.train_data == 'vggss_heard':
        # Get Train Dataloader (VGGSS heard setup)
        train_dataset = VGGSSDataset(data_path_dict['vggss'], 'vggss_heard', is_train=True,
                                     input_resolution=args.input_resolution)
    else:
        # Get Train Dataloader (Flickr)
        train_dataset = FlickrDataset(data_path_dict['flickr'], 'flickr_144k', is_train=True,
                                      input_resolution=args.input_resolution)

    ''' Create DistributedSampler '''
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                                   num_workers=args.num_workers, pin_memory=False, drop_last=True,
                                                   worker_init_fn=seed_worker)

    # Get Test Dataloader (VGGSS)
    vggss_dataset = VGGSSDataset(data_path_dict['vggss'], 'vggss_test', is_train=False,
                                 input_resolution=args.input_resolution)
    vggss_dataloader = torch.utils.data.DataLoader(vggss_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                   pin_memory=False, drop_last=False)

    if args.train_data == 'vggss_heard':
        # Get Test Dataloader (VGGSS)
        heard_dataset = VGGSSDataset(data_path_dict['vggss'], 'vggss_heard_test', is_train=False,
                                     input_resolution=args.input_resolution)
        heard_dataloader = torch.utils.data.DataLoader(heard_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                       pin_memory=False, drop_last=False)
        # Get Test Dataloader (VGGSS)
        unheard_dataset = VGGSSDataset(data_path_dict['vggss'], 'vggss_unheard_test', is_train=False,
                                       input_resolution=args.input_resolution)
        unheard_dataloader = torch.utils.data.DataLoader(unheard_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                         pin_memory=False, drop_last=False)

    # Get Test Dataloader (Flickr)
    flickr_dataset = FlickrDataset(data_path_dict['flickr'], 'flickr_test', is_train=False,
                                   input_resolution=args.input_resolution)
    flickr_dataloader = torch.utils.data.DataLoader(flickr_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                    pin_memory=False, drop_last=False)

    # Get Test Dataloader (Extended VGGSS)
    exvggss_dataset = ExtendVGGSSDataset(data_path_dict['vggss'], input_resolution=352)
    exvggss_dataloader = torch.utils.data.DataLoader(exvggss_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                     pin_memory=False, drop_last=False)

    # Get Test Dataloader (Extended Flickr)
    exflickr_dataset = ExtendFlickrDataset(data_path_dict['flickr'], input_resolution=352)
    exflickr_dataloader = torch.utils.data.DataLoader(exflickr_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                      pin_memory=False, drop_last=False)

    # Get Test Dataloader (AVS)
    avss4_dataset = AVSBenchDataset(data_path_dict['avs'], 'avs1_s4_test', is_train=False,
                                    input_resolution=args.input_resolution)
    avss4_dataloader = torch.utils.data.DataLoader(avss4_dataset, batch_size=5, shuffle=False, num_workers=1,
                                                   pin_memory=False, drop_last=False)

    avsms3_dataset = AVSBenchDataset(data_path_dict['avs'], 'avs1_ms3_test', is_train=False,
                                     input_resolution=args.input_resolution)
    avsms3_dataloader = torch.utils.data.DataLoader(avsms3_dataset, batch_size=5, shuffle=False, num_workers=1,
                                                    pin_memory=False, drop_last=False)

    ''' Optimizer '''
    module_path, module_name = args.optim.pop('module_path'), args.optim.pop('module_name')
    optimizer = getattr(import_module(module_path), module_name)(model.parameters(), **args.optim)

    ''' Scheduler '''
    scheduler = None
    if config['scheduler']:
        print(f"Scheduler: {config['scheduler']}")
        args.sched = config['sched_conf'][config['scheduler']]
        module_path, module_name = args.sched.pop('module_path'), args.sched.pop('module_name')
        scheduler = getattr(import_module(module_path), module_name)(optimizer,
                                                                     T_max=args.epoch * len(train_dataloader),
                                                                     eta_min=args.sched['eta_ratio'] * args.optim['lr'])

    ''' Autocast '''
    if config['amp']:
        if rank == 0:
            print('Using AMP')
        autocast_fn = autocast
        scaler = GradScaler()
    else:
        autocast_fn, scaler = nullcontext, None

    ''' Make distributed data parallel module '''
    model = DistributedDataParallel(model, device_ids=[device], output_device=device) if use_ddp else model
    module = model.module if isinstance(model, DistributedDataParallel) else model

    best_pth_dict = {'epoch': 0, 'best_AUC': 0.0}

    ''' Train Loop '''
    for epoch in range(args.epoch):
        module.train(True)

        total_loss_per_epopch = 0.0
        loss_add_count = 0.0

        loss_dict = {}
        loss_per_epoch_dict = {loss_name: 0.0 for loss_name in args.loss}

        if rank == 0:
            train_start_time_per_epoch = time.time()

        pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch}...", disable=(rank != 0))
        sampler.set_epoch(epoch) if use_ddp else None
        for step, data in enumerate(pbar):
            images, audios, labels = data['images'], data['audios'], data['labels']

            images = images.half()
            prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

            with autocast_fn():
                # Train step
                placeholder_tokens = module.get_placeholder_token(prompt_template.replace('{}', ''))
                placeholder_tokens = placeholder_tokens.repeat((train_dataloader.batch_size, 1))
                audio_driven_embedding = module.encode_audio(audios.to(module.device), placeholder_tokens,
                                                             text_pos_at_prompt, prompt_length).half()

                out_dict = module(images.to(module.device), audio_driven_embedding, 352)

                loss_args = {'pred_emb': audio_driven_embedding, **out_dict}

                for j, loss_name in enumerate(args.loss):
                    loss_dict[loss_name] = getattr(import_module('loss_utils'), loss_name)(**loss_args) * args.loss_w[j]
                    loss_per_epoch_dict[loss_name] += loss_dict[loss_name]
                loss = torch.sum(torch.stack(list(loss_dict.values())))

                if rank == 0:
                    if torch.isnan(loss) or torch.isinf(loss):
                        # skip if loss is nan
                        print('************Training stopped due to inf/nan loss.************')
                        sys.exit(-1)

                extra_loss = 0
                loss += extra_loss

            total_loss_per_epopch += loss.item()
            loss_add_count += 1.0
            optimizer.zero_grad()

            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if scheduler is not None:
                scheduler.step()

            avr_loss = total_loss_per_epopch / loss_add_count
            if rank == 0:
                pbar.set_description(f"Training Epoch {epoch}, Loss = {round(avr_loss, 5)}")

        dist.barrier()

        if rank == 0:
            loss_per_epoch_dict = dict(
                (loss_name, loss / loss_add_count) for loss_name, loss in loss_per_epoch_dict.items())
            training_consumed_sec += (time.time() - train_start_time_per_epoch)

            writer.add_scalars('train/overall', {'loss': total_loss_per_epopch / loss_add_count}, epoch)
            writer.add_scalars('train/loss', loss_per_epoch_dict, epoch)
            for i, param in enumerate(optimizer.param_groups):
                writer.add_scalars('train/lr', {f'param{i}': optimizer.param_groups[i]['lr']}, epoch)

            ''' Evaluate '''
            module.train(False)

            with torch.no_grad():
                viz_dir_template = os.path.join(save_path, 'Visual_results', '{}', model_exp_name, f'epoch{epoch}')

                if args.train_data == 'vggss_heard':
                    result_dict = eval_vggss_agg(module, heard_dataloader, viz_dir_template.format('vggss_heard'),
                                                 epoch, tensorboard_path=tensorboard_path)
                    eval_vggss_agg(module, unheard_dataloader, viz_dir_template.format('vggss_unheard'), epoch,
                                   tensorboard_path=tensorboard_path)
                else:
                    result_dict = eval_vggss_agg(module, vggss_dataloader, viz_dir_template.format('vggss'), epoch,
                                                 tensorboard_path=tensorboard_path)
                    eval_flickr_agg(module, flickr_dataloader, viz_dir_template.format('flickr'), epoch,
                                    tensorboard_path=tensorboard_path)
                    eval_avsbench_agg(module, avss4_dataloader, viz_dir_template.format('s4'), epoch,
                                      tensorboard_path=tensorboard_path)
                    eval_avsbench_agg(module, avsms3_dataloader, viz_dir_template.format('ms3'), epoch,
                                      tensorboard_path=tensorboard_path)
                    eval_exvggss_agg(module, exvggss_dataloader, viz_dir_template.format('exvggss'), epoch,
                                     tensorboard_path=tensorboard_path)
                    eval_exflickr_agg(module, exflickr_dataloader, viz_dir_template.format('exflickr'), epoch,
                                      tensorboard_path=tensorboard_path)

            save_dir = os.path.join(save_path, 'Train_record', model_exp_name, f'Param_{str(epoch)}.pth')
            module.save(save_dir)
            module.train(True)

            if best_pth_dict['best_AUC'] < result_dict['best_AUC']:
                best_pth_dict = result_dict
                shutil.copyfile(save_dir, os.path.join(save_path, 'Train_record', model_exp_name, f'Param_best.pth'))

    writer.close()

    if rank == 0:
        result_list = str(datetime.timedelta(seconds=training_consumed_sec)).split(".")
        print("Training time :", result_list[0])
        print(f"Best epoch: {best_pth_dict['epoch']}")

    dist.destroy_process_group() if use_ddp else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument('--model_name', type=str, default='', help='Use model config file name')
    parser.add_argument('--train_config', type=str, default='', help='Use train config file name')
    parser.add_argument('--exp_name', type=str, default='', help='postfix for save experiment')
    parser.add_argument('--save_path', type=str, default='', help='Save path for model and results')
    parser.add_argument('--vggss_path', type=str, default='', help='VGGSS dataset directory')
    parser.add_argument('--flickr_path', type=str, default='', help='Flickr dataset directory')
    parser.add_argument('--avs_path', type=str, default='', help='AVSBench dataset directory')

    args = parser.parse_args()

    data_path = {'vggss': args.vggss_data_path,
                 'flickr': args.flickr_data_path,
                 'avs': args.avs_data_path}

    # Run example
    main(args.model_name, args.exp_name, args.train_config, data_path, args.save_path)
