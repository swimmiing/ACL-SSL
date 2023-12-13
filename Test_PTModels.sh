#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

python Test_PTModels.py \
--model_name ACL_ViT16 \
--exp_name aclifa_2gpu \
--vggss_path {put dataset directory} \
--flickr_path {put dataset directory} \
--avs_path {put dataset directory} \
--save_path {put dataset directory} \
--epochs None
