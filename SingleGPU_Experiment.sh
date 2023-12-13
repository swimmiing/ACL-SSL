#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

python Train_ACL.py \
--model_name ACL_ViT16 \
--exp_name aclifa_1gpu \
--train_config Exp_ACL_v1 \
--vggss_path {put dataset directory} \
--flickr_path {put dataset directory} \
--avs_path {put dataset directory} \
--save_path {put logging directory}
