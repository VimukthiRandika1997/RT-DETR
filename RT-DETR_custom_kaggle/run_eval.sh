#!/bin/bash
set -e

export CONFIG_PATH=./configs/rtdetr/rtdetr_r101vd_6x_coco_custom.yml
export CKPT_PATH=./output/custom_rt_detr_model/checkpoint_best.pth

while getopts c:m: flag
do
    case "${flag}" in
        c) CONFIG_PATH=${OPTARG};;
        m) CKPT_PATH=${OPTARG};;
    esac
done

###** -- Evaluation -- **###
echo Starting Evaluation process...
#! Set the checkpoint path after the training is done: ./output/custom_rt_detr_model/<checkpoint_name.pth>
# -- Evaluation on Single-gpu --

# restnet101 as backbone
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c $CONFIG_PATH \
                      -t $CKPT_PATH \
                      --test-only

##############################################################################################################


# -- Evaluation on Multi-gpu --

# restnet101 as backbone
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun --nproc_per_node=4 tools/train.py -c $CONFIG_PATH \
#                                            -t $CKPT_PATH \
#                                            --test-only

##############################################################################################################
##############################################################################################################
