#! Change the model name to one of those two
# -- Downloading pretrained checkpoints --

CKPT_DIR=./model_checkpoints
MODEL_NAME=rtdetr_r101vd_6x_coco_from_paddle.pth # [ rtdetr_r101vd_6x_coco_from_paddle.pth, rtdetr_r101vd_2x_coco_objects365_from_paddle.pth ]

mkdir $CKPT_DIR

pushd $CKPT_DIR
    if ! [ -f $MODEL_NAME ]; then
        echo Downloading model checkpoint: $MODEL_NAME
        wget https://github.com/lyuwenyu/storage/releases/download/v0.1/${MODEL_NAME}
    fi
    echo $MODEL_NAME already exists!
popd
##############################################################################################################
##############################################################################################################



###** -- Training -- **###
# -- Training on Single-gpu --

# restnet101 as backbone
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r101vd_6x_coco_custom.yml \
                      -t $CKPT_DIR/$MODEL_NAME \

##############################################################################################################


# -- Training on Multi-gpu --

# restnet101 as backbone
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r101vd_6x_coco_custom.yml \
#                                            -t $CKPT_DIR/$MODEL_NAME \

echo '-------------------------------------------------'
echo models weights are saved in ./output/custom_rt_detr_model
echo '-------------------------------------------------'
##############################################################################################################
##############################################################################################################