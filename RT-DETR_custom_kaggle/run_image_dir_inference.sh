# -- Set necessary paths --
# Default is set to `coco` pretrained model weights
##############################################################################################################
CONFIG_FILE=./configs/rtdetr/rtdetr_r101vd_6x_coco.yml
CKPT_PATH=./model_checkpoints/rtdetr_r101vd_6x_coco_from_paddle.pth
CLS_LIST=./data/coco_format/obj.names # list of class names to be predicted: set this after once training is done!, default: coco namelist
IMG_PATH=./images/ 
THRH_VALUE=0.5

python img_dir_infer.py --config $CONFIG_FILE \
                        --ckpt $CKPT_PATH \
                        --image $IMG_PATH \
                        --thrh 0.5 \
                        --results --viz  
                        # --class_list $CLS_LIST \