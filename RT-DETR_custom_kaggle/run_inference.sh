# -- Set necessary paths --
##############################################################################################################
CONFIG_FILE=./configs/rtdetr/rtdetr_r101vd_6x_coco_custom.yml
CKPT_PATH=./model_checkpoints/rtdetr_r101vd_6x_coco_from_paddle.pth
CLS_LIST=./data/coco_format/obj.names # list of class names to be predicted: uncomment after once training is done!, default: coco namelist
IMG_PATH=./images/people.jpeg # test image 

python infer.py --config $CONFIG_FILE \
              --ckpt $CKPT_PATH \
              --image $IMG_PATH \
            #   --class_list $CLS_LIST
