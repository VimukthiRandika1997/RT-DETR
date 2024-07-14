import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image, ImageDraw
# import sys
# sys.path.append("../")
from src.core import YAMLConfig
import argparse
from pathlib import Path
import time
import os



class ImageReader:
    def __init__(self, resize=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Compose([
            # transforms.Resize((resize, resize)) if isinstance(resize, int) else transforms.Resize(
            #     (resize[0], resize[1])),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ])
        self.resize = resize
        self.pil_img = None   

    def __call__(self, image_path, *args, **kwargs):

        self.pil_img = Image.open(image_path).convert('RGB').resize((self.resize, self.resize))
        return self.transform(self.pil_img).unsqueeze(0)




class Model(nn.Module):
    def __init__(self, confg=None, ckpt="") -> None:
        super().__init__()
        self.cfg = YAMLConfig(confg, resume=ckpt)
        # print('## Model Architecture \n', self.cfg.model)

        if ckpt:
            checkpoint = torch.load(ckpt, map_location='cuda') 
            if 'ema' in checkpoint:
                state = checkpoint['ema']['module']
            else:
                state = checkpoint['model']
        else:
            raise AttributeError('only support resume to load model.state_dict by now.')

        # NOTE load train mode state -> convert to deploy mode
        self.cfg.model.load_state_dict(state)

        self.model = self.cfg.model.deploy()
        self.postprocessor = self.cfg.postprocessor.deploy()
        
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)



def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/rtdetr/rtdetr_r101vd_6x_coco.yml", help="model config filepath")
    parser.add_argument("--ckpt",  help="model checkpoint filepath")
    parser.add_argument("--image", help="image filepath")
    parser.add_argument("--class_list", default='./coco_class_list.txt' , help="filepath for list of classes to predicted")
    parser.add_argument("--output_dir", default="output", help="saving output path")
    parser.add_argument("--device", default="cuda")

    return parser


def main(args):
    print('## Args: ', args, '\n')

    with open(args.class_list, 'r') as f:
        label_map = f.readlines()
    label_map = [name.strip() for name in label_map]

    img_path = Path(args.image)
    device = torch.device(args.device)
    reader = ImageReader(resize=640)
    model = Model(confg=args.config, ckpt=args.ckpt)
    model.to(device=device)

    img =reader(img_path).to(device)
    size = torch.tensor([[img.shape[2], img.shape[3]]]).to(device)
    start = time.time()
    output = model(img, size)
    print(f"Time diff：{time.time() - start:.4f}s")
    labels, boxes, scores = output
    im = reader.pil_img
    draw = ImageDraw.Draw(im)
    # threshold for bbox_score and objectnessf
    thrh = 0.5

    print('## Label map: ', labels.detach().cpu().shape[-1])
    print('## No.of queries(maximum no.of objects can be detected): ', boxes.shape)

    # prob dist over label map
    scr = scores[0]
    # label_idx
    lab = labels[0][scr > thrh]
    # bbox
    box = boxes[0][scr > thrh]
    # prob 
    prob = scr[scr > thrh]
    print('## no.of detected bboxes: ', box.shape[0])
    print('## detected labels: ', lab)
    for b, i, p in zip(box, lab, prob):
        b = b.detach().cpu()
        draw.rectangle(list(b), outline='red', )
        draw.text((b[0], b[1]), text= 'label ' + str(label_map[i.item()]), fill='blue', )
        draw.text((b[0], b[1] + 10), text= 'prob ' + str(p.item()), fill='green', )

    # save detections to an image
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = Path(args.output_dir) / img_path.name
    im.save(save_path)
    im.show()
    print(f"outputs are saved at:{save_path}")

if __name__ == "__main__":
    main(get_argparser().parse_args())
    

