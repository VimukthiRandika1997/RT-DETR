import torch
from torch import nn
from torchvision.transforms import transforms
from PIL import Image, ImageDraw
from src.core import YAMLConfig
import argparse
from pathlib import Path
import time
import os
import shutil
import contextlib
import json



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
        with torch.no_grad():
            outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/rtdetr/rtdetr_r101vd_6x_coco.yml", help="model config filepath")
    parser.add_argument("--ckpt",  help="model checkpoint filepath")
    parser.add_argument("--image_dir", help="image filepath")
    parser.add_argument("--class_list", default='./coco_class_list.txt' , help="filepath for list of classes to predicted")
    parser.add_argument("--thrh", default=0.5, help="threshold value for filtering")
    parser.add_argument("--output_dir", default="output", help="saving output path")
    parser.add_argument("--viz", action="store_true", help="whether to save visualization of predictions")
    parser.add_argument("--results", action="store_true", help="whether to save json dictionary of predictions")
    parser.add_argument("--device", default="cuda")

    return parser


def _remove_existing_file(path):
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)
    

def _remove_existing_folder(path):
    if os.path.exists(path): shutil.rmtree(path)
    os.makedirs(path)


def _save_json_inference_files(json_info, path):
    with open(path, 'w') as f:
        json.dump(json_info, f, indent=4)


def _filter_thrh(scores, labels, boxes, thrh=0.5):
    # prob dist over label map
    scr = scores[0]
    # label_idx
    lab = labels[0][scr > thrh]
    # bbox
    box = boxes[0][scr > thrh]
    # prob 
    prob = scr[scr > thrh]

    return lab, box, prob


def img_inference(img_path, model, device):
    '''Image inference: one forward pass'''

    img_path = Path(img_path)
    reader = ImageReader(resize=640) # model expected size: 640 x 640 for resnet101 based model
    img =reader(img_path).to(device)
    size = torch.tensor([[img.shape[2], img.shape[3]]]).to(device)

    start = time.time()
    output = model(img, size)
    print(f"Inference time： {time.time() - start:.4f}s")

    return output, reader


def viz_img(output_dir, model_output, reader, img_path, label_map, thrh):
    '''Visualize the predictions for single image'''

    img = reader.pil_img
    draw = ImageDraw.Draw(img)

    labels, boxes, scores = model_output
    print('## Label map: ', labels.detach().cpu().shape[-1])
    print('## No.of queries(maximum no.of objects can be detected): ', boxes.shape)

    lab, box, prob = _filter_thrh(scores, labels, boxes, thrh)

    print('## no.of detected bboxes: ', box.shape[0])
    print('## detected labels: ', lab)
    for b, l, p in zip(box, lab, prob):
        b = b.detach().cpu()
        # print(np.array(b), label_map[l.item()], p.item())
        draw.rectangle(list(b), outline='red', )
        draw.text((b[0], b[1]), text= 'label ' + str(label_map[l.item()]), fill='blue', )
        draw.text((b[0], b[1] + 10), text= 'prob ' + str(p.item()), fill='green', )

    # save detections to an image
    base_path = os.path.join(output_dir, 'inference_viz_resized')
    _remove_existing_folder(base_path)
    save_path = Path(base_path) / ('resized_' + img_path.name)
    _remove_existing_file(save_path)
    img.save(save_path)
    # img.show()
    print(f"Image is saved at: {save_path}")


def convert_to_relative_coordinates(coordinates, resized_img_size: tuple, org_img_size: tuple):
    '''Convert top-left to darknet yolov4 format'''

    obj = {}
    obj["width"] = abs(coordinates[0] - coordinates[2]) 
    obj["height"] = abs(coordinates[1] - coordinates[3])
    obj["center_x"] = coordinates[0] + obj["width"] / 2
    obj["center_y"] = coordinates[1] + obj["height"] / 2

    # rescale to original image size
    scaled_width = org_img_size[0] / resized_img_size[0]
    scaled_height = org_img_size[1] / resized_img_size[1]

    # get the relative width and height relative to original image
    obj['width'] = (obj['width'] * scaled_width) / org_img_size[0]
    obj['height'] = (obj['height'] * scaled_height) / org_img_size[1]
    obj["center_x"] = (obj["center_x"] * scaled_width) / org_img_size[0]
    obj["center_y"] = (obj["center_y"] * scaled_height) / org_img_size[1]

    return obj 


def draw_json_output(output_dir, detections, img_path):
    '''Visualize detections on original image: detections are rescaled'''

    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)

    for obj in detections['objects']:
        b = obj['relative_coordinates']
        x1 =  (b['center_x'] - (b['width'] / 2 )) * img.size[0]
        y1 =  (b['center_y'] - (b['height'] / 2 )) * img.size[1]
        x2 =  x1 + (b['width'] * img.size[0])
        y2 =  y1 + (b['height'] * img.size[1])

        draw.rectangle([x1, y1, x2, y2], outline='red')
        draw.text((x1, y1), text= 'label ' + obj['name'], fill='blue', )
        draw.text((x1, y1 + 10), text= 'prob ' + str(obj['confidence']), fill='green', )

    # save detections to an image    
    save_path = Path(output_dir) / img_path.name
    img.save(save_path)
    # img.show()
    print(f"Image is saved at: {save_path}")


def create_json_output(model_output, reader, img_path, label_map, thrh):
    '''Save inference result as a json output'''

    labels, boxes, scores = model_output

    output_dict = {}
    objects_arr = []

    # filter detected bboxes based on threshold value: default is set to 0.5
    lab, box, prob = _filter_thrh(scores, labels, boxes, thrh)

    # get the filename
    output_dict['filename'] = str(img_path)

    img = reader(img_path)
    org_img = Image.open(img_path)

    # loop through detected objects
    for b, l, p in zip(box, lab, prob):
        b = b.detach().cpu().tolist()
        l = l.detach().cpu().item()
        p = p.detach().cpu().item()

        obj_dict = {}
        obj_dict["class_id"] = l
        obj_dict["name"] = label_map[l]
        obj_dict["confidence"] = p
        obj_dict["relative_coordinates"]  = convert_to_relative_coordinates(coordinates=b, 
                                                                            resized_img_size=(img.shape[2], img.shape[3]), 
                                                                            org_img_size=(org_img.size[0], org_img.size[1]))

        objects_arr.append(obj_dict)
    
    output_dict["objects"] = objects_arr

    return output_dict


def main(args):
    print('## Args: ', args, '\n')

    with open(args.class_list, 'r') as f:
        label_map = f.readlines()
    label_map = [name.strip() for name in label_map]


    device = torch.device(args.device)
    model = Model(confg=args.config, ckpt=args.ckpt)
    model.to(device=device)

    if args.results:
        base_inference_path = os.path.join(args.output_dir, 'inference_files')
        _remove_existing_folder(base_inference_path)
    if args.viz:
        base_viz_path = os.path.join(args.output_dir, 'inference_viz')
        _remove_existing_folder(base_viz_path)

    # get list of images to be inferenced
    image_filepaths = [os.path.join(args.image_dir, filename) for filename in os.listdir(args.image_dir)]
    # loop through each image
    frame_id = 0
    detection_list = []
    for image_filepath in image_filepaths:
        output, reader = img_inference(image_filepath, model, device)
        detections = create_json_output(output, reader, Path(image_filepath), label_map, float(args.thrh))

        if args.viz:
            # viz_img(args.output_dir, output, reader, Path(args.image), label_map)
            draw_json_output(base_viz_path, detections, Path(image_filepath))

        if args.results:
            filename = 'inference_' + image_filepath.split('/')[-1].split('.')[0] + '.json'
            save_path = os.path.join(base_inference_path, filename)
            detections['frame_id'] = frame_id 
            _save_json_inference_files(detections, save_path)

            detection_list.append(detections)
        frame_id += 1
    
    # save an inference file for all the images
    _save_json_inference_files(detection_list, os.path.join(args.output_dir, 'inference_all.json'))

if __name__ == "__main__":
    main(get_argparser().parse_args())
    

