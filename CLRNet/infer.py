import torch
import torchvision
import argparse
import os
from clrnet.utils.config import Config
from clrnet.models.registry import build_net
from clrnet.utils.visualization import COLORS
from clrnet.utils.net_utils import load_network
from clrnet.datasets.process.generate_lane_line import GenerateLaneLine
from clrnet.datasets.process.transforms import ToTensor
from mmcv.parallel import MMDataParallel
from PIL import Image
import cv2
import numpy as np
import time

def val_processes(sample, cfg):
    t1 = GenerateLaneLine(cfg=cfg, training=False, transforms=[dict(name='Resize',
                                                                    parameters=dict(size=dict(height=cfg.img_h, width=cfg.img_w)),
                                                                    p=1.0)])
    t2 = ToTensor(cfg=cfg, keys=['img'])
    sample = t1(sample)
    sample = t2(sample)
    return sample
    
def padding_crop(image, final_height=590, final_width=1640):
    height, width = image.shape[0], image.shape[1]
    offset = np.zeros((4,), dtype=int) # height, bottom, left, right
    pd = [0, 0, 0, 0] # height, bottom, left, right
    
    if height > final_height:
        offset[0] = final_height - height
        image = image[-offset[0]:, :, :]
    else:
        pd[0] = final_height - height
        offset[0] = pd[0]
        
    if width > final_width:
        offset[2] = (final_width - width) // 2
        offset[3] = final_width - width - offset[2]
        # -final_width + width - offset[2]
        # left = (width - final_width) // 2
        # right = width - left
        image = image[:, -offset[2]:width+offset[3], :]
    else:
        pd[2] = (final_width - width) // 2
        pd[3] = final_width - width - pd[2]
        offset[2], offset[3] = pd[2], pd[3]
        
    image = cv2.copyMakeBorder(image, pd[0], pd[1], pd[2], pd[3], cv2.BORDER_CONSTANT, value=(235, 206, 135))
    return image, offset

def get_sample(img, cfg):
    sample = {'lanes': []}
    # img = cv2.imread(file_path)
    ori_img = img.copy()
    img, _ = padding_crop(img)
    img = img[cfg.cut_height:, :, :]
    sample.update({'img': img})
    sample = val_processes(sample, cfg)
    sample['img'] = sample['img'].unsqueeze(0)
    sample.update({'ori_img': ori_img})
    return sample

def show_lanes(sample, lanes, out_file=None, width=4):
    ori_img = sample['ori_img'].copy()
    img, offset = padding_crop(ori_img)
    h, w = img.shape[0], img.shape[1]
    h_ori, w_ori = ori_img.shape[0], ori_img.shape[1]
    lanes_xys = []
    for _, lane in enumerate(lanes):
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            xys.append((x, y))
        lanes_xys.append(xys)
    lanes_xys.sort(key=lambda xys : xys[0][0])

    for idx, xys in enumerate(lanes_xys):
        for i in range(1, len(xys)):
            cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)
    
    # 移除padding
    pd = np.maximum(offset, 0)
    img = img[pd[0] : h - pd[1], pd[2] : w - pd[3], :]
    h, w = img.shape[0], img.shape[1]
    
    # 将处理后的图片嵌入到原图中
    offset = np.abs(np.minimum(offset, 0))
    ori_img[offset[0] : h_ori - offset[1], offset[2] : w_ori - offset[3], :] = img
    img = ori_img

    if out_file:
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        cv2.imwrite(out_file, img)
    return img
    
def read_image_to_tensor(file_path, height=320, width=800) -> torch.Tensor:
    img = Image.open(file_path)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((height, width)),
        torchvision.transforms.ToTensor()
    ])
    img = transforms(img)
    return img

def inference_images(model, cfg, images_dir, output_dir):
    model.eval()
    images = os.listdir(images_dir)
    images.sort()
    for img in images:
        img = cv2.imread(os.path.join(images_dir, img))
        sample = get_sample(img, cfg)
        with torch.no_grad():
            output = model(sample)
            output = model.module.heads.get_lanes(output)
        lanes = [lane.to_array(cfg) for lane in output[0]]
        output_path = os.path.join(output_dir, img)
        show_lanes(sample, lanes, out_file=output_path)
        
def inference_single_image(img, model, cfg):
    sample = get_sample(img, cfg)
    with torch.no_grad():
        output = model(sample)
        output = model.module.heads.get_lanes(output)
    lanes = [lane.to_array(cfg) for lane in output[0]]
    return show_lanes(sample, lanes)

def inference_video(model, cfg, video_path, output_dir, filename):
    model.eval()
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output = cv2.VideoWriter(os.path.join(output_dir, filename), cv2.VideoWriter_fourcc(*'XVID'),
                             fps, video_size)
    images_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame = inference_single_image(frame, model, cfg)
        output.write(frame)
    video.release()
    output.release()
    
def build_model(config_path, weights_path, use_cuda=True, training=False):
    cfg = Config.fromfile(config_path)
    cfg.load_from = weights_path
    cfg.resume_from, cfg.finetune_from = None, None
    cfg.view = False
    cfg.seed = 0
    cfg.gpus = 1
    
    model = build_net(cfg)
    model = MMDataParallel(model, device_ids=range(cfg.gpus))
    resume_model(model, cfg)
    if use_cuda:
        model = model.cuda()
    if not training:
        model.eval()
    return model, cfg
        
def main():
    # image_path = '/opt/data/private/hyy/datasets/CULane/driver_100_30frame/05251517_0433.MP4/00000.jpg'
    video_path = '/opt/data/private/hyy/projects/CLRNet/video_test/dubai_day_1080p.mp4'
    
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(gpu) for gpu in args.gpus)
    cfg = Config.fromfile(args.config)
    
    cfg.gpus = len(args.gpus)
    cfg.load_from = args.load_from
    cfg.resume_from = args.resume_from
    cfg.finetune_from = args.finetune_from
    cfg.view = args.view
    cfg.seed = args.seed
    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs

    model = build_net(cfg)
    model = MMDataParallel(model, device_ids=range(cfg.gpus))
    resume_model(model, cfg)
    model = model.cuda()
    model.eval()
    
    # print(type(model))
    # print(model)
    # img = read_image_to_tensor(image_path).cuda()
    # img = img.unsqueeze(0)
    # img = [img]
    # img = {'img': img}
    # img = get_sample(image_path, cfg)
    # with torch.no_grad():
    #     output = model(img)
    #     # print('output', output)
    #     output = model.module.heads.get_lanes(output)
    # # print(len(output))
    # lanes = [lane.to_array(cfg) for lane in output[0]]
    # # print(lanes)
    # image = cv2.imread(image_path)
    # imshow_lanes(image, lanes, out_file='./vis.png')
    
    start = time.time()
    inference_video(model, cfg, video_path, output_dir='video_test', filename='test_r18.avi')
    end = time.time()
    print(f'Time: {end - start:.5f}s')
    
def resume_model(model, cfg):
    if not cfg.load_from and not cfg.finetune_from:
        return
    load_network(model, cfg.load_from, finetune_from=cfg.finetune_from)       
    
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dirs',
                        type=str,
                        default=None,
                        help='work dirs')
    parser.add_argument('--load_from',
                        default=None,
                        help='the checkpoint file to load from')
    parser.add_argument('--resume_from',
            default=None,
            help='the checkpoint file to resume from')
    parser.add_argument('--finetune_from',
            default=None,
            help='the checkpoint file to resume from')
    parser.add_argument('--view', action='store_true', help='whether to view')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test',
        action='store_true',
        help='whether to test the checkpoint on testing set')
    parser.add_argument('--gpus', nargs='+', type=int, default='0')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()