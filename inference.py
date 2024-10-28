# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision
from model import movingcolor_arch
from core.utils import to_tensors
import warnings
warnings.filterwarnings("ignore")
from core.utils import detect_edges
import time
import shutil
import tempfile
import subprocess
from multiprocessing import Pool
torch.backends.cudnn.enabled = False

def run_cmd(cmd):
    print(cmd)
    subprocess.call(cmd, shell=True)

def make_video(input_dir, img_fmt, video_filename, fps=25):
    cmd = 'ffmpeg -y -loglevel error -framerate %s -i "%s/%s" -vcodec libx264 -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" \"%s\"' %(fps, input_dir, img_fmt, video_filename)

    run_cmd(cmd)

def save_frames(np_array, temp_dir):
    for i, frame in enumerate(np_array):
        filename = os.path.join(temp_dir, f"{i:05d}.png")
        cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def make_video_from_np(np_array, save_path, fps=25, temp_dir_root=None):
    # Set the temporary directory to the current directory if temp_dir_root is None
    if temp_dir_root is None:
        temp_dir_root = os.getcwd()

    # Create a temporary directory in the specified root directory
    temp_dir = tempfile.mkdtemp(dir=temp_dir_root)

    try:
        # Save each frame as an image in the temporary directory
        save_frames(np_array, temp_dir)

        # Create the video using the saved images
        make_video(temp_dir, "%05d.png", save_path, round(fps))
    except Exception as e:
        print("An error occurred while creating the video: ", e)
    finally:
        # Clean up: remove the temporary directory and its contents
        try:
            shutil.rmtree(temp_dir)
        except OSError as e:
            print("An error occurred while cleaning up the temporary directory: ", e)

def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)

def resize_frames(frames, size=None):    
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
        
    return frames, process_size, out_size

def read_frame(file_path):
    frame = Image.open(file_path).convert('RGB')
    return frame

#  read frames from video
def read_frame_from_videos(frame_root, num_processes=40):
    fr_lst = sorted([os.path.join(frame_root, fr) for fr in os.listdir(frame_root)])

    with Pool(processes=num_processes) as pool:
        frames = pool.map(read_frame, fr_lst)

    size = frames[0].size if frames else (0, 0)
    return frames, size

# read frame-wise masks
def read_mask(mpath, size, mask_dilates=2, edge_mask_dir=False):
    masks_img = []
    masks_edge = []
    
    if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
        if size is None:
            masks_img = [Image.open(mpath)]
        else:
            masks_img = [Image.open(mpath).resize(size, Image.NEAREST)]
    else:  
        mnames = sorted(os.listdir(mpath))
        for mp in mnames:
            mask_img = np.uint8((cv2.imread(os.path.join(mpath, mp), 0) > 0) * 255)
            mask_img = Image.fromarray(mask_img)
            if size is not None:
                mask_img = mask_img.resize(size, Image.NEAREST)
            masks_img.append(mask_img)

    if not edge_mask_dir:  # support upload the edge mask
        for mask_img in masks_img:
            # generate edge mask
            mask_edge = detect_edges(np.array(mask_img), kernel_size=3, dilation_iteration=mask_dilates)
            masks_edge.append(Image.fromarray(mask_edge))
    else:
        print(f'Use edge mask from edge_mask_dir: {edge_mask_dir}')
        for mp in mnames:
            mask_edge = np.uint8((cv2.imread(os.path.join(edge_mask_dir, mp), 0) > 0) * 255)
            mask_edge = Image.fromarray(mask_edge)
            if size is not None:
                mask_edge = mask_edge.resize(size, Image.NEAREST)
            masks_edge.append(mask_edge)
    
    return masks_img, masks_edge

def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--video', type=str, default='datasets/davis/input_folder/bear', help='Path of the input video or image folder.')
    parser.add_argument(
        '-m', '--mask', type=str, default='datasets/davis/mask_folder/bear', help='Path of the mask(s) or mask folder.')
    parser.add_argument(
        '-d', '--video_distort', type=str, default='datasets_video/davis/distort_folder/high_brightness/bear', help='Path of the ref image.')
    parser.add_argument(
        '-e', '--edge_mask_dir', type=str, default='', help='Path of the edge mask(s) or mask folder.')
    parser.add_argument(
        '-o', '--save_root', type=str, default='results', help='Output folder. Default: results')
    parser.add_argument(
        "--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
    parser.add_argument(
        '--height', type=int, default=240, help='Height of the processing video.')
    parser.add_argument(
        '--width', type=int, default=424, help='Width of the processing video.')
    parser.add_argument(
        '--mask_dilation', type=int, default=8, help='Mask dilation for video and flow masking.')
    parser.add_argument(
        "--ref_stride", type=int, default=50, help='Stride of global reference frames.')
    parser.add_argument(
        "--neighbor_length", type=int, default=10, help='Length of local neighboring frames.')
    parser.add_argument(
        "--subvideo_length", type=int, default=5000, help='Length of sub-video for long video inference.')
    parser.add_argument(
        '--scale_h', type=float, default=1.0, help='Outpainting scale of height for video_outpainting mode.')
    parser.add_argument(
        '--scale_w', type=float, default=1.2, help='Outpainting scale of width for video_outpainting mode.')
    parser.add_argument(
        '--save_frames', default=True, help='Save output frames. Default: False')
    parser.add_argument(
        '--arch', default='MCGenerator_l1_vgg_gan', help='Architecture of the model. Default: MCGenerator')
    parser.add_argument(
        '--ckpt_path', default='experiments_model/train_MCGenerator_l1_vgg_gan/gen/latest.pth', help='Path of the checkpoint')
    parser.add_argument(
        '--fps', type=int, default=25, help='Frame per second. Default: 24')
    args = parser.parse_args()

    save_root = args.save_root
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)
    video_name = os.path.basename(args.video)

    frames, ori_size = read_frame_from_videos(args.video)
    frames_distort, size = read_frame_from_videos(args.video_distort)

    if not args.resize_ratio == 1.0:
        size = (int(args.resize_ratio * size[0]), int(args.resize_ratio * size[1]))    

    frames, size, out_size = resize_frames(frames, size)
    frames_distort, size, out_size = resize_frames(frames_distort, size)
        
    masks_img, masks_edge = read_mask(args.mask, size, mask_dilates=args.mask_dilation, edge_mask_dir=args.edge_mask_dir)
    frames = frames[:len(masks_img)] # cap the number of frames to the number of masks
    frames_distort = frames_distort[:len(masks_img)]

    w, h = size

    # for saving the masked frames or video
    masked_frame_for_save = []
    for i in range(len(frames)):
        mask_ = np.expand_dims(np.array(masks_img[i]),2).repeat(3, axis=2)/255.
        img = np.array(frames[i])
        green = np.zeros([h, w, 3]) 
        green[:,:,1] = 255
        alpha_ratio = 0.6
        # alpha_ratio = 1.0
        fuse_img = (1-alpha_ratio)*img + alpha_ratio*green
        fuse_img = mask_ * fuse_img + (1-mask_)*img
        masked_frame_for_save.append(fuse_img.astype(np.uint8))

    frames_inp = [np.array(f).astype(np.uint8) for f in frames]

    img_size = frames_inp[0].shape[:2][::-1]
    frames_tensor = to_tensors()(frames).unsqueeze(0) 
    frame_distort_tensors = to_tensors()(frames_distort).unsqueeze(0)
    masks_edge_tensor = to_tensors()(masks_edge).unsqueeze(0)
    frames_distort_masked_tensor = frame_distort_tensors * (1 - masks_edge_tensor)

    ckpt_path = args.ckpt_path
    model = getattr(movingcolor_arch,args.arch)().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
    for param in model.parameters():
        param.grad = None
    model.eval()

    with torch.no_grad():
        video_length = frames_tensor.size(1)
        print(f'\nProcessing: {video_name} [{video_length} frames]...')
        
        ori_frames = frames_inp
        comp_frames = [None] * video_length

        neighbor_stride = args.neighbor_length // 2
        if video_length > args.subvideo_length:
            if args.ref_stride > 0:
                ref_num = args.subvideo_length // args.ref_stride
            else:
                ref_num = 0
        else:
            ref_num = -1
        
        # ---- feature propagation + transformer ----
        for f in tqdm(range(0, video_length, max(1,neighbor_stride))):
            if neighbor_stride == 0: # frame by frame
                neighbor_ids = [f]
            if args.ref_stride == 0:
                ref_ids = []
            if neighbor_stride>0:
                neighbor_ids = [
                    i for i in range(max(0, f - neighbor_stride),
                                        min(video_length, f + neighbor_stride + 1))
                ]
            if args.ref_stride > 0:
                ref_ids = get_ref_index(f, neighbor_ids, video_length, args.ref_stride, ref_num)
            selected_frames = frames_tensor[:, neighbor_ids + ref_ids, :, :, :].to(device)
            selected_masked_distort = frames_distort_masked_tensor[:, neighbor_ids + ref_ids, :, :, :].to(device)
            selected_mask_edges = masks_edge_tensor[:, neighbor_ids + ref_ids, :, :, :].to(device)
            selected_pred_flows_bi, updated_masks = None, None

            l_t = len(neighbor_ids)
            x_input = torch.cat([selected_frames, selected_masked_distort, selected_mask_edges], 2)
            pred_img = model(x_input, selected_pred_flows_bi, selected_mask_edges, updated_masks, l_t)
            
            pred_img = pred_img.view(-1, 3, h, w)
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                pred_np = np.array(pred_img[i]).astype(np.uint8)
                if comp_frames[idx] is None:
                    comp_frames[idx] = pred_np
                else: 
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + pred_np.astype(np.float32) * 0.5
                    
                comp_frames[idx] = comp_frames[idx].astype(np.uint8)
            # torch.cuda.empty_cache()

    # save each frame
    if args.save_frames:
        for idx in range(video_length):
            f = comp_frames[idx]
            f = cv2.resize(f, ori_size, interpolation = cv2.INTER_CUBIC)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img_save_root = os.path.join(save_root, 'output_frames', str(idx).zfill(4)+'.png')
            imwrite(f, img_save_root)
                    
    # save videos frame
    comp_frames = [cv2.resize(f, ori_size) for f in comp_frames]
    frames_distort = [cv2.resize(np.array(f), ori_size) for f in frames_distort]
    video_results = {'output': comp_frames, 'direct_compose': frames_distort}
    # save comp_frames
    for key, value in video_results.items():
        save_path = os.path.join(save_root, key, f'{video_name}.mp4')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        make_video_from_np(value, save_path, args.fps)