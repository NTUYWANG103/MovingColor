import os
import io
import cv2
import random
import numpy as np
from PIL import Image, ImageOps
import zipfile
import math

import torch
import matplotlib
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib import pyplot as plt
from torchvision import transforms

# matplotlib.use('agg')

# ###########################################################################
# Directory IO
# ###########################################################################

def detect_edges(mask, kernel_size=3, dilation_iteration=0, erosion_iteration=0, threshold1=0, threshold2=138):
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edge_mask = cv2.Canny(blurred.astype(np.uint8), threshold1=threshold1, threshold2=threshold2)

    # Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    # dilate (zoom) the edge mask
    if dilation_iteration > 0:
        edge_mask = cv2.dilate(edge_mask, kernel, iterations=dilation_iteration)
    if erosion_iteration > 0:
        edge_mask = cv2.erode(edge_mask, kernel, iterations=erosion_iteration)

    return edge_mask

def adaptive_detect_edge(mask, dilation_ratio=1.0, non_linear_exponent=1.0, threshold1=0, threshold2=138):
    # Detect edges using Canny
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edge_mask = cv2.Canny(blurred.astype(np.uint8), threshold1=threshold1, threshold2=threshold2)

    # Calculate area (number of white pixels) and perimeter (edge length)
    area = np.sum(mask > 0)
    perimeter = np.sum(edge_mask > 0)

    # Avoid division by zero in case there is no perimeter
    if perimeter == 0:
        perimeter = 1

    # Determine the dilation kernel size based on a non-linear relationship
    # of the area-perimeter ratio, adjusted by the dilation ratio and non-linear exponent
    kernel_size = int((area / perimeter)**non_linear_exponent * dilation_ratio)
    kernel_size = max(1, kernel_size)  # Ensure kernel size is at least 1

    # Create a kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Dilate the edge mask
    edge_mask_dilated = cv2.dilate(edge_mask, kernel)

    return edge_mask_dilated

def read_dirnames_under_root(root_dir):
    dirnames = [
        name for i, name in enumerate(sorted(os.listdir(root_dir)))
        if os.path.isdir(os.path.join(root_dir, name))
    ]
    print(f'Reading directories under {root_dir}, num: {len(dirnames)}')
    return dirnames

class TrainZipReader(object):
    file_dict = dict()

    def __init__(self):
        super(TrainZipReader, self).__init__()

    @staticmethod
    def build_file_dict(path):
        file_dict = TrainZipReader.file_dict
        if path in file_dict:
            return file_dict[path]
        else:
            file_handle = zipfile.ZipFile(path, 'r')
            file_dict[path] = file_handle
            return file_dict[path]

    @staticmethod
    def imread(path, idx):
        zfile = TrainZipReader.build_file_dict(path)
        filelist = zfile.namelist()
        filelist.sort()
        data = zfile.read(filelist[idx])
        #
        im = Image.open(io.BytesIO(data))
        return im


class TestZipReader(object):
    file_dict = dict()

    def __init__(self):
        super(TestZipReader, self).__init__()

    @staticmethod
    def build_file_dict(path):
        file_dict = TestZipReader.file_dict
        if path in file_dict:
            return file_dict[path]
        else:
            file_handle = zipfile.ZipFile(path, 'r')
            file_dict[path] = file_handle
            return file_dict[path]

    @staticmethod
    def imread(path, idx):
        zfile = TestZipReader.build_file_dict(path)
        filelist = zfile.namelist()
        filelist.sort()
        data = zfile.read(filelist[idx])
        file_bytes = np.asarray(bytearray(data), dtype=np.uint8)
        im = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        # im = Image.open(io.BytesIO(data))
        return im


# ###########################################################################
# Data augmentation
# ###########################################################################


def to_tensors():
    return transforms.Compose([Stack(), ToTorchFormatTensor()])


class GroupRandomHorizontalFlowFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img_group, flowF_group, flowB_group):
        v = random.random()
        if v < 0.5:
            ret_img = [
                img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group
            ]
            ret_flowF = [ff[:, ::-1] * [-1.0, 1.0] for ff in flowF_group]
            ret_flowB = [fb[:, ::-1] * [-1.0, 1.0] for fb in flowB_group]
            return ret_img, ret_flowF, ret_flowB
        else:
            return img_group, flowF_group, flowB_group


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if is_flow:
                for i in range(0, len(ret), 2):
                    # invert flow pixel values when flipping
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group],
                                axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(
                pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


# ###########################################################################
# Create masks with random shape
# ###########################################################################


def create_random_shape_with_random_motion(video_length, imageHeight=240, imageWidth=432):
    # Get a random shape
    height = random.randint(imageHeight // 3, imageHeight - 1)
    width = random.randint(imageWidth // 3, imageWidth - 1)
    edge_num = random.randint(6, 8)
    ratio = random.randint(6, 8) / 10

    region = get_random_shape(edge_num=edge_num, ratio=ratio, height=height, width=width)
    region_width, region_height = region.size

    # Get random initial position
    x, y = random.randint(0, imageHeight - region_height), random.randint(0, imageWidth - region_width)

    # Increase the max speed for faster movement
    velocity = get_random_velocity(max_speed=5)  # Increased from 3 to 5

    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    m.paste(region, (y, x, y + region_width, x + region_height))
    masks = [m.convert('L')]

    # Return fixed masks set to 0
    # if random.uniform(0, 1) > 0.5:
    #     return masks * video_length

    # Return moving masks with faster movement and acceleration
    for _ in range(video_length - 1):
        x, y, velocity = random_move_control_points(x, y, imageHeight, imageWidth, velocity, 
                                                    region.size, 
                                                    maxLineAcceleration=(5, 1),  # Increased acceleration
                                                    maxInitSpeed=10)  # Increased from 3 to 10
        m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        m.paste(region, (y, x, y + region_width, x + region_height))
        masks.append(m.convert('L'))
        # detect it's edge
        # kernel_size = random.randint(1,8)
        # mask_edge = Image.fromarray(detect_edges(np.array(m), kernel_size=kernel_size, dilation_iteration=1))
        # masks.append(mask_edge.convert('L'))
    return masks


def create_random_shape_with_random_motion_zoom_rotation(video_length, zoomin=0.5, zoomout=2, rotmin=1, rotmax=10, imageHeight=240, imageWidth=432):
    # get a random shape
    assert zoomin < 1, "Zoom-in parameter must be smaller than 1"
    assert zoomout > 1, "Zoom-out parameter must be larger than 1"
    assert rotmin < rotmax, "Minimum value of rotation must be smaller than maximun value !"
    height = random.randint(imageHeight//3, imageHeight-1)
    width = random.randint(imageWidth//3, imageWidth-1)
    edge_num = random.randint(6, 8)
    ratio = random.randint(6, 8)/10
    region = get_random_shape(
        edge_num=edge_num, ratio=ratio, height=height, width=width)
    region_width, region_height = region.size
    # get random position
    x, y = random.randint(
        0, imageHeight-region_height), random.randint(0, imageWidth-region_width)
    velocity = get_random_velocity(max_speed=30)
    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    m.paste(region, (y, x, y+region.size[0], x+region.size[1]))
    masks = [m.convert('L')]
    # return fixed masks
    # if random.uniform(0, 1) > 0.5:
    #     return masks*video_length  # -> directly copy all the base masks
    # return moving masks
    for _ in range(video_length-1):
        x, y, velocity = random_move_control_points(
            x, y, imageHeight, imageWidth, velocity, region.size, maxLineAcceleration=(10, 2), maxInitSpeed=3)
        m = Image.fromarray(
            np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        ### add by kaidong, to simulate zoon-in, zoom-out and rotation
        extra_transform = random.uniform(0, 1)
        # zoom in and zoom out
        if extra_transform > 0.75:
            resize_coefficient = random.uniform(zoomin, zoomout)
            region = region.resize((math.ceil(region_width * resize_coefficient), math.ceil(region_height * resize_coefficient)), Image.NEAREST)
            m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
            region_width, region_height = region.size
        # rotation
        elif extra_transform > 0.5:
            m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
            m = m.rotate(random.randint(rotmin, rotmax))
            # region_width, region_height = region.size
        ### end
        else:
            m.paste(region, (y, x, y+region.size[0], x+region.size[1]))
        masks.append(m.convert('L'))
        # mask_edge = Image.fromarray(detect_edges(np.array(m), kernel_size=kernel_size, dilation_iteration=1))
        # masks.append(mask_edge.convert('L'))
    return masks


def get_random_shape(edge_num=9, ratio=0.7, width=432, height=240):
    '''
      There is the initial point and 3 points per cubic bezier curve.
      Thus, the curve will only pass though n points, which will be the sharp edges.
      The other 2 modify the shape of the bezier curve.
      edge_num, Number of possibly sharp edges
      points_num, number of points in the Path
      ratio, (0, 1) magnitude of the perturbation from the unit circle,
    '''
    points_num = edge_num*3 + 1
    angles = np.linspace(0, 2*np.pi, points_num)
    codes = np.full(points_num, Path.CURVE4)
    codes[0] = Path.MOVETO
    # Using this instead of Path.CLOSEPOLY avoids an innecessary straight line
    verts = np.stack((np.cos(angles), np.sin(angles))).T * \
        (2*ratio*np.random.random(points_num)+1-ratio)[:, None]
    verts[-1, :] = verts[0, :]
    path = Path(verts, codes)
    # draw paths into images
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='black', lw=2)
    ax.add_patch(patch)
    ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.axis('off')  # removes the axis to leave only the shape
    fig.canvas.draw()
    # convert plt images into numpy images
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape((fig.canvas.get_width_height()[::-1] + (3,)))
    plt.close(fig)
    # postprocess
    data = cv2.resize(data, (width, height))[:, :, 0]
    data = (1 - np.array(data > 0).astype(np.uint8))*255
    corrdinates = np.where(data > 0)
    xmin, xmax, ymin, ymax = np.min(corrdinates[0]), np.max(
        corrdinates[0]), np.min(corrdinates[1]), np.max(corrdinates[1])
    region = Image.fromarray(data).crop((ymin, xmin, ymax, xmax))
    return region


def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration
    if dist == 'uniform':
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    return (speed, angle)


def get_random_velocity(max_speed=3, dist='uniform'):
    if dist == 'uniform':
        speed = np.random.uniform(max_speed)
    elif dist == 'guassian':
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(
            f'Distribution type {dist} is not supported.')
    angle = np.random.uniform(0, 2 * np.pi)
    return (speed, angle)


def random_move_control_points(X,
                               Y,
                               imageHeight,
                               imageWidth,
                               lineVelocity,
                               region_size,
                               maxLineAcceleration=(3, 0.5),
                               maxInitSpeed=3):
    region_width, region_height = region_size
    speed, angle = lineVelocity
    X += int(speed * np.cos(angle))
    Y += int(speed * np.sin(angle))
    lineVelocity = random_accelerate(lineVelocity,
                                     maxLineAcceleration,
                                     dist='guassian')
    if ((X > imageHeight - region_height) or (X < 0)
            or (Y > imageWidth - region_width) or (Y < 0)):
        lineVelocity = get_random_velocity(maxInitSpeed, dist='guassian')
    new_X = np.clip(X, 0, imageHeight - region_height)
    new_Y = np.clip(Y, 0, imageWidth - region_width)
    return new_X, new_Y, lineVelocity


if __name__ == '__main__':

    trials = 10
    for _ in range(trials):
        video_length = 10
        # The returned masks are either stationary (50%) or moving (50%)
        masks = create_random_shape_with_random_motion(video_length,
                                                       imageHeight=240,
                                                       imageWidth=432)

        for m in masks:
            cv2.imshow('mask', np.array(m))
            cv2.waitKey(500)
