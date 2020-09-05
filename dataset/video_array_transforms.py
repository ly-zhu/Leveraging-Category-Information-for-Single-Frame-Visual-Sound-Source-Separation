import random
import numbers
import torchvision.transforms.functional as F
from PIL import Image
import torch
import scipy.ndimage as ndimage
import numpy as np
#import cv2
from skimage.transform import rescale

class Resize(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size, order=2):
        self.size = size
        self.order = order

    def __call__(self, frames, flows):
        h, w, _ = frames[0].shape
        #print('in: ',frames[0].shape)
        out_frames = []
        for frame in frames:
            #h, w, _ = frame.shape
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                out_frames.append(frame)
                continue
                #return inputs,target
            if w < h:
                ratio = self.size/w
            else:
                ratio = self.size/h

            #inputs[0] = ndimage.interpolation.zoom(inputs[0], ratio, order=self.order)
            #inputs[1] = ndimage.interpolation.zoom(inputs[1], ratio, order=self.order)
            frame = rescale(frame, (ratio, ratio, 1), anti_aliasing=False)
            #frame = cv2.resize(frame, ratio, cv2.INTER_LINEAR)
            #frame = ndimage.interpolation.zoom(frame, (ratio, ratio, 1), order=self.order)
            frame *= ratio
            out_frames.append(frame)

        out_flows = []
        for flo in flows:
            #h, w, _ = flo.shape
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                out_flows.append(flo)
                continue
                #return inputs,target
            if w < h:
                ratio = self.size/w
            else:
                ratio = self.size/h

            #inputs[0] = ndimage.interpolation.zoom(inputs[0], ratio, order=self.order)
            #inputs[1] = ndimage.interpolation.zoom(inputs[1], ratio, order=self.order)
            flo = rescale(flo, (ratio, ratio, 1), anti_aliasing=False)
            #flo = cv2.resize(flo, ratio, cv2.INTER_LINEAR)
            #flo = ndimage.interpolation.zoom(flo, (ratio, ratio, 1), order=self.order)
            flo *= ratio
            out_flows.append(flo)
        #print('out: ', out_frames[0].shape)
        return out_frames, out_flows#inputs, target



class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, frames, flows):
        h, w, _ = frames[0].shape
        th, tw = self.size      
        x = random.randint(0, w - tw)
        y = random.randint(0, h - th)

        out_frames = []
        for frame in frames:
            #h, w, _ = frame.shape
            #th, tw = self.size
            if w == tw and h == th:
                out_frames.append(frame)
                continue
                #return inputs,target

            #x = random.randint(0, w - tw)
            #y = random.randint(0, h - th)
            frame = frame[y: y + th,x: x + tw]
            out_frames.append(frame)
        out_flows = []
        for flo in flows:
            #h, w, _ = flo.shape
            #th, tw = self.size
            if w == tw and h == th:
                out_flows.append(flo)
                continue
                #return inputs,target

            #x = random.randint(0, w - tw)
            #y = random.randint(0, h - th)
            flo = flo[y: y + th,x: x + tw]
            out_flows.append(flo)

        return out_frames, out_flows#inputs, target[y1: y1 + th,x1: x1 + tw]



class CenterCrop(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, frames, flows):
        h, w, _ = frames[0].shape
        th, tw = self.size
        x = int(round((w - tw) / 2.))
        y = int(round((h - th) / 2.))
        out_frames = []
        for frame in frames:
            #h, w, _ = frame.shape
            #th, tw = self.size
            #x = int(round((w - tw) / 2.))
            #y = int(round((h - th) / 2.))
            frame = frame[y: y + th, x: x + tw]
            out_frames.append(frame)

        out_flows = []
        for flo in flows:
            #h, w, _ = frame.shape
            #th, tw = self.size
            #x = int(round((w - tw) / 2.))
            #y = int(round((h - th) / 2.))
            flo = flo[y: y + th, x: x + tw]
            out_flows.append(flo)

        return out_frames, out_flows


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, frames, flows):
        out_frames = []
        for frame in frames:
            if random.random() < 0.5:
                frame = np.copy(np.fliplr(frame))
                frame[:,:,0] *= -1
            out_frames.append(frame)

        out_flows = []
        for flo in flows:
            if random.random() < 0.5:
                flo = np.copy(np.fliplr(flo))
                flo[:,:,0] *= -1
            out_flows.append(flo)

        return out_frames, out_flows

class ToTensor(object):
    """Convert a list of ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a list of PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x L xH x W) in the range
    [0.0, 1.0].
    """

    def __call__(self, frames, flows):
        """
        Args:
            frames: a list of (PIL Image or numpy.ndarray).
        Returns:
            a list of Tensor: Converted images.
        """
        #print(frames[0].shape)
        out_frames = []
        for frame in frames:
            out_frames.append(F.to_tensor(frame))
        out_flows = []
        for flo in flows:
            out_flows.append(F.to_tensor(flo))
        return out_frames, out_flows


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, frames, flows):
        out_frames = []
        for frame in frames:
            assert(isinstance(frame, np.ndarray))
            frame = np.transpose(frame, (2, 0, 1))
            tensor_frame = torch.from_numpy(frame)
            out_frames.append(tensor_frame.float())
        out_flows = []
        for flo in flows:
            assert(isinstance(flo, np.ndarray))
            flo = np.transpose(flo, (2, 0, 1))
            tensor_flo = torch.from_numpy(flo)
            out_flows.append(tensor_flo.float())
        # put it from HWC to CHW format
        #return tensor_frame.float(), tensor_flo.float()
        return out_frames, out_flows

class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, frames, flows):
        for t in self.co_transforms:
            #print('t', t)
            frames,flows = t(frames,flows)
        return frames,flows


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames, flows):
        """
        Args:
            frames: a list of Tensor image of size (C, H, W) to be normalized.
        Returns:
            a list of Tensor: a list of normalized Tensor images.
        """
        out_frames = []
        for frame in frames:
            out_frames.append(F.normalize(frame, self.mean, self.std))
        out_flows = []
        for flo in flows:
            out_flows.append(F.normalize(flo, [0,0], [1,1]))
        return out_frames, out_flows


class Stack(object):
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, frames, flows):
        """
        Args:
            frames: a list of (L) Tensor image of size (C, H, W).
        Returns:
            Tensor: a video Tensor of size (C, L, H, W).
        """
        return torch.stack(frames, dim=self.dim), torch.stack(flows, dim=self.dim)
