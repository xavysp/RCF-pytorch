from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
from PIL import Image
import numpy as np
import cv2
from utils import image_normalization



def prepare_image_PIL(im):
    im = im[:,:,::-1] - np.zeros_like(im) # rgb to bgr
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im

def prepare_image_cv2(im):
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im


class BSDS_RCFLoader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, split='train', transform=False,args=None):
        self.root = args.dataset
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.curr_dataset = args.train_dataset
            self.dataset_dir = join(self.root,args.train_dataset)
            if args.train_dataset.lower()=='ssmihd':
                self.ssmihd_dir = self.list_path=join(self.dataset_dir,'edges')
                self.list_path=join(self.ssmihd_dir,args.train_list)
            else:
                self.list_path = join(self.dataset_dir, args.train_list)
        elif self.split == 'test':
            self.curr_dataset = args.test_dataset
            self.dataset_dir = join(self.root, args.test_dataset)
            if args.test_dataset.lower()=='ssmihd':
                self.ssmihd_dir = self.list_path=join(self.dataset_dir,'edges')
                self.list_path=join(self.ssmihd_dir,args.test_list)
            else:
                self.list_path = join(self.dataset_dir, args.test_list)
        else:
            raise ValueError("Invalid split type!")
        with open(self.list_path, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]  # Remove the newline at last.

        if self.split == 'train':
            if args.train_dataset.lower() == 'ssmihd':
                pairs = [line.split() for line in lines]
                self.images_path = [pair[0] for pair in pairs]
                self.edges_path = [pair[1] for pair in pairs]
            else:
                pairs = [line.split() for line in lines]
                self.images_path = [pair[0] for pair in pairs]
                self.edges_path = [pair[1] for pair in pairs]
        else: # test
            self.images_path = lines
            self.images_name = []  # Used to save temporary edges.
            pa = [line.split() for line in self.images_path]
            if args.test_dataset.lower() == 'ssmihd':
                self.images_path = [pair[0] for pair in pa]
                for i in range(len(self.images_path)):
                    folder, filename = os.path.split(pa[i][0])
                    name, ext = splitext(filename)
                    self.images_name.append(name)
            else:
                # data which is not ssmihd
                self.images_path = [pair[0] for pair in pa]
                for path in pa:
                    folder, filename = os.path.split(path[0])
                    name, ext = splitext(filename)
                    self.images_name.append(name)
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        edge = None
        if self.split == "train":
            # Get edge.
            if self.curr_dataset.lower() == 'ssmihd':
                gt_dir = join(self.dataset_dir, 'edges', 'edge_maps', 'train')
                edge_path = join(gt_dir, self.edges_path[index])
                edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
                edge = cv2.resize(edge, dsize=(400, 400))
                edge = image_normalization(edge)
                edge = edge[np.newaxis, :, :]  # Add one channel at first (CHW).
                edge[edge == 0] = 0
                edge[np.logical_and(edge>0, edge<127.5)] = 2
                edge[edge >= 127.5] = 1
                edge = edge.astype(np.float32)
            else:
                edge_path = join(self.dataset_dir, self.edges_path[index])
                edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
                edge = image_normalization(edge)
                edge = edge[np.newaxis, :, :]  # Add one channel at first (CHW).
                edge[edge == 0] = 0
                edge[np.logical_and(edge > 0, edge < 127.5)] = 2
                edge[edge >= 127.5] = 1
                edge = edge.astype(np.float32)

        # Get image.
        if self.curr_dataset.lower() == 'ssmihd' and self.split == "train":
            x_dir = join(self.dataset_dir, 'edges', 'imgs', 'train')
            image_path = join(x_dir, self.images_path[index])
            image = cv2.imread(image_path).astype(np.float32)
            image = cv2.resize(image, dsize=(400, 400))
        elif self.curr_dataset.lower() == 'ssmihd' and self.split != "train":
            x_dir = join(self.dataset_dir, 'edges', 'imgs', 'test')
            image_path = join(x_dir, self.images_path[index])
            image = cv2.imread(image_path).astype(np.float32)
            # image = cv2.resize(image, dsize=(400, 400))
        else:
            image_path = join(self.dataset_dir, self.images_path[index])
            image = cv2.imread(image_path).astype(np.float32)
        # Note: Image arrays read by OpenCV and Matplotlib are slightly different.
        # Matplotlib reading code:
        #   image = plt.imread(image_path).astype(np.float32)
        #   image = image[:, :, ::-1]            # RGB to BGR.
        # Reference:
        #   https://oldpan.me/archives/python-opencv-pil-dif
        image = image - np.array((104.00698793,  # Minus statistics.
                                  116.66876762,
                                  122.67891434))
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW.
        image = image.astype(np.float32)  # To float32.

        # Return image and (possible) edge.
        if self.split == 'train':
            return image, edge
        else:
            return image
