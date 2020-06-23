import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.autograd.variable as Variable
import numpy as np
import scipy.io as sio
from os.path import join as pjoin
import skimage.io as io
import time
import skimage
import warnings
import cv2 as cv

def cv_imshow(title='default',img=None):
    print(img.shape)
    cv.imshow(title,img)
    cv.waitKey(0)
    cv.destroyAllWindows()


class Logger(object):
  def __init__(self, fpath=None):
    self.console = sys.stdout
    self.file = None
    if fpath is not None:
      self.file = open(fpath, 'w')

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
        self.file.write(msg)

  def flush(self):
    self.console.flush()
    if self.file is not None:
        self.file.flush()
        os.fsync(self.file.fileno())

  def close(self):
    self.console.close()
    if self.file is not None:
        self.file.close()

class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def load_pretrained(model, fname, optimizer=None):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch
    """
    if os.path.isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer, checkpoint['epoch']
        else:
            return model, checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(fname))

def load_vgg16pretrain(model, vggmodel='data/vgg16convs.mat'):
    vgg16 = sio.loadmat(vggmodel)
    torch_params =  model.state_dict()
    for k in vgg16.keys():
        name_par = k.split('-')
        size = len(name_par)
        if size  == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(vgg16[k])
            torch_params[name_space] = torch.from_numpy(data)
    model.load_state_dict(torch_params)

def load_vgg16pretrain_half(model, vggmodel='data/vgg16convs.mat'):
    vgg16 = sio.loadmat(vggmodel)
    torch_params =  model.state_dict()
    for k in vgg16.keys():
        name_par = k.split('-')
        size = len(name_par)
        if size  == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(vgg16[k])
            shape = data.shape
            index = int(shape[0]/2)
            if len(shape) == 1:
                data = data[:index]
            else:
                data = data[:index,:,:,:]
            torch_params[name_space] = torch.from_numpy(data)
    model.load_state_dict(torch_params)

def load_fsds_caffe(model, fsdsmodel='caffe-fsds.mat'):
    fsds = sio.loadmat(fsdsmodel)
    torch_params =  model.state_dict()
    for k in fsds.keys():
        name_par = k.split('-')
        #print (name_par)
        size = len(name_par)

        data = np.squeeze(fsds[k])


        if 'upsample' in name_par:
           # print('skip upsample')
            continue 


        if size  == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(fsds[k])
            if data.ndim==2:
                data = np.reshape(data, (data.shape[0], data.shape[1]))

            torch_params[name_space] = torch.from_numpy(data)

        if size  == 3:
           # if 'bias' in name_par:
            #    continue

            name_space = name_par[0] + '_' + name_par[1]+ '.' + name_par[2]
            data = np.squeeze(fsds[k])
           # print(data.shape)
            if data.ndim==2:
               # print (data.shape[0])
                data = np.reshape(data,(data.shape[0], data.shape[1]))
            if data.ndim==1 :                
                data = np.reshape(data, (1, len(data), 1, 1))
            if data.ndim==0:
                data = np.reshape(data, (1))

            torch_params[name_space] = torch.from_numpy(data)

        if size == 4:
           # if 'bias' in name_par:
            #    continue
            data = np.squeeze(fsds[k])
            name_space = name_par[0] + '_' + name_par[1] + name_par[2] + '.' + name_par[3]
            if data.ndim==2:
                data = np.reshape(data,(data.shape[0], data.shape[1], 1, 1))

            torch_params[name_space] = torch.from_numpy(data)

    model.load_state_dict(torch_params)
    print('loaded')


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1,4,1,1]):
            torch.nn.init.constant_(m.weight, 0.25)
        if m.bias is not None:
            m.bias.data.zero_()

def image_normalization(img, img_min=0, img_max=255):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)
    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255
    :return: a normalized image, if max is 255 the dtype is uint8
    """
    img = np.float32(img)
    epsilon=1e-12 # whenever an inconsistent image
    img = (img-np.min(img))*(img_max-img_min)/((np.max(img)-np.min(img))+epsilon)+img_min
    return img

def restore_rgb(config,I):
    """
    :param config: [args.channel_swap, args.mean_pixel_value]
    :param I: and image or a set of images
    :return: an image or a set of images restored
    """

    if  len(I)>3 and not type(I)==np.ndarray:
        I =np.array(I)
        I = I[:,:,:,0:3]
        n = I.shape[0]
        for i in range(n):
            x = I[i,...]
            x = np.array(x, dtype=np.float32)
            x += config[1]
            x = x[:, :, config[0]]
            x = image_normalization(x)
            I[i,:,:,:]=x
    elif len(I.shape)==3 and I.shape[-1]==3:
        I = np.array(I, dtype=np.float32)
        I += config[1]
        I = I[:, :, config[0]]
        I = image_normalization(I)
    else:
        print("Sorry the input data size is out of our configuration")
    # print("The enterely I data {} restored".format(I.shape))
    return I

def visualize_result(imgs_list, arg):
    """
    function for Pytorch
    :param imgs_list: a list of 8 tensors
    :param arg:
    :return:
    """
    n_imgs = len(imgs_list)
    if n_imgs==8:
        img,gt,ed1,ed2,ed3,ed4,ed5,edf=imgs_list
        img = np.transpose(np.squeeze(img),[1,2,0])
        img = restore_rgb([arg.channels_swap,arg.mean_pixel_values[:3]],img) if arg.train_dataset.lower()=='ssmihd'\
            else img
        img = np.uint8(image_normalization(img))
        h,w,c = img.shape
        gt = np.squeeze(gt)
        gt = np.uint8(image_normalization(gt))
        gt = cv.bitwise_not(cv.cvtColor(gt,cv.COLOR_GRAY2BGR))
        ed1 = np.squeeze(ed1)
        ed1 = np.uint8(image_normalization(ed1))
        ed1 = cv.bitwise_not(cv.resize(cv.cvtColor(ed1, cv.COLOR_GRAY2BGR),dsize=(w,h)))
        ed2 = np.squeeze(ed2)
        ed2 = np.uint8(image_normalization(ed2))
        ed2 = cv.bitwise_not(cv.resize(cv.cvtColor(ed2, cv.COLOR_GRAY2BGR),dsize=(w,h)))
        ed3= np.squeeze(ed3)
        ed3 = np.uint8(image_normalization(ed3))
        ed3 = cv.bitwise_not(cv.resize(cv.cvtColor(ed3, cv.COLOR_GRAY2BGR),dsize=(w,h)))
        ed4 = np.squeeze(ed4)
        ed4 = np.uint8(image_normalization(ed4))
        ed4 = cv.bitwise_not(cv.resize(cv.cvtColor(ed4, cv.COLOR_GRAY2BGR),dsize=(w,h)))
        ed5 = np.squeeze(ed5)
        ed5 = np.uint8(image_normalization(ed5))
        ed5 = cv.bitwise_not(cv.resize(cv.cvtColor(ed5, cv.COLOR_GRAY2BGR),dsize=(w,h)))
        edf = np.squeeze(edf)
        edf = np.uint8(image_normalization(edf))
        edf = cv.bitwise_not(cv.resize(cv.cvtColor(edf, cv.COLOR_GRAY2BGR),dsize=(w,h)))
        res = [img,ed1,ed2,ed3,ed4,ed5,edf,gt]
        if n_imgs%2==0:
            imgs = np.zeros((img.shape[0]*2+10,img.shape[1]*(n_imgs//2)+((n_imgs//2-1)*5),3))
        else:
            imgs = np.zeros((img.shape[0] * 2 + 10, img.shape[1] * ((1+n_imgs) // 2) + ((n_imgs // 2 ) * 5), 3))
            n_imgs +=1
        k=0
        imgs = np.uint8(imgs)
        i_step = img.shape[0]+10
        j_step = img.shape[1]+5
        for i in range(2):
            for j in range(n_imgs//2):
                if k<len(res):
                    imgs[i*i_step:i*i_step+img.shape[0],j*j_step:j*j_step+img.shape[1],:]=res[k]
                    k+=1
                else:
                    pass
        return imgs
    else:
        raise NotImplementedError