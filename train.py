import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.optim import lr_scheduler

import pytorch_ssim
import  time
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torch.nn.modules.loss import _Loss
from net.Ushape_Trans import *

#from dataset import prepare_data, Dataset
from net.utils import *
import cv2
import matplotlib.pyplot as plt
from utility import plots as plots, ptcolor as ptcolor, ptutils as ptutils, data as data
from loss.LAB import *
from loss.LCH import *


import torch.utils.data as dataf
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import time as time
import datetime
import sys
from torchvision.utils import save_image
import csv
import random


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dtype = 'float32'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.set_default_tensor_type(torch.FloatTensor)


generator = Generator().cuda()
discriminator = Discriminator().cuda()

def split(img):
    output=[]
    output.append(F.interpolate(img, scale_factor=0.125))
    output.append(F.interpolate(img, scale_factor=0.25))
    output.append(F.interpolate(img, scale_factor=0.5))
    output.append(img)
    return output


def training_x():

    training_x = []
    path = 'E:/DataSet-d/LSUI/Train-L/input/'  # 要改
    path_list = os.listdir(path)
    path_list.sort(key=lambda x: int(x.split('.')[0]))
    for item in path_list:
        impath = path + item

        imgx = cv2.imread(path + item)

        imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
        imgx = cv2.resize(imgx, (256, 256))
        training_x.append(imgx)
    X_train = []
    for features in training_x:
        X_train.append(features)

    X_train = np.array(X_train)


    X_train = X_train.astype(dtype)
    X_train = torch.from_numpy(X_train)

    X_train = X_train.permute(0, 3, 1, 2)


    X_train = X_train / 255.0
    X_train.shape

    return X_train



def training_x_d():

    training_x_d = []
    path = 'E:/DataSet-d/LSUI/Train-L/input-d/'  # 要改
    path_list = os.listdir(path)
    path_list.sort(key=lambda x: int(x.split('.')[0]))
    for item in path_list:
        impath = path + item

        imgx = cv2.imread(path + item)

        imgx = cv2.resize(imgx, (256, 256))
        training_x_d.append(imgx)

    X_train_d = []

    for features in training_x_d:
        X_train_d.append(features)

    X_train_d = np.array(X_train_d)


    X_train_d = X_train_d.astype(dtype)
    X_train_d = torch.from_numpy(X_train_d)

    X_train_d = X_train_d.permute(0, 3, 1, 2)

    X_train_d = X_train_d / 255.0

    return X_train_d

def training_y():

    training_y = []
    path = 'E:/DataSet-d/LSUI/Train-L/GT/'  # 要改
    path_list = os.listdir(path)

    path_list.sort(key=lambda x: int(x.split('.')[0]))
    for item in path_list:
        impath = path + item

        imgx = cv2.imread(path + item)
        imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
        imgx = cv2.resize(imgx, (256, 256))
        training_y.append(imgx)

    y_train = []

    for features in training_y:
        y_train.append(features)

    y_train = np.array(y_train)

    y_train = y_train.astype(dtype)
    y_train = torch.from_numpy(y_train)
    y_train = y_train.permute(0, 3, 1, 2)

    y_train = y_train / 255.0
    y_train.shape

    return y_train


def test_x():
    test_x = []
    path = 'E:/DataSet-d/LSUI/Test-L400/input/'  # 要改
    path_list = os.listdir(path)
    path_list.sort(key=lambda x: int(x.split('.')[0]))

    for item in path_list:
        impath = path + item

        imgx = cv2.imread(path + item)
        imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
        imgx = cv2.resize(imgx, (256, 256))
        test_x.append(imgx)

    x_test = []

    for features in test_x:
        x_test.append(features)

    x_test = np.array(x_test)

    x_test = x_test.astype(dtype)
    x_test = torch.from_numpy(x_test)
    x_test = x_test.permute(0, 3, 1, 2)

    x_test = x_test / 255.0

    return x_test

def test_x_d():
    test_x_d = []
    path ='E:/DataSet-d/LSUI/Test-L400/input-d/' # 要改
    path_list = os.listdir(path)
    path_list.sort(key=lambda x: int(x.split('.')[0]))

    for item in path_list:
        impath = path + item

        imgx = cv2.imread(path + item)

        imgx = cv2.resize(imgx, (256, 256))
        test_x_d.append(imgx)

    x_test_d = []

    for features in test_x_d:
        x_test_d.append(features)

    x_test_d = np.array(x_test_d)

    x_test_d = x_test_d.astype(dtype)
    x_test_d = torch.from_numpy(x_test_d)
    x_test_d = x_test_d.permute(0, 3, 1, 2)

    x_test_d = x_test_d / 255.0

    return x_test_d


def test_Y():
    test_Y = []
    path = 'E:/DataSet-d/LSUI/Test-L400/GT/'  # 要改
    path_list = os.listdir(path)
    path_list.sort(key=lambda x: int(x.split('.')[0]))
    for item in path_list:
        impath = path + item

        imgx = cv2.imread(path + item)
        imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
        imgx = cv2.resize(imgx, (256, 256))
        test_Y.append(imgx)

    Y_test = []

    for features in test_Y:
        Y_test.append(features)

    Y_test = np.array(Y_test)

    Y_test = Y_test.astype(dtype)
    Y_test = torch.from_numpy(Y_test)
    Y_test = Y_test.permute(0, 3, 1, 2)

    Y_test = Y_test / 255.0

    return Y_test


# sample_images该函数用于在神经网络训练期间保存验证集中生成的样本
def sample_images(batches_done,x_test,x_test_d,Y_test):
    """Saves a generated sample from the validation set"""

    generator.eval()

    i = random.randrange(1, 90)  # i=random.randrange(1,90)可能会选择 1 到 89 之间的随机整数

    real_A = Variable(x_test[i, :, :, :]).cuda()

    real_A_d = Variable(x_test_d[i, :, :, :]).cuda()
    real_B = Variable(Y_test[i, :, :, :]).cuda()
    real_A = real_A.unsqueeze(0)
    real_A_d = real_A_d.unsqueeze(0)
    real_B = real_B.unsqueeze(0)
    fake_B = generator(real_A, real_A_d)
    # print(fake_B.shape)
    imgx = fake_B[3].data
    imgy = real_B.data
    x = imgx[:, :, :, :]
    y = imgy[:, :, :, :]
    img_sample = torch.cat((x, y), -2)
    results_dir = 'images_d_r/results'
    os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist
    file_path = os.path.join(results_dir, f"{batches_done}.png")
    # Saving the image
    save_image(img_sample, file_path, nrow=5, normalize=True)

def train_g_d(start_epoch, X_train, y_train,X_train_d,x_test,x_test_d,Y_test):


    dataset = dataf.TensorDataset(X_train, X_train_d, y_train)
    loader = dataf.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    criterion_GAN = nn.MSELoss(size_average=False)
    criterion_pixelwise = nn.MSELoss(size_average=False)

    MSE = nn.MSELoss(size_average=False)
    SSIM = pytorch_ssim.SSIM()
    L_vgg = VGG19_PercepLoss()
    L_lab = lab_Loss()
    L_lch = lch_Loss()

    criterion_pixelwise.cuda()
    L_vgg.cuda()
    MSE.cuda()
    SSIM.cuda()
    criterion_GAN.cuda()
    L_lab.cuda()
    L_lch.cuda()

    lambda_pixel = 0.1
    lambda_lab = 0.001
    lambda_lch = 1
    lambda_con = 100
    lambda_ssim = 100

    patch = (1, 256 // 2 ** 5, 256 // 2 ** 5)

    LR = 0.0005
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
    scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=40, gamma=0.8)
    scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=40, gamma=0.8)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    f1 = open('psnr.csv', 'w', encoding='utf-8')  # 要改
    csv_writer1 = csv.writer(f1)
    f2 = open('SSIM.csv', 'w',
              encoding='utf-8')  # 要改
    csv_writer2 = csv.writer(f2)


    checkpoint_interval = 5
    epochs = start_epoch

    n_epochs = 500
    sample_interval = 1000

    use_pretrain = False

    if use_pretrain:
        # Load pretrained models
        start_epoch = epochs
        generator.load_state_dict(torch.load("saved_models/G/generator_%d.pth" % (start_epoch)))
        discriminator.load_state_dict(torch.load("saved_models/D/discriminator_%d.pth" % (start_epoch)))
        print('successfully loading epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('No pretrain model found, training will start from scratch！')

    psnr_list = []
    prev_time = time.time()


    for epoch in range(epochs, n_epochs):
        for i, batch in enumerate(loader):
            real_A = Variable(batch[0]).cuda()
            real_A_d = Variable(batch[1]).cuda()
            real_B = Variable(batch[2]).cuda()
            real_A1 = split(real_A)
            real_Ad = split(real_A_d)
            real_B1 = split(real_B)
            print(len(real_A1))
            print(len(real_Ad))
            print(len(real_B1))



            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)  # 全1
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)  # 全0

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()
            # GAN loss
            print(real_A.shape)
            print(real_A_d.shape)
            print("----------------")
            fake_B = generator(real_A, real_A_d)
            # fake_B1 = list(map(lambda x: x.detach(), fake_B))
            pred_fake = discriminator(fake_B, real_A1)

            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = (criterion_pixelwise(fake_B[0], real_B1[0]) + \
                          criterion_pixelwise(fake_B[1], real_B1[1]) + \
                          criterion_pixelwise(fake_B[2], real_B1[2]) + \
                          criterion_pixelwise(fake_B[3], real_B1[3])) / 4.0
            loss_ssim = -(SSIM(fake_B[0], real_B1[0]) + \
                          SSIM(fake_B[1], real_B1[1]) + \
                          SSIM(fake_B[2], real_B1[2]) + \
                          SSIM(fake_B[3], real_B1[3])) / 4.0
            ssim_value = - loss_ssim.item()
            loss_con = (L_vgg(fake_B[0], real_B1[0]) + \
                        L_vgg(fake_B[1], real_B1[1]) + \
                        L_vgg(fake_B[2], real_B1[2]) + \
                        L_vgg(fake_B[3], real_B1[3])) / 4.0
            loss_lab = (L_lab(fake_B[0], real_B1[0]) + \
                        L_lab(fake_B[1], real_B1[1]) + \
                        L_lab(fake_B[2], real_B1[2]) + \
                        L_lab(fake_B[3], real_B1[3])) / 4.0
            loss_lch = (L_lch(fake_B[0], real_B1[0]) + \
                        L_lch(fake_B[1], real_B1[1]) + \
                        L_lch(fake_B[2], real_B1[2]) + \
                        L_lch(fake_B[3], real_B1[3])) / 4.0

            # Total loss

            loss_G = (loss_GAN + lambda_pixel * loss_pixel + lambda_ssim * loss_ssim + \
                      lambda_con * loss_con + lambda_lab * loss_lab + lambda_lch * loss_lch)

            loss_G.backward(retain_graph=True)
            optimizer_G.step()


            # ---------------------
            #  Train Discriminator
            # ---------------------


            optimizer_D.zero_grad()

            # Real loss


            pred_real = discriminator(real_B1, real_A1)

            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            fake_B[0] = fake_B[0].detach()
            fake_B[1] = fake_B[1].detach()
            fake_B[2] = fake_B[2].detach()
            fake_B[3] = fake_B[3].detach()

            pred_fake1 = discriminator(fake_B, real_A1)
            loss_fake = criterion_GAN(pred_fake1, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)
            # loss_D=loss_real

            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(loader) + i
            batches_left = n_epochs * len(loader) - batches_done
            out_train = torch.clamp(fake_B[3], 0., 1.)
            psnr_train = batch_PSNR(out_train, real_B, 1.)
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d][PSNR: %f] [SSIM: %f] [D loss: %f] [G loss: %f],[lab: %f],[lch: %f], [pixel: %f],[VGG_loss: %f], [adv: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(loader),
                    psnr_train,
                    ssim_value,
                    loss_D.item(),
                    loss_G.item(),
                    0.001 * loss_lab.item(),
                    1 * loss_lch.item(),
                    0.1 * loss_pixel.item(),
                    100 * loss_con.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )



            # If at sample interval save image
            if batches_done % sample_interval == 0:
                sample_images(batches_done,x_test,x_test_d,Y_test)
                csv_writer1.writerow([str(psnr_train)])
                csv_writer2.writerow([str(ssim_value)])

        scheduler_G.step()
        scheduler_D.step()
        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/G/generator_%d.pth" % (epoch))
            torch.save(discriminator.state_dict(), "saved_models/D/discriminator_%d.pth" % (epoch))




def main():
    X_train=training_x()
    y_train=training_y()
    X_train_d=training_x_d()
    x_test=test_x()
    Y_test=test_Y()
    x_test_d=test_x_d()

    train_g_d(start_epoch=0,X_train=X_train,y_train=y_train,X_train_d=X_train_d,x_test=x_test,x_test_d=x_test_d,Y_test=Y_test)

if __name__ == '__main__':
    main()

































