import pytorch_ssim
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import os
import sys
import argparse
import time
import dataloader
import net
import pytorch_ssim as pytorch_ssim
import numpy as np
from torchvision import transforms
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from utils import TVLoss, print_network
#10:42开始训练
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):

    dehaze_net = net.dehaze_net().cpu()#.cuda()
    dehaze_net.apply(weights_init)

    print('---------- Networks architecture -------------')
    print_network(dehaze_net)

    train_dataset = dataloader.dehazing_loader(config.orig_images_path,
                                             config.hazy_images_path)
    val_dataset = dataloader.dehazing_loader(config.orig_images_path,
                                             config.hazy_images_path, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    TV_loss = TVLoss().cpu()
    ssim = pytorch_ssim.SSIM().cpu()#gpus_list[0]


    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    dehaze_net.train()

    #trainloss记录loss值
    train_loss = []
    for epoch in range(config.num_epochs):
        for iteration, (img_orig, img_haze) in enumerate(train_loader):

            img_orig = img_orig.cpu()#.cuda()
            img_haze = img_haze.cpu()#.cuda()

            clean_image = dehaze_net(img_haze)


            #loss1 = criterion1(clean_image, img_orig)#MSE损失
            #loss2 = criterion2(clean_image, img_orig)#SSIM损失
           # loss = 0.6*loss1+0.4*loss2 #总损失函数L=0.6*MSE+0.4*L1
            ssim_loss = 1 - ssim(clean_image,img_orig)
            tv_loss = TV_loss(clean_image)
            loss = ssim_loss + 0.001 * tv_loss
            train_loss.append(loss.item())#loss值加入列表中

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(),config.grad_clip_norm)#torch.nn.utils.clip_grad_norm 已弃用
            optimizer.step()

            if ((iteration+1) % config.display_iter) == 0:
                print("Loss at iteration", iteration+1, ":", loss.item())
            if ((iteration+1) % config.snapshot_iter) == 0:

                torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

    with open("./train_loss.txt", 'w') as train_los:#loss写入文件
        train_los.write(str(train_loss))

        # Validation Stage
        for iter_val, (img_orig, img_haze) in enumerate(val_loader):

            img_orig = img_orig.cpu()#.cuda()
            img_haze = img_haze.cpu()#.cuda()

            clean_image = dehaze_net(img_haze)

            torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig),0), config.sample_output_folder+str(iter_val+1)+".jpg")

        torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth")










if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--orig_images_path', type=str, default="data/images/")
    parser.add_argument('--hazy_images_path', type=str, default="data/data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train(config)






'''


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):

    dehaze_net = net.dehaze_net().cpu()#.cuda()
    dehaze_net.apply(weights_init)

    train_dataset = dataloader.dehazing_loader(config.orig_images_path,
                                             config.hazy_images_path)
    val_dataset = dataloader.dehazing_loader(config.orig_images_path,
                                             config.hazy_images_path, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    criterion = nn.MSELoss().cpu()#.cuda()
    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    dehaze_net.train()

    for epoch in range(config.num_epochs):
        for iteration, (img_orig, img_haze) in enumerate(train_loader):

            img_orig = img_orig.cpu()#.cuda()
            img_haze = img_haze.cpu()#.cuda()

            clean_image = dehaze_net(img_haze)

            loss = criterion(clean_image, img_orig)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(),config.grad_clip_norm)#torch.nn.utils.clip_grad_norm 已弃用
            optimizer.step()

            if ((iteration+1) % config.display_iter) == 0:
                print("Loss at iteration", iteration+1, ":", loss.item())
            if ((iteration+1) % config.snapshot_iter) == 0:

                torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

        # Validation Stage
        for iter_val, (img_orig, img_haze) in enumerate(val_loader):

            img_orig = img_orig.cpu()#.cuda()
            img_haze = img_haze.cpu()#.cuda()

            clean_image = dehaze_net(img_haze)

            torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig),0), config.sample_output_folder+str(iter_val+1)+".jpg")

        torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth")










if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--orig_images_path', type=str, default="data/images/")
    parser.add_argument('--hazy_images_path', type=str, default="data/data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train(config)





'''


