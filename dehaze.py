import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob


def dehaze_image(image_path):

	data_hazy = Image.open(image_path)
	data_hazy = (np.asarray(data_hazy)/255.0)

	data_hazy = torch.from_numpy(data_hazy).float()
	data_hazy = data_hazy.permute(2,0,1)
	data_hazy = data_hazy.cpu().unsqueeze(0) #.cpu()#.cuda()

	dehaze_net = net.dehaze_net().cpu()#.cuda()
	dehaze_net.load_state_dict(torch.load('snapshots/origin.pth',map_location=torch.device('cpu')))

	clean_image = dehaze_net(data_hazy)
	torchvision.utils.save_image(clean_image,"results/" + image_path.split("/")[-1])#torch.cat((data_hazy, clean_image),0)


#torchvision.utils.save_image(clean_image,"results/" + image_path.split("/")[-1])#torch.cat((data_hazy, clean_image),0),
	

if __name__ == '__main__':

	test_list = glob.glob("test_images/*")

	for image in test_list:

		dehaze_image(image)
		print(image, "done!")
