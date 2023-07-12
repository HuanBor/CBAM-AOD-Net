import torch
from net import dehaze_net as AOD
import torchvision.transforms as transforms
from PIL import Image
import torchvision

def test_on_img_(state_dict_path, img_cv2):
    state_dict = torch.load(state_dict_path,map_location=torch.device('cpu'))
    model = AOD()
    model.load_state_dict(state_dict)

    img = Image.fromarray(img_cv2)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    result_img = model(img)

    return result_img

def test_on_img(state_dict_path, img_path):
    state_dict = torch.load(state_dict_path,map_location=torch.device('cpu'))
    model = AOD()
    model.load_state_dict(state_dict)

    img = Image.open(img_path)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    result_img = model(img)

    return result_img

if __name__=='__main__':
    img_name = 'river'
    suffix = '.png'
    result = test_on_img('snapshots/Epoch9.pth', 'testbench/'+img_name+suffix)
    torchvision.utils.save_image(result, 'testbench/'+img_name+'_result'+suffix)
    
