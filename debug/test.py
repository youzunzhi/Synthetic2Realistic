from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import torch, os
import numpy as np
import imageio
import sys
sys.path.append('../')
from model.network import _UNetGenerator
use_cuda = False


def main():
    dataset = DataLoader(RGBDataset(), batch_size=16, shuffle=False, num_workers=0)
    dataset_size = len(dataset.dataset)
    print ('testing images = %d ' % dataset_size)
    model = TestModel()
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        model.save_results()


class TestModel(object):
    def __init__(self):
        self.model_names = ['img2task']
        self.net_img2task = _UNetGenerator(3, 1)
        self.save_dir = 'eigen_test_results/'
        if use_cuda:
            self.net_img2task.cuda()
        self.net_img2task = torch.nn.DataParallel(self.net_img2task)
        self.load_networks()
        self.pred_i = 0

    def load_networks(self):
        save_filename = '../model_on_github.pth'
        self.net_img2task.load_state_dict(torch.load(save_filename))

    def set_input(self, input):
        self.img = input[0]
        self.img_name = input[1]
        
        if use_cuda:
            self.img = self.img.cuda()

    def test(self):
        self.img = Variable(self.img)

        with torch.no_grad():
            self.depth_pred = self.net_img2task.forward(self.img)[-1]

    def save_results(self):
        for depth_pred in self.depth_pred:
            depth_pred_np = tensor2im(depth_pred)
            pred_name = '%s.png' % str(self.pred_i)
            self.pred_i += 1
            save_path = os.path.join(self.save_dir, pred_name)
            save_image(depth_pred_np, save_path)


def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])

    imageio.imwrite(image_path, image_numpy)


def get_img_paths():
    txt_file = '../datasplit/eigen_test_files.txt'
    image_paths = []
    with open(txt_file) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip()
        image_paths.append(path)

    return image_paths

def get_transform():
    transforms_list = []

    transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
    transforms_list += [
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    return transforms.Compose(transforms_list)

# convert a tensor into a numpy array
def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes

    return image_numpy.astype(imtype)


class RGBDataset(Dataset):
    def __init__(self):
        self.img_paths = get_img_paths()

        self.transform_augment = get_transform()

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        img_name = img_path.split('/')[-1]

        img = Image.open(img_path).convert('RGB')
        img = img.resize([640, 192], Image.BICUBIC)
        img = self.transform_augment(img)
        return img, img_name

    def __len__(self):
        return self.img_paths

if __name__ == '__main__':
    main()