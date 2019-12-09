import pandas as pd
import torch, os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model.network import define_G


def main():
    baseline_dataset = RGBDepthDataset()
    train_dataloader = DataLoader(baseline_dataset, batch_size=32, shuffle=True, num_workers=0)
    epoch_count = 1
    niter = 6
    niter_decay = 4
    model = TNetModel()
    for epoch in range(epoch_count, niter + niter_decay + 1):
        for batch_i, batch in enumerate(train_dataloader):
            print('batch:', batch_i, end=' ')
            model.set_input(batch)
            model.optimize_parameters()
            model.save_weights(epoch)



class RGBDepthDataset(Dataset):
    def __init__(self):
        csv_path = 'data/vkitti_train.csv'
        img_size = [640, 192]
        max_depth = 8000.
        rgb_normalize = True
        self.frame = pd.read_csv(csv_path, header=None)
        rgb_transforms_list = [transforms.Resize(img_size), transforms.ToTensor()]
        if rgb_normalize:
            rgb_transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.rgb_transform = transforms.Compose(rgb_transforms_list)
        self.depth_transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),
                                                   ClampNormalizeTransform(max_depth, 0.5, 0.5)])

    def __getitem__(self, idx):
        rgb_name = self.frame.iloc[idx, 0]
        depth_name = self.frame.iloc[idx, 1]
        rgb = Image.open(rgb_name).convert('RGB')
        depth = Image.open(depth_name)

        rgb = self.rgb_transform(rgb)
        depth = self.depth_transform(depth)

        return rgb, depth, rgb_name, depth_name

    def __len__(self):
        return len(self.frame)


class ClampNormalizeTransform(object):
    def __init__(self, max_depth, mean, std):
        self.max_depth = max_depth
        self.mean = mean
        self.std = std

    def __call__(self, depth):
        depth = depth.type(torch.float)
        clamped = torch.clamp(depth, 0., self.max_depth)
        normalized = (clamped / self.max_depth - self.mean) / self.std
        return normalized


class TNetModel(object):
    def __init__(self):
        gpu_ids = [0] if USE_CUDA else []
        self.net_img2task = define_G(input_nc=3, output_nc=1, gpu_ids=gpu_ids)

        self.l1loss = torch.nn.L1Loss()
        self.l2loss = torch.nn.MSELoss()

        self.optimizer_img2task = torch.optim.Adam(self.net_img2task.parameters(), lr=0.0001, betas=(0.9, 0.999))

        self.schedulers = get_scheduler(self.optimizer_img2task)

    def set_input(self, input):
        self.input = input
        self.img_source = input[0]
        self.lab_source = input[1]

        if USE_CUDA:
            self.img_source = self.img_source.cuda()
            self.lab_source = self.lab_source.cuda()

    def forward(self):
        from torch.autograd import Variable
        self.img_s = Variable(self.img_source)
        self.lab_s = Variable(self.lab_source)

    def foreward_G_basic(self, net_G, img_s):

        fake = net_G(img_s)

        size = len(fake)

        f_s = fake[0]
        img_fake = fake[1:]

        img_s_fake = []

        for img_fake_i in img_fake:
            img_s_fake.append(img_fake_i)

        return img_s_fake, f_s, size

    def backward_task(self):

        self.lab_s_g, self.lab_f_s, size = \
            self.foreward_G_basic(self.net_img2task, self.img_s)

        lab_real = scale_pyramid(self.lab_s, size - 1)
        task_loss = 0
        for (lab_fake_i, lab_real_i) in zip(self.lab_s_g, lab_real):
            task_loss += self.l1loss(lab_fake_i, lab_real_i)

        lambda_rec_lab = 100.
        self.loss_lab_s = task_loss * lambda_rec_lab

        total_loss = self.loss_lab_s

        total_loss.backward()
        print('loss = ', total_loss.item())

    def optimize_parameters(self):

        self.forward()
        # task network
        self.optimizer_img2task.zero_grad()
        self.backward_task()
        self.optimizer_img2task.step()

    def save_weights(self, epoch):
        save_fname = os.path.join(OUTPUT_DIR, 'TNet_' + str(epoch) + '.pth')
        torch.save(self.net_img2task.state_dict(), save_fname)
        print('saved model to', save_fname)


def get_scheduler(optimizer):
    from torch.optim import lr_scheduler
    lr_policy = 'lambda'
    epoch_count = 1
    niter = 6
    niter_decay = 4

    if lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + 1 + epoch_count - niter) / float(niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


def scale_pyramid(img, num_scales):
    scaled_imgs = [img]

    s = img.size()

    h = s[2]
    w = s[3]

    for i in range(1, num_scales):
        ratio = 2 ** i
        nh = h // ratio
        nw = w // ratio
        import torch.nn.functional as F
        scaled_img = F.upsample(img, size=(nh, nw), mode='bilinear', align_corners=True)
        scaled_imgs.append(scaled_img)

    scaled_imgs.reverse()
    return scaled_imgs


if __name__ == '__main__':
    CUDA_ID = '0'
    USE_CUDA = torch.cuda.is_available()
    import time
    OUTPUT_DIR = 'runs/' + 'TNet-[{}]'.format(time.strftime('%Y-%m-%d-%X', time.localtime(time.time())))
    if USE_CUDA:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_ID
    main()
