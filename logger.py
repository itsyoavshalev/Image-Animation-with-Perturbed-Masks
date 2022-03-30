import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections

class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['driving2'], inp['source'], inp['target'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, mask_generator=None, optimizer_generator=None, optimizer_mask_generator=None):
        checkpoint = torch.load(checkpoint_path)

        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if mask_generator is not None:
            mask_generator.load_state_dict(checkpoint['mask_generator'])
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_mask_generator is not None:
            optimizer_mask_generator.load_state_dict(checkpoint['optimizer_mask_generator'])

        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out, save_w=True):
        self.epoch = epoch
        self.models = models
        if save_w:
            if (self.epoch + 1) % self.checkpoint_freq == 0:
                self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)


class Visualizer:
    def __init__(self, draw_border=False, colormap='gist_rainbow'):
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, driving2, source, target, out):
        images = []

        source = source.data.cpu().numpy()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append(source)

        # source_mask_int = out['source_mask']
        # source_mask_int = source_mask_int.data.cpu()
        # source_mask_int = np.transpose(source_mask_int, [0, 2, 3, 1])
        # images.append(source_mask_int)

        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append(driving)

        # driving_mask_int = out['driving_mask']
        # driving_mask_int = driving_mask_int.data.cpu()
        # driving_mask_int = np.transpose(driving_mask_int, [0, 2, 3, 1])
        # images.append(driving_mask_int)

        # if 'driving_mask_dist' in out:
        #     driving_mask_dist = out['driving_mask_dist']
        #     driving_mask_dist = driving_mask_dist.data.cpu()
        #     driving_mask_dist = np.transpose(driving_mask_dist, [0, 2, 3, 1])
        #     images.append(driving_mask_dist)

        # if 'driving2_mask' in out:
        #     driving2 = driving2.data.cpu().numpy()
        #     driving2 = np.transpose(driving2, [0, 2, 3, 1])
        #     images.append(driving2)

        #     driving2_mask_int = out['driving2_mask']
        #     driving2_mask_int = driving2_mask_int.data.cpu()
        #     driving2_mask_int = np.transpose(driving2_mask_int, [0, 2, 3, 1])
        #     images.append(driving2_mask_int)

        #     fixed_d2_mask = out['fixed_d2_mask']
        #     fixed_d2_mask = fixed_d2_mask.data.cpu()
        #     fixed_d2_mask = np.transpose(fixed_d2_mask, [0, 2, 3, 1])
        #     images.append(fixed_d2_mask)

        #     final_d2 = out['final_d2'].data.cpu().numpy()
        #     final_d2 = np.transpose(final_d2, [0, 2, 3, 1])
        #     images.append(final_d2)

        prediction1 = out['first_phase_prediction'].data.cpu().numpy()
        prediction2 = out['second_phase_prediction'].data.cpu().numpy()
        prediction1 = np.transpose(prediction1, [0, 2, 3, 1])
        prediction2 = np.transpose(prediction2, [0, 2, 3, 1])
        images.append(prediction1)
        images.append(prediction2)

        # if 'driving_mask_int_disturbed' in out:
        #     driving_mask_int_disturbed = out['driving_mask_int_disturbed']
        #     driving_mask_int_disturbed = driving_mask_int_disturbed.repeat(1,3,1,1)
        #     driving_mask_int_disturbed = F.interpolate(driving_mask_int_disturbed, size=(256, 256), mode='bilinear')
        #     driving_mask_int_disturbed = driving_mask_int_disturbed.data.cpu()
        #     driving_mask_int_disturbed = np.transpose(driving_mask_int_disturbed, [0, 2, 3, 1])
        #     images.append(driving_mask_int_disturbed)
        # if 'fixed_mask' in out:
        #     fixed_mask = out['fixed_mask']
        #     fixed_mask = fixed_mask.repeat(1,3,1,1)
        #     fixed_mask = F.interpolate(fixed_mask, size=(256, 256), mode='bilinear')
        #     fixed_mask = fixed_mask.data.cpu()
        #     fixed_mask = np.transpose(fixed_mask, [0, 2, 3, 1])
        #     images.append(fixed_mask)

        if target is not None:
            target = target.data.cpu()
            target = np.transpose(target, [0, 2, 3, 1])
            images.append(target)

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)

        return image
