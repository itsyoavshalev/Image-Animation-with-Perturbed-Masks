from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d
from torchvision import models
import numpy as np
from torch.autograd import grad


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict

class GeneratorFullModel(torch.nn.Module):
    def __init__(self, mask_generator, generator, train_params):
        super(GeneratorFullModel, self).__init__()
        self.mask_generator = mask_generator
        self.generator = generator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']
        self.l1 = nn.L1Loss()

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    def forward(self, x, predict_mask):
        mask_source = self.mask_generator(x['source'])
        mask_driving = self.mask_generator(x['driving'])
        mask_driving2 = self.mask_generator(x['driving2'])

        generated = self.generator(x['source'], x['driving'], mask_source=mask_source, mask_driving=mask_driving, mask_driving2=mask_driving2, predict_mask=predict_mask, animate=False)
        generated.update({'r_mask_source': mask_source, 'r_mask_driving': mask_driving, 'r_mask_driving2': mask_driving2})

        loss_values = {}
        
        if predict_mask:
            loss_values['mask_correction'] = 100 * self.l1(generated['fixed_mask'] ,generated['driving_mask_int_detached'])
        else:
            loss_values['mask_correction'] = torch.zeros(10).cuda()

        pyramide_real = self.pyramid(x['target'])
        pyramide_first_generated = self.pyramid(generated['first_phase_prediction'])
        pyramide_second_generated = self.pyramid(generated['second_phase_prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total1 = 0
            value_total2 = 0
            for scale in self.scales:
                x_vgg1 = self.vgg(pyramide_first_generated['prediction_' + str(scale)])
                x_vgg2 = self.vgg(pyramide_second_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value1 = torch.abs(x_vgg1[i] - y_vgg[i].detach()).mean()
                    value2 = torch.abs(x_vgg2[i] - y_vgg[i].detach()).mean()
                    value_total1 += self.loss_weights['perceptual'][i] * value1
                    value_total2 += self.loss_weights['perceptual'][i] * value2

            loss_values['perceptual1'] = value_total1
            loss_values['perceptual2'] = value_total2

        return loss_values, generated