import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, Hourglass, HourglassNoRes
# import cv2
import imgaug.augmenters as iaa
import numpy as np

class Generator(nn.Module):
    def __init__(self, num_channels, block_expansion, max_features, num_down_blocks, num_bottleneck_blocks, th):
        super(Generator, self).__init__()

        self.first = SameBlock2d(5, 256, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        
        self.num_channels = num_channels
        self.hourglass = Hourglass(block_expansion=64, in_features=8, max_features=1024, num_blocks=5)
        self.mask_fix_net = HourglassNoRes(block_expansion=64, in_features=5, max_features=1024, num_blocks=5)
        self.ref = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(7, 7), padding=(3, 3))
        self.final_hourglass = nn.Conv2d(in_channels=self.hourglass.out_filters, out_channels=3, kernel_size=(7, 7),padding=(3, 3))
        self.th = th

    def ResizePartsOfImage(self, mask, horizontal):
        pivots = np.sort(np.random.choice(np.arange(15,46), 5, replace=False))
        pivots = np.append(pivots, [64]) 
        parts = []
        last_pivot = 0
        for pivot in pivots:
            if horizontal:
                current_part = mask[:, :, last_pivot:pivot]
                aug = iaa.Sequential([
                    #iaa.Resize({"width": (0.5, 1.5)})
                    iaa.Affine(scale={"x": (0.75, 1.25)})
                ])
                current_res = aug(images=current_part)
                parts.append(current_res)
            else:
                current_part = mask[:, last_pivot:pivot]
                aug = iaa.Sequential([
                    iaa.Affine(scale={"y": (0.75, 1.25)})
                ])
                current_res = aug(images=current_part)
                parts.append(current_res)

            last_pivot = pivot

        if horizontal:
            parts = np.concatenate(parts, axis=2)
            aug = iaa.Sequential([
                    iaa.Affine(scale={"y": (0.75, 1.25)})
                ])
            parts = aug(images=parts)
        else:
            parts = np.concatenate(parts, axis=1)
            aug = iaa.Sequential([
                    iaa.Affine(scale={"x": (0.75, 1.25)})
                ])
            parts = aug(images=parts)
        
        return parts

    def disturbe_mask(self, mask):
        t = mask.dtype

        result = mask.permute(0,2,3,1).detach().clone() * 255
        result = np.array(result.cpu())
        result = result.astype(np.uint8)

        result = self.ResizePartsOfImage(result, True)
        result = self.ResizePartsOfImage(result, False)
        result[result<=self.th]=0

        self.aug = iaa.Sequential([
            iaa.AdditivePoissonNoise(20)
        ])

        result = self.aug(images=result)
        result = torch.from_numpy(result).permute(0,3,1,2).cuda().type(t)
        result = (result / 255.0)

        return result.detach().clone()


    def disturbe_mask_anim(self, mask):
        t = mask.dtype

        result = mask.permute(0,2,3,1).detach().clone() * 255
        result[result<=self.th]=0

        result = np.array(result.cpu())
        result = result.astype(np.uint8)

        self.aug = iaa.Sequential([
            iaa.Affine(scale=0.75),
        ])

        result = self.aug(images=result)
        result = torch.from_numpy(result).permute(0,3,1,2).cuda().type(t)
        result = (result / 255.0)

        return result.detach().clone()

    def forward(self, source_image, driving_image, mask_driving, mask_source, mask_driving2, predict_mask, animate):
        if animate:
            return self.animate(source_image, driving_image, mask_source=mask_source, mask_driving=mask_driving)
        
        output_dict = {}

        bs = source_image.shape[0]

        source_mask_int = F.pad(input=mask_source, pad=(3, 3, 3, 3), mode='constant', value=0)
        driving_mask_int = F.pad(input=mask_driving, pad=(3, 3, 3, 3), mode='constant', value=0)
      
        e_h, e_w = driving_mask_int.shape[2:]
        s_i = F.interpolate(source_image, size=(e_h, e_w), mode='bilinear')

        if predict_mask:
            driving_mask_int_disturbed = self.disturbe_mask(driving_mask_int)
            input_disturbed_mask = torch.cat((s_i, source_mask_int.detach()), dim=1)
            input_disturbed_mask = torch.cat((input_disturbed_mask, driving_mask_int_disturbed), dim=1)
            fixed_mask = self.mask_fix_net(input_disturbed_mask)
            fixed_mask = self.ref(fixed_mask) / 0.1
            fixed_mask = F.sigmoid(fixed_mask)

            with torch.no_grad():
                driving2_mask_int = F.pad(input=mask_driving2, pad=(3, 3, 3, 3), mode='constant', value=0)

                th = self.th / 255.0
                driving2_mask_int = self.disturbe_mask_anim(driving2_mask_int)

                aaa = torch.cat((s_i, source_mask_int.detach()), dim=1)
                input_driving2 = torch.cat((aaa, driving2_mask_int.detach()), dim=1)
                fixed_d2_mask = self.mask_fix_net(input_driving2)
                fixed_d2_mask = self.ref(fixed_d2_mask) / 0.1
                fixed_d2_mask = F.sigmoid(fixed_d2_mask)

                d2_out = torch.cat((aaa, fixed_d2_mask.detach()), dim=1)
                d2_out = self.first(d2_out)
                d2_out = self.bottleneck(d2_out)
                for i in range(len(self.up_blocks)):
                    d2_out = self.up_blocks[i](d2_out)
                d2_out = self.final(d2_out)
                final_d2 = F.sigmoid(d2_out)

        out = torch.cat((s_i, source_mask_int), dim=1)
        out = torch.cat((out, driving_mask_int), dim=1)
        out = self.first(out)
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        first_phase_prediction = F.sigmoid(out)

        driving_mask_int_detached = driving_mask_int.detach().clone()
        source_mask_int = F.interpolate(source_mask_int, size=(256, 256), mode='bilinear')
        driving_mask_int = F.interpolate(driving_mask_int, size=(256, 256), mode='bilinear')

        input = torch.cat((source_image, source_mask_int.detach()), dim=1)
        input = torch.cat((input, first_phase_prediction.detach()), dim=1)
        input = torch.cat((input, driving_mask_int.detach()), dim=1)

        out = self.hourglass(input)
        out = self.final_hourglass(out)
        second_phase_prediction = F.sigmoid(out)

        output_dict['driving_mask'] = driving_mask_int.repeat(1,3,1,1)
        output_dict['source_mask'] = source_mask_int.repeat(1,3,1,1)

        if predict_mask:
            driving2_mask_int = F.interpolate(driving2_mask_int, size=(256, 256), mode='bilinear')
            output_dict['driving2_mask'] = driving2_mask_int.repeat(1,3,1,1)
            output_dict['fixed_mask'] = fixed_mask
            fixed_d2_mask = F.interpolate(fixed_d2_mask, size=(256, 256), mode='bilinear')
            output_dict['fixed_d2_mask'] = fixed_d2_mask.repeat(1,3,1,1)
            output_dict['final_d2'] = final_d2
            output_dict['driving_mask_int_detached'] = driving_mask_int_detached
            output_dict['driving_mask_int_disturbed'] = driving_mask_int_disturbed

        output_dict["first_phase_prediction"] = first_phase_prediction
        output_dict["second_phase_prediction"] = second_phase_prediction

        return output_dict

    def animate(self, source_image, driving_image, mask_driving, mask_source):
        output_dict = {}

        bs = source_image.shape[0]
        source_mask_int = F.pad(input=mask_source, pad=(3, 3, 3, 3), mode='constant', value=0)        
        driving_mask_int = F.pad(input=mask_driving, pad=(3, 3, 3, 3), mode='constant', value=0)
        driving_mask_int_dist = self.disturbe_mask_anim(driving_mask_int)
        
        e_h, e_w = driving_mask_int_dist.shape[2:]
        s_i = F.interpolate(source_image, size=(e_h, e_w), mode='bilinear')

        aaa = torch.cat((s_i, source_mask_int), dim=1)
        input_disturbed_mask = torch.cat((aaa, driving_mask_int_dist), dim=1)
        fixed_mask = self.mask_fix_net(input_disturbed_mask)
        fixed_mask = self.ref(fixed_mask) / 0.1
        fixed_mask = F.sigmoid(fixed_mask)

        out = torch.cat((aaa, fixed_mask), dim=1)
        out = self.first(out)
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        first_phase_prediction = F.sigmoid(out)

        source_mask_int = F.interpolate(source_mask_int, size=(256, 256), mode='bilinear')
        driving_mask_int = F.interpolate(driving_mask_int, size=(256, 256), mode='bilinear')
        driving_mask_int_dist = F.interpolate(driving_mask_int_dist, size=(256, 256), mode='bilinear')
        fixed_mask = F.interpolate(fixed_mask, size=(256, 256), mode='bilinear')

        input = torch.cat((source_image, source_mask_int.detach()), dim=1)
        input = torch.cat((input, first_phase_prediction.detach()), dim=1)
        input = torch.cat((input, fixed_mask.detach()), dim=1)

        out = self.hourglass(input)
        out = self.final_hourglass(out)
        second_phase_prediction = F.sigmoid(out)

        output_dict['driving_mask'] = driving_mask_int.repeat(1,3,1,1)
        output_dict['driving_mask_dist'] = driving_mask_int_dist.repeat(1,3,1,1)
        output_dict['source_mask'] = source_mask_int.repeat(1,3,1,1)
        output_dict['fixed_mask'] = fixed_mask
        output_dict["first_phase_prediction"] = first_phase_prediction
        output_dict["second_phase_prediction"] = second_phase_prediction

        return output_dict
