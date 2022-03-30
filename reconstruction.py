import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio
from sync_batchnorm import DataParallelWithCallback


def reconstruction(config, generator, mask_generator, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        epoch = Logger.load_cpk(checkpoint, generator=generator, mask_generator=mask_generator)
        print('checkpoint:' + str(epoch))
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    loss_list = []
    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        mask_generator = DataParallelWithCallback(mask_generator)

    generator.eval()
    mask_generator.eval()

    recon_gen_dir = './log/recon_gen'
    os.makedirs(recon_gen_dir,exist_ok=False)

    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            predictions = []
            visualizations = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            mask_source = mask_generator(x['video'][:, :, 0])

            video_gen_dir = recon_gen_dir + '/' + x['name'][0]
            os.makedirs(video_gen_dir,exist_ok=False)
            
            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]
                driving = x['video'][:, :, frame_idx]
                mask_driving = mask_generator(driving)
                out = generator(source, driving, mask_source=mask_source, mask_driving=mask_driving, mask_driving2=None, animate=False, predict_mask=False)
                out['mask_source'] = mask_source
                out['mask_driving'] = mask_driving
                
                predictions.append(np.transpose(out['second_phase_prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source, driving=driving, target=None, out=out, driving2=None)
                visualizations.append(visualization)

                loss_list.append(torch.abs(out['second_phase_prediction'] - driving).mean().cpu().numpy())

                frame_name = str(frame_idx).zfill(7) + '.png'
                second_phase_prediction = out['second_phase_prediction'].data.cpu().numpy()
                second_phase_prediction = np.transpose(second_phase_prediction, [0, 2, 3, 1])
                second_phase_prediction = (255 * second_phase_prediction).astype(np.uint8)
                imageio.imsave(os.path.join(video_gen_dir, frame_name), second_phase_prediction[0])

            predictions = np.concatenate(predictions, axis=1)

            image_name = x['name'][0] + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    print("Reconstruction loss: %s" % np.mean(loss_list))
    
