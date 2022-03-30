import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from frames_dataset import PairedDataset
from logger import Logger, Visualizer
import imageio
from scipy.spatial import ConvexHull
import numpy as np
from sync_batchnorm import DataParallelWithCallback


def animate(config, generator, mask_generator, checkpoint, log_dir, dataset):
    log_dir = os.path.join(log_dir, 'animation')
    png_dir = os.path.join(log_dir, 'png')
    animate_params = config['animate_params']

    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, mask_generator=mask_generator)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        mask_generator = DataParallelWithCallback(mask_generator)

    generator.eval()
    mask_generator.eval()

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            visualizations = []

            driving_video = x['driving_video']
            source_frame = x['source_video'][:, :, 0, :, :]

            source_mask = mask_generator(source_frame)

            for frame_idx in range(driving_video.shape[2]):
                driving_frame = driving_video[:, :, frame_idx]
                driving_mask = mask_generator(driving_frame)
                out = generator(source_frame, driving_frame, mask_source=source_mask, mask_driving=driving_mask, animate=True, mask_driving2=None, predict_mask=False)

                out['driving_mask'] = driving_mask
                out['source_mask'] = source_mask

                predictions.append(np.transpose(out['first_phase_prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
                predictions.append(np.transpose(out['second_phase_prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source_frame, driving=driving_frame, driving2=None, target=None, out=out)
                visualization = visualization
                visualizations.append(visualization)

            predictions = np.concatenate(predictions, axis=1)
            result_name = "-".join([x['driving_name'][0], x['source_name'][0]])
            # imageio.imsave(os.path.join(png_dir, result_name + '.png'), (255 * predictions).astype(np.uint8))

            image_name = result_name + animate_params['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)
