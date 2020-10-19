"""
This Module provides a script, that saves the models outputs for specific images, stored in 'show_cases/imgs'.
Images will be written to 'show_cases/predictions'.
"""
import sys; sys.path.append('..')
from src import data
from src import nn
from src.os_utils import join
from colorama import Fore
import json
import torch
import cv2
import os
import numpy as np
import tqdm


def write_imgs(model, dset, device, s_batch=1, model_name='model'):
    """
    Writing collages to 'show_cases/predictions'.
    A collage feature the original image the L channel and the predicted image in that order from left to right.

    :param model: model to be used
    :param dset: data set, containing images in 'show_cases/imgs'
    :param device: GPU device
    :param s_batch: size of batch
    :param model_name: name of model
    """
    data_iter = data.RAMDataSetIter(dset, s_batch=s_batch, shuffle=False)
    with torch.no_grad():
        model.to(device)
        model.eval()
        it_count = 0
        prog_bar = tqdm.tqdm(total=len(dset), bar_format=f'{Fore.GREEN}{model_name} show case images {Fore.RESET}    ' +
                                                          "{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET))
        while True:
            try:
                batch, targets = data_iter.next()
            except StopIteration:
                break
            batch = batch.to(device)
            targets = targets.to(device)
            y = torch.nn.functional.hardtanh(model(batch), min_val=0, max_val=1).cpu().data.numpy()
            paths = dset.paths[it_count:it_count+len(y)]
            for i, pred_ab in enumerate(y):
                # lightness channel
                L = batch[i].cpu().data.numpy()
                bw = np.repeat(L*255., 3, axis=0).transpose(1, 2, 0).astype(np.uint8)
                # true a, b channels
                img_ab = targets[i].cpu().data.numpy()
                # ture image
                img_Lab = np.vstack([L, img_ab]) * 255.
                img_Lab = img_Lab.transpose(1, 2, 0).astype(np.uint8)
                img_BGR = cv2.cvtColor(img_Lab, cv2.COLOR_LAB2BGR)
                # estimated image
                pred_Lab = np.vstack([L, pred_ab]) * 255.
                pred_Lab = pred_Lab.transpose(1, 2, 0).astype(np.uint8)
                pred_BGR = cv2.cvtColor(pred_Lab, cv2.COLOR_LAB2BGR)
                # summary: ture image, lightness, estimated image
                summary = np.hstack([img_BGR, bw, pred_BGR])
                # channels: true channels, estimated channels
                channels = np.hstack([np.vstack(img_ab), np.vstack(pred_ab)]) * 255.
                channels = channels.astype(np.uint8)
                # saving summary and channels
                cv2.imwrite(join(f'show_cases/predictions/{model_name}_I{it_count+i+1}.png'), summary)
                cv2.imwrite(join(f'show_cases/predictions/{model_name}_I{it_count+i+1}_channels.png'), channels)
            it_count += 1
            prog_bar.update(len(y))
        model.cpu()
        batch.cpu()
        targets.cpu()
        prog_bar.close()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{Fore.BLUE}generating showcase images{Fore.RESET}')
    for model_name in sorted(os.listdir('models')):
        state_dict = torch.load(join(f'models/{model_name}/state_dict.pth'),
                                map_location=torch.device('cpu'))
        with open(join(f'models/{model_name}/kwargs.json'), 'r') as f_kwargs:
            kwargs = json.load(f_kwargs)
        model = nn.ConvNet(**kwargs['model'])
        model.load_state_dict(state_dict)
        kwargs['dset']['path'] = join('show_cases/imgs/')
        kwargs['dset']['suffix'] = '.jpg'
        dset = data.RAMDataSet(**kwargs['dset'])
        write_imgs(model, dset, device, model_name=model_name)
