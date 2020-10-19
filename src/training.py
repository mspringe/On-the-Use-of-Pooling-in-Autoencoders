"""
This module provides a training script for all model variations considered.

Args-parsing is defined in :func:`src.training.arg_parser`.
"""
import sys; sys.path.append('../')
from src import data
from src import nn
from src.os_utils import join
import torch
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams.update({'font.size': 48})
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from tqdm import tqdm
import os
import argparse
import cv2
import json
from colorama import Fore


class Trainer:
    """
    This class handles the training od any :class:`src.nn.ConvNet` model.
    """

    def __init__(self, model, dset, device, s_batch=32, model_name='model'):
        """
        :param model: ConvNet model
        :param dset: data set to use in training
        :param device: GPU device
        :param s_batch: batch size
        :param model_name: name of model
        """
        self.model = model
        self.dset = dset
        self.device = device
        self.s_batch = s_batch
        self.model_name = model_name
        self.print_c = Fore.GREEN
        if not os.path.isdir(join(f'logs/{self.model_name}')):
            os.makedirs(join(f'logs/{self.model_name}'))

    def log_step(self, y, batch, targets, str_it_count):
        """
        Saving a sample of the batch.

        :param y: predicted ab channels
        :param batch: true L channels
        :param targets: true ab channels
        :param str_it_count: prefix of the saved images (embodies the current optimization step)
        :return:
        """
        # index of sample for displayal
        k = np.random.randint(0, y.shape[0])
        # BW images
        if y.shape[1] == 1:
            img = y[k].cpu().numpy()[0]
            img = np.repeat(img[:,:,None], 3, axis=2)
            orig = targets[k].cpu().data.numpy()[0]
            orig = np.repeat(orig[:,:,None], 3, axis=2)
            # logging images
            cv2.imwrite(join(f'logs/{self.model_name}/{str_it_count}_est.png'), img)
            cv2.imwrite(join(f'logs/{self.model_name}/{str_it_count}_orig.png'), orig)
            # assemble the plot iamge
            img = cv2.resize(img, (128, 128))
            orig = cv2.resize(orig, (128, 128))
            p_I = np.vstack([orig, img])
        # Lab images
        else:
            ab = y[k].cpu().data.numpy()
            orig_ab = targets[k].cpu().data.numpy()
            L = batch[k].cpu().data.numpy()
            # estimated image and fmaps
            est_Lab = np.vstack([L, ab]) * 255.
            est_RGB = cv2.cvtColor(est_Lab.transpose(1,2,0).astype(np.uint8), cv2.COLOR_Lab2RGB) / 255.
            est_Lab = np.hstack(est_Lab[1:]) / 255.
            est_ab_BW = np.repeat(est_Lab[:,:,None], 3, axis=2)
            # original image and fmaps
            orig_Lab = np.vstack([L, orig_ab]) * 255.
            orig_RGB = cv2.cvtColor(orig_Lab.transpose(1,2,0).astype(np.uint8), cv2.COLOR_Lab2RGB) / 255.
            orig_Lab = np.hstack(orig_Lab[1:]) / 255.
            orig_ab_BW = np.repeat(orig_Lab[:,:,None], 3, axis=2)
            # logging images
            cv2.imwrite(join(f'logs/{self.model_name}/{str_it_count}_colour_est.png'),
                        cv2.cvtColor((est_RGB*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(join(f'logs/{self.model_name}/{str_it_count}_ab_est.png'),
                        (est_ab_BW*255).astype(np.uint8))
            cv2.imwrite(join(f'logs/{self.model_name}/{str_it_count}_colour_orig.png'),
                        cv2.cvtColor((orig_RGB*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(join(f'logs/{self.model_name}/{str_it_count}_ab_orig.png'),
                        (orig_ab_BW*255).astype(np.uint8))
            # assemble compressed visual summary (the plot image)
            est_RGB = cv2.resize(est_RGB, (128, 128))
            orig_RGB = cv2.resize(orig_RGB, (128, 128))
            est_ab_BW = cv2.resize(est_ab_BW, (2*128, 128))
            orig_ab_BW = cv2.resize(orig_ab_BW, (2*128, 128))
            p_I = np.vstack([np.hstack([orig_RGB, orig_ab_BW]),
                             np.hstack([est_RGB,  est_ab_BW])])
        return p_I

    def plot_progress(self, error_series, plt_imgs, plt_imgs_t):
        """
        This method generates an informative plot, displaying the loss over time along logged images.

        :param error_series: loss values over time (/ steps)
        :param plt_imgs: logged images
        :param plt_imgs_t: time (/ step) at which respective images have been logged
        :return:
        """
        fig, ax = plt.subplots(figsize=(160, 40))
        ax.plot(np.arange(1, 1+len(error_series)), error_series, alpha=0.9)
        for i, p_I in enumerate(plt_imgs):
            imagebox = OffsetImage(p_I, zoom=1.)
            x = plt_imgs_t[i]
            ab = AnnotationBbox(imagebox, (x, error_series[x-1]),
                                xybox=(0., 256.),
                                xycoords='data',
                                boxcoords="offset points",
                                arrowprops={'arrowstyle':'->'}) 
            ax.add_artist(ab)
        plt.savefig(join(f'logs/{self.model_name}/progress.png'), bbox_inches='tight', dpi=50)
        plt.close(fig)
    
    def opt_loss(self):
        """
        :return: optimizer and loss function
        """
        # defining optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5, betas=(0.9, 0.99), weight_decay=0.00005)
        # define the loss function
        f_loss = torch.nn.L1Loss()
        return optimizer, f_loss

    def data_iter(self):
        """
        :return: iterator for the data set
        """
        # construct the data iterator
        if not isinstance(self.dset, data.RAMDataSet):
            data_loader = torch.utils.data.DataLoader(self.dset, batch_size=self.s_batch, shuffle=True, num_workers=8)
            data_iter = iter(data_loader)
        else:
            data_iter = data.RAMDataSetIter(self.dset, s_batch=self.s_batch)
        return data_iter

    def train_convnet(self, iterations=int(1e5)):
        """
        Training the model.

        :param iterations: number of optimization steps to train for
        :return: trained model
        """
        optimizer, f_loss = self.opt_loss()
        data_iter = self.data_iter()
        # objects to keep track of the training-progress
        prog_bar = tqdm(total=iterations, bar_format=f'{self.print_c}{self.model_name} training {Fore.RESET}    ' +
                                                     "{l_bar}%s{bar}%s{r_bar}" % (self.print_c, Fore.RESET))
        error_series = []
        plt_imgs = []
        plt_imgs_t = []
        # move model to GPU-device
        self.model = self.model.to(self.device)
        # the training loop
        it_count = 0
        while it_count < iterations:
            # obtain batch and targets
            try:
                batch, targets = data_iter.next()
            except StopIteration:
                if not isinstance(self.dset, data.RAMDataSet):
                    data_iter = self.data_iter()
                else:
                    data_iter.reset()
                batch, targets = data_iter.next()
            # move data to GPU-device
            batch = batch.to(self.device)
            targets = targets.to(self.device)
            # forward pass
            y = self.model(batch)
            y = torch.nn.functional.hardtanh(y, min_val=0, max_val=1)
            # backward propagation and optimization step
            error = f_loss(y, targets)
            error.backward()
            optimizer.step()
            it_count += 1
            # keeping track of loss values
            error_series.append(error.cpu().data.item())
            # saving some images every 5%, to visualize learning process
            if (100. * (it_count)/iterations) % 5. == 0:
                # logging images
                str_it_count = (len(str(iterations)) - len(str(it_count))) * '0' + str(it_count)
                p_I = self.log_step(y, batch, targets, str_it_count)
                # keeping track of logged images
                plt_imgs.append(p_I)
                plt_imgs_t.append(it_count)
                # construct the plot
                self.plot_progress(error_series, plt_imgs, plt_imgs_t)
            # loop completed, updating the progress bar
            prog_bar.update(1)
        self.model.cpu()
        prog_bar.close()
        return self.model


def print_info(args):
    """
    Printing a table that summarizes the training configuration.

    :param args: arguments passed to the script
    """
    print(f'''{Fore.WHITE}
Training has been initialized as follows:
\tdata set:                 {args.dset}
\twidth of images:          {args.w}
\theight of images:         {args.h}
\titerations:               {args.iterations}
\tbatch size:               {args.s_batch}
          {Fore.RESET}''')


def arg_parser():
    """
    :return: args-parser with the following arguments

    ========== ================ ==================
    option     semantic         default
    ========== ================ ==================
    --w        width of images  128
    --h        height of images 128
    --s_batch  size of batches  16
    --dset     name of data set daisy
    ========== ================ ==================
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--w', default=128, help='width of image', type=int)
    ap.add_argument('--h', default=128, help='height of image', type=int)
    ap.add_argument('--s_batch', default=16, help='size of batch', type=int)
    ap.add_argument('--iterations', default=int(1e5), help='number of optimization steps', type=int)
    ap.add_argument('--dset', default='daisy', help='dataset to use', type=str)
    return ap
 

if __name__ == '__main__':
    # checking for GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args parsing
    args = arg_parser().parse_args()
    print_info(args)
    w, h = args.w, args.h
    s_batch = args.s_batch
    iterations = args.iterations
    # loading the dataset
    if args.dset == 'daisy':
        dset_kwargs = dict(path=join('/media/ssd1/data_sets/flowers/daisy/train'), width=w, height=h, black_white=False,
                           suffix='.jpg')
        dset = data.RAMDataSet(**dset_kwargs)
        dset_kwargs['path'] = join('/media/ssd1/data_sets/flowers/daisy/test')
        c_out = 2
    else:
        raise NotImplementedError(f'dataset {args.dset} is unknown')
    # training models with 0-3 pooling operations
    for n_pooling in [0, 1, 2, 3][::-1]:
        # keeping track of the data set- and model parameters
        model_kwargs = dict(c_out=c_out, img_w=w, img_h=h, act='ReLU', n_pooling=n_pooling)
        kwargs = dict(model=model_kwargs, dset=dset_kwargs)
        model_name = f'ConvNet_pool{n_pooling}'
        # model initialization
        print(f'\n{Fore.BLUE}Initializing model with {n_pooling} pooling- and upsampling layers{Fore.RESET}')
        model = nn.ConvNet(**model_kwargs)
        print(f'{Fore.BLUE}Number of model parameters: ' +
              f'{Fore.WHITE}{sum(p.numel() for p in model.parameters())}{Fore.RESET}')
        # training the model
        trainer = Trainer(model, dset, device, s_batch=s_batch, model_name=model_name)
        trainer.train_convnet(iterations)
        # storing the models state
        print(f'{Fore.BLUE}Training finished, saving the model to ' +
              f'{Fore.WHITE}{join(f"models/{model_name}/state_dict.pth")}{Fore.RESET}')
        if not os.path.isdir(join(f'models/{model_name}')):
            os.makedirs(join(f'models/{model_name}'))
        torch.save(model.state_dict(), join(f'models/{model_name}/state_dict.pth'))
        # storing information on models initialization and the data set
        with open(join(f'models/{model_name}/kwargs.json'), 'w') as f_out:
            json.dump(kwargs, f_out)
        print(f'{Fore.GREEN}Model has been saved{Fore.RESET}')
