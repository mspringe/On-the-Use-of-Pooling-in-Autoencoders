"""
This module provides a script to evaluate all models, trained prior to the execution of this script.
"""
import sys; sys.path.append('../')
import src.data as data
import src.nn as nn
from src.os_utils import join
import torch
import numpy as np
from colorama import Fore
import os
import json
import tqdm
import matplotlib.pyplot as plt


class Evaluator:
    """
    This class provides evaluation of :class:`src.nn.ConvNet` models
    """

    def __init__(self, model, dset, device=torch.device('cpu'), s_batch=16, model_name='model'):
        """
        :param model: model for evaluation
        :param dset: data set to evaluate on
        :param device: GPU device
        :param s_batch: size of batches
        :param model_name: name of the model
        """
        self.model = model
        self.dset = dset
        self.device = device
        self.model_name = model_name
        self.print_c = Fore.GREEN
        self.s_batch = s_batch

    def data_iter(self):
        """"
        Construction of the data iterator

        :return: data iterator for the evaluators dataset
        """
        if not isinstance(self.dset, data.RAMDataSet):
            data_loader = torch.utils.data.DataLoader(self.dset, batch_size=self.s_batch, shuffle=True, num_workers=8)
            data_iter = iter(data_loader)
        else:
            data_iter = data.RAMDataSetIter(self.dset, s_batch=self.s_batch)
        return data_iter

    def accuracy(self, ab_est, ab_target, max_diff=0.1):
        """
        Calculates the accuracy, based on the mean :math:`l_{1mean}` of the L1-loss.
        Accuracy is defined as :math:`1 - l_{1mean}`.

        :param ab_est: estamated a and b channels
        :param ab_target: true a and b channels
        :return: mean accuracy on batch
        """
        diffs = torch.abs(ab_est - ab_target)
        in_range = (diffs <= max_diff)
        in_range.shape
        return in_range.cpu().data.numpy().astype(float).mean()

    def hist(self, ab_est, ab_target, inf_diff=0.1):
        """
        Calculating accuracies for a variety of upper bounds :math:'b_i, \ldots, b_N \in ^N: b_i < inf_{diff}',
        appropriate bounds are generated automatically, only :math:`inf_{diff}` required.
        Accuracies are stored in a histogram, sorted ascending by the upper bounds value.

        :param ab_est: estimated a and b channels
        :param ab_target: true a and b channels
        :param inf_diff: infimum of upper bounds
        :return: bounds and histogram of accuracies with respective upper bounds
        """
        bounds = np.linspace(0, inf_diff)
        bins = [self.accuracy(ab_est, ab_target, i) for i in bounds]
        return bounds, np.array(bins)

    def evaluate(self, bounds=[0.01, 0.05, 0.1]):
        """"
        Evaluation of the model on the data set.

        :return: mean accuracy on data set
        """
        data_iter = self.data_iter()
        # setting up the evaluation environment
        with torch.no_grad():
            self.model.eval()
            self.model.to(self.device)
            accs = []
            for max_diff in bounds:
                vals = []
                prog_bar = tqdm.tqdm(total=len(self.dset),
                                     bar_format=f'{self.print_c}{self.model_name} evaluation [0, {max_diff}]    ' +
                                                "%s{l_bar}%s{bar}%s{r_bar}" % (Fore.RESET, self.print_c, Fore.RESET))
                while True:
                    # fetch data
                    try:
                        batch, targets  = data_iter.next()
                    except StopIteration:
                        break
                    batch = batch.to(self.device)
                    targets = targets.to(self.device)
                    # apply model
                    y = torch.nn.functional.hardtanh(self.model(batch), min_val=0, max_val=1)
                    # calculate accuracy
                    acc = self.accuracy(y.cpu(), targets.cpu(), max_diff=max_diff)
                    # store for numeric stability
                    vals.append(acc)
                    prog_bar.update(len(y))
                prog_bar.close()
                accs.append(np.mean(vals))
                data_iter.reset()
            self.model.cpu()
        return accs
    
    def evaluate_hist(self, inf_diff=10):
        """
        Evaluating how many values lie within certain ranges.

        :param inf_diff: infimum of upper bounds for the histogram
        :return: histogram of percentages of pairwise distances within a discrete range, bounded by inf_diff
        """
        data_iter = self.data_iter()
        prog_bar = tqdm.tqdm(total=len(self.dset),
                             bar_format=f'{self.print_c}{self.model_name} hist    ' +
                                        "%s{l_bar}%s{bar}%s{r_bar}" % (Fore.RESET, self.print_c, Fore.RESET))
        hists = []
        bounds = None
        # setting up the evaluation environment
        with torch.no_grad():
            self.model.eval()
            self.model.to(self.device)
            while True:
                # fetch data
                try:
                    batch, targets  = data_iter.next()
                except StopIteration:
                    break
                batch = batch.to(self.device)
                targets = targets.to(self.device)
                # apply model
                y = torch.nn.functional.hardtanh(self.model(batch), min_val=0, max_val=1)
                # calculate accuracies
                bounds, h = self.hist(y.cpu(), targets.cpu(), inf_diff=inf_diff)
                # storing for numeric stability
                hists.append(h)
                prog_bar.update(len(y))
            self.model.cpu()
        prog_bar.close()
        hist = np.mean(hists, axis=0)
        return bounds, hist


if __name__ == '__main__':
    # establishing GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # evaluating the models
    accs = {}
    print(f'\n{Fore.BLUE} evaluating all models stored in {Fore.WHITE}models/{Fore.RESET}')
    for model_name in sorted([p for p in os.listdir('models') if os.path.isdir(f'models/{p}')]):
        # loading the model and corresponding dataset
        with open(join(f'models/{model_name}/kwargs.json'), 'r') as f_kwargs:
            kwargs = json.load(f_kwargs)
        state_dict = torch.load(f'models/{model_name}/state_dict.pth', map_location=torch.device('cpu'))
        dset = data.RAMDataSet(**kwargs['dset'])
        model = nn.ConvNet(**kwargs['model'])
        model.load_state_dict(state_dict)
        # initializing the evaluator
        evaluator = Evaluator(model, dset, device, model_name=model_name)
        # storing and saving some discrete upper bounds, for a quick look/ preview
        accs[model_name] = evaluator.evaluate()
        print(f'{Fore.BLUE}accuracy {model_name}: {Fore.WHITE}{accs[model_name]}{Fore.RESET}')
        # constructing a plot visualizing the distribution of abs. differences, that lie within certain ranges
        bounds, hist = evaluator.evaluate_hist()
        l, = plt.plot(bounds, hist)
        l.set_label(model_name)
    # path sanity for upcoming I/O operations
    if not os.path.isdir('logs'):
        os.makedirs('logs')
    # saving the plot
    plt.legend()
    plt.xlabel('upper bound on pairwise abs. differences')
    plt.ylabel('% of values that the bound applies to')
    plt.savefig(join(f'logs/hst'))
    # also logging the preview on the distribution of abs. differences, that lie within certain ranges
    with open(join(f"logs/results_{'&'.join(sorted(accs.keys()))}"), 'w') as f_out:
        json.dump(accs, f_out)
