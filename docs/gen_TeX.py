import sys; sys.path.append('..')
from src import nn
import torch


def to_str(layer, x, w, h, c):
    spacer = 40
    if isinstance(layer, torch.nn.Conv2d):
        o = layer.out_channels
        return '\ConvLayer{%d}{%d}{%d}{%d}{%d}{%d}{%s}' % (x+spacer, 0, 0, h, w, o, 'blue!30!green!10'), x+2*spacer+o, w, h, o
    elif isinstance(layer, torch.nn.Linear):
        o = layer.out_features
        return '\FCLayer{%d}{%d}{%d}{%d}{%s}' % (x+spacer, 0, 0, o, 'gray!50'), x+2*spacer+1, 1, 1, o
    elif isinstance(layer, nn.MaxGridPool):
        s = 8
        return '\ConvLayer{%d}{%d}{%d}{%d}{%d}{%d}{%s}' % (
            x + spacer, 0, 0, int(h / s), int(w / s), c, 'yellow!90!green!25'), x + 2 * spacer + c, w / s, h / s, c
    elif isinstance(layer, torch.nn.MaxPool2d):
        s = layer.stride
        return '\ConvLayer{%d}{%d}{%d}{%d}{%d}{%d}{%s}' % (x+spacer, 0, 0, int(h/s), int(w/s), c, 'yellow!90!green!25'), x+2*spacer+c, w/s, h/s, c
    elif isinstance(layer, torch.nn.ConvTranspose2d):
        o = layer.out_channels
        return '\ConvLayer{%d}{%d}{%d}{%d}{%d}{%d}{%s}' % (x+spacer, 0, 0, h, w, o, 'blue!20'), x+2*spacer+o, w, h, o
    elif isinstance(layer, nn.UpsampleGrid):
        s = 8, 8
        return '\ConvLayer{%d}{%d}{%d}{%d}{%d}{%d}{%s}' % (x + spacer, 0, 0, int(h * s[0]), int(w * s[1]), c, 'orange!15'), x + 2 * spacer + c, w * s[0], h * s[1], c
    elif isinstance(layer, torch.nn.Upsample):
        s = 2, 2
        return '\ConvLayer{%d}{%d}{%d}{%d}{%d}{%d}{%s}' % (x+spacer, 0, 0, int(h*s[0]), int(w*s[1]), c, 'orange!15'), x+2*spacer+c, w*s[0], h*s[1], c
    elif isinstance(layer, nn.Reshape):
        s = layer.shape_out
        if len(s) > 1:
            return '\ConvLayer{%d}{%d}{%d}{%d}{%d}{%d}{%s}' % (x+spacer, 0, 0, s[1], s[2], s[0], 'white'), x+2*spacer+s[0], s[1], s[2], s[0]
        else:
            return '\FCLayer{%d}{%d}{%d}{%d}{%s}' % (x+spacer, 0, 0, s[0], 'white'), x+2*spacer+1, 1, 1, s[0]

    return None, x, w, h, c



class LayerGen:

    def __init__(self):
        self.x = 0
        self.w = 32
        self.h = 32
        self.c = 1

    def reset(self):
        self.x = 0
        self.w = 32
        self.h = 32
        self.c = 1

    def get_layers(self, model):
        layers = []
        for l in model.children():
            if isinstance(l, torch.nn.Sequential):
                layers += self.get_layers(l)
            else:
                print(l)
                s, self.x, self.w, self.h, self.c = to_str(l, self.x, self.w, self.h, self.c)
                if s is not None:
                    layers.append(s)
        return layers


if __name__ == '__main__':
    lg = LayerGen()
    for n_pool in [0, 1, 2, 3]:
        lg.reset()
        model = nn.ConvNet(img_h=lg.h, img_w=lg.w, n_pooling=n_pool, act='ReLU', c_out=2)
        layers = lg.get_layers(model)
        #layers[-1] = layers[-1][:-len('{blue!20}')] + '{red!10}'
        with open('CNN_template.tex', 'r') as f_template:
            str_LaTeX = ''.join(f_template.readlines())
            half1, half2 = str_LaTeX.split('%#hook#')
            str_LaTeX = half1 + '\n'.join(layers) + half2
        with open(f'TeX/CNN_{n_pool}.tex', 'w') as f_out:
            f_out.write(str_LaTeX)
