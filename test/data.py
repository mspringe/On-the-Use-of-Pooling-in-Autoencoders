import sys
sys.path.append('../')
import unittest
import src.data as data
import matplotlib.pyplot as plt


class TestFastCoco(unittest.TestCase):

    def setUp(self):
        self.coco = data.FastCoco(ann_path='/Users/mspringe/Datasets/annotations/instances_val2017.json',
                                  img_path='/Users/mspringe/Datasets/val2017')

    def test_loader(self):
        img, I = self.coco[1]
        print(img.shape, I.shape)
        plt.imshow(I[0], cmap='bone')
        plt.show()
        plt.imshow(img[0], cmap='bone')
        plt.show()
