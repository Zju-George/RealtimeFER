import matplotlib.pyplot as plt
import os
from skimage import io
import pandas as pd
import numpy as np
import heapq
import torch
from skimage import transform
np.set_printoptions(suppress=True)

LABELS = ('neutral', 'happiness', 'surprise', 'sadness', 'anger',
          'disgust', 'fear', 'contempt', 'unknown', 'NF')
TestDistribution = [1258, 928, 447, 446, 321, 20, 97, 28, 28, 0]
TrainDistribution = [10295, 7526, 3557, 3530, 2463, 191, 655, 168, 171, 2]

TrainDir = 'data/FERPlus-master/data/FER2013Train'
TestDir = 'data/FERPlus-master/data/FER2013Test'
ValidDir = 'data/FERPlus-master/data/FER2013Valid'
dataPath = {'Train': (TrainDir, os.path.join(TrainDir, 'label.csv')),
            'Test': (TestDir, os.path.join(TestDir, 'label.csv')),
            'Valid': (ValidDir, os.path.join(ValidDir, 'label.csv')),
            }


def showSample(img=None, label=None, show=False):
    if type(img) == torch.Tensor:
        assert type(label) == torch.Tensor
        img = img.detach().numpy().transpose((1, 2, 0))
        label = label.detach().numpy()
    ax = plt.subplot(1, 1, 1)
    # print(f'[DEBUG] label: {label}')
    top2 = heapq.nlargest(2, range(len(label)), label.take)
    emotion = [LABELS[top2[0]], LABELS[top2[1]]]
    title = list(zip(emotion, label[top2]))
    ax.set_title('emotion top2: {}'.format(title))
    if show:
        plt.imshow(img, cmap='gray')
        plt.show()
    return emotion[0]


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(img, (new_h, new_w))
        return {'img': img, 'label': label}


class ToTensor(object):
    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        assert np.sum(label) == 10
        img = img.transpose((2, 0, 1))
        label = label * 1. / np.sum(label)
        label = label.astype(np.float64)
        return {'img': torch.from_numpy(img), 'label': torch.from_numpy(label)}


def computeWeights(dataDistribution):
    nSamples = np.array(dataDistribution).sum(axis=0)
    nClasses = len(dataDistribution)
    weights = np.sqrt(nSamples/nClasses * 1./dataDistribution)
    weights[9] = 1.
    return weights


def main():
    data = pd.read_csv(os.path.join(TrainDir, 'label.csv'), header=None)
    n = 0
    imgName = os.path.join(TrainDir, data.iloc[n, 0])
    img = io.imread(imgName)
    img = np.expand_dims(img, 2)
    img = np.concatenate((img, img, img), axis=2)
    label = data.iloc[n, 2:]
    label = np.asarray(label)
    showSample(img=img, label=label, show=True)


if __name__ == '__main__':
    print(f'weights: {computeWeights(TrainDistribution)}')
