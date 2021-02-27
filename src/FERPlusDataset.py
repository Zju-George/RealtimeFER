from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.utils import *
# Ignore warnings
import warnings

warnings.filterwarnings('ignore')


class FERPlusDataset(Dataset):
    def __init__(self, type='Train', transform=None):
        self.type = type
        self.dataDir, self.csv = dataPath[type]
        self.data = pd.read_csv(self.csv, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imgName = os.path.join(self.dataDir, self.data.iloc[idx, 0])
        # print(f'[Debug] imgName: {imgName}')
        img = np.expand_dims(io.imread(imgName), 2)
        img = np.concatenate((img, img, img), axis=2)
        label = self.data.iloc[idx, 2:]
        label = np.asarray(label)
        sample = {'img': img, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample


def countSamples(dataset):
    classSamples = np.zeros(10, dtype=np.int)
    for i in range(len(dataset)):
        labelIndex = LABELS.index(showSample(**dataset[i]))
        assert 0 <= labelIndex <= 10
        classSamples[labelIndex] += 1
        # print(classSamples)
    print(f'{dataset.type}Data class distribution: {classSamples}')
    return classSamples


if __name__ == '__main__':
    dataset = FERPlusDataset(type="Train", transform=transforms.Compose([Rescale(224), ToTensor()]))
    countSamples(dataset)
