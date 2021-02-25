from FERPlusDataset import *
from utils import *
from Net import *
import torch.optim as optim


class Train(object):
    def __init__(self, dataType='Train', trainedEpoch=0, batch_size=4, criterion=softCELoss):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.dataType = dataType
        self.trainedEpoch = trainedEpoch
        self.criterion = criterion
        if self.criterion == weightedSoftCELoss:
            self.weights = torch.tensor(computeWeights(TrainDistribution), requires_grad=False).unsqueeze_(0)
            print(f'weights: {self.weights}')
        else:
            self.weights = None
        if trainedEpoch == 0:
            self.net = Net().to(self.device)
        else:
            self.net = Net()
            self.net.load_state_dict(torch.load('models/net' + str(self.trainedEpoch) + '.pkl'))
            self.net = self.net.to(self.device)
            pass
        self.dataSet = FERPlusDataset(type=dataType, transform=transforms.Compose([Rescale(224), ToTensor()]))
        self.trainDataLoader = DataLoader(self.dataSet, batch_size=batch_size, shuffle=True, num_workers=4)
        self.net.train()
        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.00005)
        self.avgLoss = 0.
        self.oneEpochCnt = len(self.dataSet) // batch_size

    def oneEpoch(self):
        self.avgLoss = 0.
        for idx, batch in enumerate(self.trainDataLoader):
            # img = batch['img'][0]
            # label = batch['label'][0]
            # showSample(img, label)
            # ---------------------
            self.optimizer.zero_grad()
            imgs = batch['img'].to(self.device, dtype=torch.float)
            labels = batch['label'].to(self.device, dtype=torch.float)
            digits = self.net(imgs)
            loss = self.criterion(digits, labels, weights=self.weights)
            # print(f'CE: {softCELoss(digits, labels)} weightedCE: {weightedSoftCELoss(digits, labels)}')
            loss.backward()
            self.optimizer.step()
            self.avgLoss += loss
            if idx % 100 == 0 and idx != 0:
                print(f'[{idx}/{self.oneEpochCnt}] avg loss: {self.avgLoss / 100.}')
                self.avgLoss = 0.
        self.trainedEpoch += 1
        torch.save(self.net.state_dict(), 'models/net' + str(self.trainedEpoch) + '.pkl')
        print(f'trainedEpoch: {self.trainedEpoch}')

    def testAccuracy(self):
        self.net.eval()
        total = 3000
        correct = 0
        for i in range(total):
            sample = self.dataSet[i]
            emotion = showSample(**sample)
            predict = self.net(sample['img'].unsqueeze_(0).to(self.device, dtype=torch.float))
            predict = F.softmax(predict, dim=1)[0]
            predict = predict.cpu().detach().numpy()
            top = heapq.nlargest(1, range(len(predict)), predict.take)
            predict = LABELS[top[0]]
            # print(f'emotion: {emotion} predict: {predict}')
            if emotion == predict:
                correct += 1
        print(f'epoch{self.trainedEpoch} {self.dataType}accuracy: {correct / total}')


def trainOneEpoch():
    train = Train()
    train.oneEpoch()


def trainSeveralEpoch(cnt):
    criterion = softCELoss
    # criterion = weightedSoftCELoss
    train = Train(dataType='Train', trainedEpoch=7, criterion=criterion)
    for i in range(cnt):
        train.oneEpoch()


def evalAccuracy(dataType='Test', trainedEpoch=1):
    train = Train(dataType=dataType, trainedEpoch=trainedEpoch, criterion=weightedSoftCELoss)
    train.testAccuracy()


if __name__ == '__main__':
    trainSeveralEpoch(10)
    for i in range(7, 17):
        evalAccuracy(dataType='Train', trainedEpoch=i)
        evalAccuracy(dataType='Test', trainedEpoch=i)

# Adam, lr:0.0001,
# Accuracy:
# epoch | KLdiv | softCE       | weightedSoftCE |
#   0   | 0.005 | 0.005        |
#   1   | 0.55  | 0.618, 0.598 | 0.545, 0.56    |
#   2   |   *   | 0.669, 0.652 | 0.686, 0.668   |
#   3   |   *   | 0.721, 0.695 | 0.728, 0.697   |
#   4   |       | 0.763, 0.726 |
#   5   |       | 0.789, 0.753 |
#   6   |       | 0.807, 0.76  |                |
#   7   |       |      , 0.77
# ------lr:0.00005

# 14 is the best    0.788
# Problem: people show and stuck
