import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch

sm = nn.Softmax(dim=1)
kldiv = nn.KLDivLoss()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.fc1 = nn.Linear(1000, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.resnet(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def printNet(net):
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name, param.shape, ' grad: {}'.format(param.grad))


def softCELoss(digits, label, weights=None):
    log_sm_digits = F.log_softmax(digits, dim=1)
    loss = torch.sum(-1. * label.mul(log_sm_digits), dim=1).mean(dim=0)
    return loss


def KLLoss(digits, label):
    log_sm_digits = F.log_softmax(digits, dim=1)
    sm_label = F.softmax(label, dim=1)
    # print(f'output: {log_sm_digits} label: {sm_label}')
    # print(f'loss: {kldiv(log_sm_digits, sm_label)}')
    loss = kldiv(log_sm_digits, sm_label)
    return loss


def weightedSoftCELoss(digits, label, weights=torch.ones((1, 10), requires_grad=False)):
    batch = digits.shape[0]
    device = digits.device
    weights = torch.cat((weights, ) * batch).to(device)
    # print(weights, weights.device, weights.shape, weights.requires_grad)
    log_sm_digits = F.log_softmax(digits, dim=1)
    loss = torch.sum(-1. * weights * label * log_sm_digits, dim=1).mean(dim=0)
    return loss


def main():
    net = Net()
    net.load_state_dict(torch.load('models/net.pkl'))
    net.eval()
    print(f'sm: {sm(net(torch.rand((1, 3, 224, 224))))}')


if __name__ == '__main__':
    # main()

    weightedSoftCELoss(torch.rand((2, 3, 224, 224)), 0)
