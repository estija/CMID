import torch
import torch.nn as nn
import torch.nn.functional as F

model_attributes = {
    'bert': {
        'feature_type': 'text'
    },
    'inception_v3': {
        'feature_type': 'image',
        'target_resolution': (299, 299),
        'flatten': False
    },
    'wideresnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet34': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': False
    },
    'raw_logistic_regression': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': True,
    },
    'bert-base-uncased': {
        'feature_type': 'text'
    },
    'densenet121': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
}

'''
class Conv1DClassifier(nn.Module):
  def __init__(self,inputSize=300,outputSize=1,f=7):
    super(Conv1DClassifier, self).__init__()
    self.inputSize=inputSize
    self.f=f
    self.conv1 = nn.Conv1d(inputSize, 16, f, 1) #filter size: 7x7
    self.fc1 = nn.Linear(16, outputSize)

  def forward(self, x):
    x = x.unsqueeze(dim=1)
    x = F.relu(self.conv1(x))

    x = x.squeeze()
    x = self.fc1(x)
    return torch.squeeze(x)
'''

class Conv1DClassifier(nn.Module):
    def __init__(self, inputSize = 300, outputSize = 1, f = 7):
        super(Conv1DClassifier, self).__init__()

        self.inputSize = inputSize
        self.conv1 = nn.Conv1d(1, 10, kernel_size=f, padding=(f-1)//2) # 16 x 1 x 300 -> 16 x 10 x 300
        self.conv2 = nn.Conv1d(10, 32, f-2, padding=(f-3)//2)  # 16 x 10 x 300 -> 16 x 32 x 300
        self.linear1 = nn.Linear(32*inputSize, outputSize)

    def forward(self, x):
        x = x.unsqueeze(dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(-1, 32 * self.inputSize)
        x = self.linear1(x)

        return torch.squeeze(x)




class LinearReg(nn.Module):
  def __init__(self, inputSize = 224*224*3, outputSize=1, bias=False):
        super(LinearReg, self).__init__()
        self.inputSize = inputSize
        self.linear = torch.nn.Linear(inputSize, outputSize, bias = bias)
        #self.apply(self._init_weights)
        
  def forward(self, x):
        x = x.view(-1, self.inputSize)
        out = self.linear(x)
        return torch.squeeze(out)

class FCN(nn.Module):
  def __init__(self, inputSize = 224*224*3, outputSize=1, bias=False):
        super(FCN, self).__init__()
        self.inputSize = inputSize
        self.linear1 = torch.nn.Linear(inputSize, 100, bias = bias)
        self.linear2 = torch.nn.Linear(100, 25, bias = bias)
        self.linear3 = torch.nn.Linear(25, outputSize, bias = bias)
        #self.apply(self._init_weights)

  def forward(self, x):
        x = x.view(-1, self.inputSize)
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        return torch.squeeze(out)

class ConvNet2(nn.Module):
  def __init__(self,inputSize=224,outputSize=1,f=7):
    super(ConvNet2, self).__init__()
    self.inputSize=inputSize
    self.f=f
    self.conv1 = nn.Conv2d(3, 1, f, 1) #filter size: 7x7
    self.fc1 = nn.Linear((inputSize-f+1)**2, outputSize)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = x.view(-1, (self.inputSize-self.f+1)**2)
    x = self.fc1(x)
    return torch.squeeze(x)


class ConvNet(nn.Module):
  def __init__(self,inputSize=224,outputSize=1,f=7):
    super(ConvNet, self).__init__()
    self.inputSize=inputSize
    self.f=f
    self.conv1 = nn.Conv2d(3, 10, f, 1)
    self.conv2 = nn.Conv2d(10, 20, f-3, 1)
    self.fc1 = nn.Linear(20 * ((int(inputSize/4)-(f-3)+1)**2), 2000)
    self.fc2 = nn.Linear(2000, outputSize)

  def forward(self, x):
    x = F.leaky_relu(self.conv1(x))
    x = F.avg_pool2d(x, 2, 2)
    x = F.leaky_relu(self.conv2(x))
    x = F.avg_pool2d(x, 2, 2)
    x = x.view(-1, 20 * ((int(self.inputSize/4)-(self.f-3)+1)**2))
    x = F.leaky_relu(self.fc1(x))
    logits = self.fc2(x)
    return logits


