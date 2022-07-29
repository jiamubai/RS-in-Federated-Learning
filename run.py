from learntosketch import *
import torch.optim as optim
import torch.nn.functional as F

# import torch
# import torch.nn as nn
# from datasets import MNIST_truncated
# import torchvision.transforms as transforms
# from sklearn import preprocessing
#
#
# transform = transforms.Compose([transforms.ToTensor()])
#
# mnist_train_ds = MNIST_truncated('./data', train=True, download=True, transform=transform)
# mnist_test_ds = MNIST_truncated('./data', train=False, download=True, transform=transform)
#
# X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
# X_test, y_test = mnist_test_ds.data, mnist_test_ds.target
#
# print(X_train.shape)
# X_train = X_train.reshape(60000, -1)
#
#
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = torch.tensor(scaler.transform(X_train))
#
# X_train = X_train.data.numpy()
# y_train = y_train.data.numpy()
# X_test = X_test.data.numpy()
# y_test = y_test.data.numpy()
# print(X_train.shape)

import torch
import torchvision

n_epochs = 10
batch_size_train = 256
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('~/torch_datasets', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('~/torch_datasets', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  # net.to('cuda:3')
  net.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    data = torch.reshape(data, (len(target), -1))
    output = net(data)
    # print('output:', torch.argmax(output[0]))
    # print('target:', target[0])
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      # torch.save(net.state_dict(), '~/results/model.pth')
      # torch.save(optimizer.state_dict(), '~/results/optimizer.pth')

def test():
  net.eval()
  test_loss = 0
  train_loss = 0
  correct = 0
  with torch.no_grad():
    # for data, target in test_loader:
    for data, target in train_loader:
      data = torch.reshape(data, (len(target), -1))
      output = net(data)
      # test_loss += F.nll_loss(output, target, size_average=False).item()
      train_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  # test_loss /= len(test_loader.dataset)
  # test_losses.append(test_loss)
  train_loss /= len(train_loader.dataset)
  # train_losses.append(train_losses)
  print('\nTrain set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    train_loss, correct, len(train_loader.dataset),
    100. * correct / len(train_loader.dataset)))

net = SketchNetwork(K = 14, R = 6000, d = 28*28, OUT = 10, aggregation = 'avg', dropout_rate = 0.0, hash_func = 'P-stable')
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()
