from . import node
import time

__all__ = ['fednode']

class fednode(node.P2PNode):        
    def _handle(self, data, conn):
        """
        handle data from other nodes
        """
        if data[0] == 'weights':
            self.logger.info("recv weights : {}".format(data[1]))
        super()._handle(data, conn)
    
    def mainjob(self):
        """
        this will be learning process
        
        # test code(to check recv size)
        # messege = input("data to send :")
        # self.send_data(self.predecessor_addr, ('weights', messege))
        # time.sleep(2)
        """

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

def learning():
    """
    a simple example to test CNN weight serialization using cuda
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=8, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=8, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define a Convolution Neural Network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('< 1 >    [%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            # print accuracy
            if i % 1000 == 999:
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                print('< 1 >    Accuracy on the 10000 test images: %d %%' % (
                    100 * correct / total))
 
    print('Finished Training')

    # save model
    torch.save(net.state_dict(), './model.pth')

    # save weights
    weights = net.state_dict()
    return weights

def learning2():
    """
    a simple example to test weight serialization using CUDA
    dataset : cifar10
    model : resnet50
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                              shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=4, pin_memory=True)


    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # define a Convolution Neural Network
    model = torchvision.models.resnet50(pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    torch.backends.cudnn.benchmark = True
    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            #zero the parameter gradients
            #optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i == 194:
                print('< 2 >    [%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            # print accuracy
            if i == 194:
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                print('< 2 >    Accuracy on the 10000 test images: %d %%' % (
                    100 * correct / total))

if __name__ == '__main__':
    #learning()
    start = time.time()
    learning2()
    end =time.time()
    print("time : ", end-start)