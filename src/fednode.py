try:
    from . import node
except:
    import node

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
        # messege = input("data to send :")
        # self.send_data(self.predecessor_addr, ('weights', messege))
        # time.sleep(2)
        
    def learn(self):
        """
        learning process
        """
        

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy

class learning:
    """
    a class for learning
    model : resnet50
    data : cifar10
    output : epoch, acc, loss, val_acc, val_loss, time
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.transform1 = transforms.Compose([
                    transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
        
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=self.transform1)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=512,
                                          shuffle=True, num_workers=8)
        
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=512,
                                         shuffle=False, num_workers=8)
        
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.net = torchvision.models.resnet50(pretrained=False)
        self.net.fc = nn.Linear(2048, 10)
        self.net = self.net.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        
        self.epoch = 0
        self.acc = 0
        self.loss = 0
        self.val_acc = 0
        self.val_loss = 0
        self.time = 0
        
        self.learning()
        
    def learning(self):
        """
        learning process
        """
        torch.backends.cudnn.benchmark = True
        for epoch in range(500):  # loop over the dataset multiple times
            start = time.time()
            self.epoch = epoch
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
        
                # zero the parameter gradients
                #self.optimizer.zero_grad()
                for param in self.net.parameters():
                    param.grad = None
        
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        
                # print acc, loss, val_acc, val_loss and train time for every 2000 mini-batches
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            ctime = time.time() - start
            self.time += ctime
            
            if epoch % 5 == 4:
                self.acc = 100.*correct/total
                self.loss = running_loss/2000
                self.val_acc, self.val_loss = self.val()

                print('[%2d] loss: %.3f | acc: %.2f %% | val_loss: %.3f | val_acc: %.2f %% | time: %.3fs [ %.3fs]' %
                        (epoch + 1, self.loss, self.acc, self.val_loss, self.val_acc, ctime, self.time))
                running_loss = 0.0
                correct = 0
                total = 0
                if(self.val_acc > 90):
                    break
        print('Finished Training')
    
    def val(self):
        """
        validation process
        """
        correct = 0
        total = 0
        loss = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return 100.*correct/total, loss/len(self.testloader)
         
if __name__ == '__main__':
    #learning()
    start = time.time()
    learning()
    end =time.time()
    print("time : ", end-start)