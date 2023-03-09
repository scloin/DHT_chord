try:
    from . import node
except:
    import node

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy

'''Train CIFAR10 with PyTorch.'''
from collections import OrderedDict
import zlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

import os
import argparse

try:
    from models import *
except:
    from .models import *

try:
    from utils import progress_bar
except:
    from .utils import progress_bar

from ._merge_param import merge_param

from ._custom_dataloader import customDataLoader, custom_dataset

import socket, threading, selectors,time,pickle
import numpy as np
#import msgpack
import gzip

__all__ = ['fednode']

class fednode(node.P2PNode):
    def __init__(self, logger, addr, host_addr=None, container=False, device=None, data_config:list=[1,0]):
        self.st = time.time()
        self.device = device
        self.learning= self.learning2(self)
        super().__init__(logger, addr, host_addr, container)
            
    def _handle(self, data, conn):
        """
        handle data from other nodes
        """
        if data[0] == 'push_param':
            "the case of push param"
            self._handle_push_param(data,conn)
            pass
        elif data[0] == 'weight_req':
            "the case of pull param"
            self._handle_pull_param(conn)
            pass
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
        self.learning.run(self)
        
    def learn(self):
        """
        learning process
        """
    
    def _handle_push_param(self, para, conn: socket.socket):
        """
        push param to other node
        """
        size = para[1]
        rounds = para[2]
        id = para[3]
        conn.send(b'ok')
        #print(size, rounds, id)
        recv_t = time.time()            
        data=[]
        while size>0:
            s = conn.recv(size)
            #print(len(s))
            if not s: break
            data.append(s)
            size -= len(s)
        data = b''.join(data)
        
        conn.close()
        print('recv parameters time:{:.4f}'.format(time.time()-recv_t))
        unzip_t = time.time()
        #data = gzip.decompress(data)
        #print(len(data))
        data = pickle.loads(data)
        print('unzip time:{:.4f}'.format(time.time()-unzip_t))
        check=False
        for i,par in enumerate(self.learning.parQueue):
            if(par[0]==id):
                self.learning.parQueue[i] = (id, rounds, data)
                check=True
        if check!=True:
            self.learning.parQueue.append((id, rounds, data))

    
    def _handle_pull_param(self ,conn: socket.socket):
        """
        pull param from other node
        """
        print("recv_weight_req from",conn.getsockname())
        seri_t = time.time()
        #par = self.learning.net.state_dict()
        #print(self.id)
        #self.learning.parQueue[0]=(self.id, self.learning.rounds, par)
        #print(par)
        while(self.learning.parQueue[0][2]==None):
            time.sleep(0.1)
        par = pickle.dumps(self.learning.parQueue[0][2])
        #print size of par for set the buffer size
        #print('size of par:{}B'.format(len(par)))
        par = gzip.compress(par)
        #print('size of par:{}B'.format(len(par)))
        
        #print(par)
        #par = msgpack.packb(par.numpy())
        #print('serialize the model',end=' ')
        #print('time:{:.2f}'.format(time.time()-seri_t))
        send_t = time.time()
        print("send_meta_data")
        conn.send(pickle.dumps(('push_param', len(par), self.learning.rounds, self.id)))

        #sock.send(pickle.dumps((len(par))))
        #conn.settimeout(10)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if conn.recv(1024).decode('utf-8')=='ok':
            print('recv ok')
            conn.send(par)
        conn.close()
        print('send parameters time:{:.2f}'.format(time.time()-send_t))
    
    class learning2:
        
        def __init__(self, fednode):
            self.st = time.time()
            self.prev_t=time.time()
            self.alive = True
            self.rounds =0
            self.parQueue = [(id, 0,None),] #will be size same as number of workers
            
            # parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
            # parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
            # parser.add_argument('--resume', '-r', action='store_true',
            #                     help='resume from checkpoint')
            # args = parser.parse_args()
            self.device = fednode.device if torch.cuda.is_available() else 'cpu'
            #self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
            
            best_acc = 0  # best test accuracy
            self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch

            # Data
            #print('==> Preparing data..')
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            trainset = custom_dataset(
                root='./data', train=True, download=True, transform=transform_train, 
                split_number=4, split_id=0)
            self.trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=128, shuffle=True, num_workers=2)

            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)
            self.testloader = data.DataLoader(
                testset, batch_size=100, shuffle=False, num_workers=2)

            self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')

            # Model
            #print('==> Building model..')
            self.net = ResNet18()
            self.net = self.net.to(self.device)
            if self.device!='cpu':
                #self.net = torch.nn.DataParallel(self.net)
                cudnn.benchmark = True
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(self.net.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        # Training
        def train(self, epoch):
            net, trainloader, device, optimizer, criterion = self.net, self.trainloader, self.device, self.optimizer, self.criterion
            
            print('\nEpoch: %d' % epoch)
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
                if (total==10000):
                    print("train loss : %0.4f  train acc : %0.2f" %(train_loss/(batch_idx+1),100.*correct/total))


        def test(self, epoch):
            net, testloader, device, criterion = self.net, self.testloader, self.device, self.criterion
            global best_acc
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                    if (total==10000):
                        print("test  loss : %0.4f  test acc : %0.2f" %(test_loss/(batch_idx+1),100.*correct/total))
                    if (100.*correct/total>90 and total==10000):
                        print(time.time()-self.st)
                        #exit()

        def run(self, fednode):
            self.addr = fednode.addr
            
            self.id = fednode.id
            self.finger_table = fednode.finger_table
            
            print(self.id, self.addr)
            net, start_epoch, scheduler = self.net, self.start_epoch, self.scheduler
            train, test = self.train, self.test
            
            for rounds in range(30):
                self.rounds = rounds
                if rounds!=0:
                    """
                    wait until self.par == rounds
                    """
                    while True:
                        if len(self.parQueue)>1:
                            break
                        time.sleep(1)
                    up_t = time.time()
                    # self.parQueue.append(self.par[1])
                    # print([par[0] for par in self.parQueue])
                    # print([par[1] for par in self.parQueue])
                    self.net.load_state_dict(self.merge_param(self.parQueue, self.device))
                    
                    self.net = self.net.to(torch.device(self.device))
                    print("parameters loaded time:{:.4f}".format(time.time()-up_t))
                    self.parQueue = [(id, 0,None),]
                    
                
                for epoch in range(start_epoch, start_epoch+10):
                    train(epoch)
                    test(epoch)
                    scheduler.step()
                    print("epoch time : {:.4f} ({:.4f})".format(time.time()-self.st, time.time()-self.prev_t))
                    self.prev_t = time.time()
                    

                start_epoch+=10
                par = self.net.state_dict()
                self.parQueue[0]=(self.id, rounds, par)
                #self.pull_param()
                self.push_param(rounds)
                #print("end of round {}".format(rounds))

        
        def push_param(self, rounds: int=0):
                seri_t = time.time()
                # par = self.net.state_dict()
                # self.parQueue[0]=(self.id, rounds, par)
                #print(par)
                
                par = pickle.dumps(self.parQueue[0][2])
                #par = gzip.compress(par)
                
                #print(par)
                #par = msgpack.packb(par.numpy())
                print('serialize the model',end=' ')
                print('time:{:.4f}'.format(time.time()-seri_t))
                send_t = time.time()
                for finger in set(self.finger_table):
                    #print(finger)
                    if(finger[1]==self.id):
                        #print('skip self')
                        continue
                    else:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.connect(finger[0])
                        sock.send(pickle.dumps(('push_param', len(par), rounds, self.id)))

                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        if sock.recv(1024).decode('utf-8')=='ok':
                            sock.send(par)
                        sock.close()
                        print('send parameters time:{:.4f}'.format(time.time()-send_t))

        def pull_param(self):

            #finger 에 잌ㅅ는 애들..
            for finger in set(self.finger_table):
                    #print(finger)
                    if(finger[1]==self.id):
                        #print('skip self')
                        continue
                    else:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.connect(finger[0])
                        sock.send(pickle.dumps(('weight_req',)))
                        #sock.send(pickle.dumps((len(par))))
                        print(finger,"send_weight_req")
                        #sock.settimeout(60)
                        sock.setblocking(True)
                        para = sock.recv(1024)
                        para = pickle.loads(para)
                        #print(para)
                        size = para[1]
                        rounds = para[2]
                        id = para[3]
                        sock.send(b'ok')
                        print('sent ok')
                        print(size, rounds, id)
                        
                        # data=b''
                        # sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        # sock.settimeout(60)
                        # while True:
                        #     datas = sock.recv(size)
                        #     print(len(datas))
                        #     if not datas: break
                        #     data += datas
                        
                        data=[]
                        #sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        #sock.setblocking(False)
                        while size>0:
                            s = sock.recv(size)
                            #print(len(s))
                            if not s: break
                            data.append(s)
                            size -= len(s)
                        data = b''.join(data)
                        print(len(data))
                        
                        sock.close()
                        print('received data')
                        data = gzip.decompress(data)
                        print(len(data))
                        data = pickle.loads(data)
                        check=False
                        for i,par in enumerate(self.parQueue):
                            #print(par[0])
                            if(par[0]==id):
                                self.parQueue[i] = (id, rounds, data)
                                check=True
                        if check!=True:
                            print('------------',id)
                            self.parQueue.append((id, rounds, data))
                
        
        def merge_param(self, P : list((int,int,OrderedDict([(str, torch.Tensor),])),), device):
            # """
            # merge and mean parameters from other nodes.
            # Device of Tensor can be different, so we need to move the parameters to the device of the first parameter.
            # input : list(tuple(int,int,OrderedDict([(str, torch.Tensor),])),)
            # output : OrderedDict([(str, torch.Tensor),])
            # """
            # par = P[0][2]
            # for i in range(1,len(P)):
            #     for key in par.keys():
            #         par[key].add_(P[i][2][key].to(torch.device(device)))
            # for key in par.keys():
            #     par[key] = par[key]/len(P)
            # #par = par.to(torch.device(device))
            par = merge_param(P, device)
            return par


if __name__ == '__main__':
    fingertable = [("localhost",12000)]
    fednode.learning2(14000,[])

    # fingertable = [("localhost",14000)]
    # fednode.learning2(12000,[])