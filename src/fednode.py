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

import socket, threading, selectors,time,pickle
import numpy as np
#import msgpack
import gzip

__all__ = ['fednode']

class fednode(node.P2PNode):
    def __init__(self, logger, addr, host_addr=None, container=False, device=None):
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
        
        # data=b''
        # conn.settimeout(10)
        # while True:
        #     datas = conn.recv(size)
        #     if not datas: break
        #     data += datas
            
        data=[]
        while size>0:
            s = conn.recv(size)
            #print(len(s))
            if not s: break
            data.append(s)
            size -= len(s)
        data = b''.join(data)
        
        conn.close()
        data = gzip.decompress(data)
        #print(len(data))
        data = pickle.loads(data)
        check=False
        for i,par in enumerate(self.learning.parQueue):
            if(par[0]==id):
                self.learning.parQueue[i] = (id, rounds, data)
                check=True
        if check!=True:
            self.learning.parQueue.append((id, rounds, data))

    
    def _handle_pull_param(self ,conn):
        """
        pull param from other node
        """
        seri_t = time.time()
        #par = self.learning.net.state_dict()
        print(self.id)
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
        print('serialize the model',end=' ')
        print('time:{:.2f}'.format(time.time()-seri_t))
        send_t = time.time()

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
            print('==> Preparing data..')
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

            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train)
            self.trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=128, shuffle=True, num_workers=2)

            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)
            self.testloader = torch.utils.data.DataLoader(
                testset, batch_size=100, shuffle=False, num_workers=2)

            self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')

            # Model
            print('==> Building model..')
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

                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


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

                    progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                    if (100.*correct/total>50 and total==10000):
                        print(time.time()-self.st)
                        exit()

        def run(self, fednode):
            self.addr = fednode.addr
            
            self.id = fednode.id
            self.finger_table = fednode.finger_table
            
            
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
                    print([par[0] for par in self.parQueue])
                    print([par[1] for par in self.parQueue])
                    self.net.load_state_dict(self.merge_param(self.parQueue, self.device))
                    
                    self.net = self.net.to(torch.device(self.device))
                    print("parameters loaded time:{:.4f}".format(time.time()-up_t))
                    self.parQueue = [(id, 0,None),]
                    
                
                for epoch in range(start_epoch, start_epoch+5):
                    train(epoch)
                    test(epoch)
                    scheduler.step()

                start_epoch+=5
                par = self.net.state_dict()
                self.parQueue[0]=(self.id, rounds, par)
                self.push_param(rounds)
                print("end of round {}".format(rounds))

        
        def push_param(self, rounds: int=0):
                seri_t = time.time()
                # par = self.net.state_dict()
                # self.parQueue[0]=(self.id, rounds, par)
                #print(par)
                
                par = pickle.dumps(self.parQueue[0][2])
                #print size of par for set the buffer size
                #print('size of par:{}B'.format(len(par)))
                par = gzip.compress(par)
                #print('size of par:{}B'.format(len(par)))
                
                #print(par)
                #par = msgpack.packb(par.numpy())
                print('serialize the model',end=' ')
                print('time:{:.2f}'.format(time.time()-seri_t))
                send_t = time.time()
                #sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                #sock.connect(('localhost',14141))
                for finger in set(self.finger_table):
                    print(finger)
                    if(finger[1]==self.id):
                        print('skip self')
                        continue
                    else:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.connect(finger[0])
                        sock.send(pickle.dumps(('push_param', len(par), rounds, self.id)))
                        #sock.send(pickle.dumps((len(par))))
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        if sock.recv(1024).decode('utf-8')=='ok':
                            sock.send(par)
                        sock.close()
                        print('send parameters time:{:.2f}'.format(time.time()-send_t))

        def pull_param(self):

            #finger 에 잌ㅅ는 애들..
            for finger in set(self.finger_table):
                    print(finger)
                    if(finger[1]==self.id):
                        print('skip self')
                        continue
                    else:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.connect(finger[0])
                        sock.send(pickle.dumps(('weight_req',)))
                        #sock.send(pickle.dumps((len(par))))
                        sock.settimeout(60)
                        para = sock.recv(1024)
                        para = pickle.loads(para)
                        print(para)
                        size = para[1]
                        rounds = para[2]
                        id = para[3]
                        sock.send(b'ok')
                        print('sent ok')
                        #print(size, rounds, id)
                        
                        # data=b''
                        # sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        # sock.settimeout(60)
                        # while True:
                        #     datas = sock.recv(size)
                        #     print(len(datas))
                        #     if not datas: break
                        #     data += datas
                        
                        data=[]
                        while size>0:
                            s = sock.recv(size)
                            #print(len(s))
                            if not s: break
                            data.append(s)
                            size -= len(s)
                        data = b''.join(data)
                        
                        
                        sock.close()
                        #print('received data')
                        data = gzip.decompress(data)
                        #print(len(data))
                        data = pickle.loads(data)
                        check=False
                        for i,par in enumerate(self.parQueue):
                            #print(par[0])
                            if(par[0]==id):
                                self.parQueue[i] = (id, rounds, data)
                                check=True
                        if check!=True:
                            #print('------------',id)
                            self.parQueue.append((id, rounds, data))
                
        
        def merge_param(self, P : list((int,int,OrderedDict([(str, torch.Tensor),])),), device):
            """
            merge and mean parameters from other nodes.
            Device of Tensor can be different, so we need to move the parameters to the device of the first parameter.
            input : list(tuple(int,int,OrderedDict([(str, torch.Tensor),])),)
            output : OrderedDict([(str, torch.Tensor),])
            """
            par = P[0][2]
            for i in range(1,len(P)):
                for key in par.keys():
                    par[key].add_(P[i][2][key].to(torch.device(device)))
            for key in par.keys():
                par[key] = par[key]/len(P)
            #par = par.to(torch.device(device))
            return par



'''
class learning2:
    def __init__(self):
        
        self.alive = True
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('localhost',13000))
        self.socket.setblocking(False)
        self.socket.listen(5)
        #self.par = (0, None)
        
        self.parQueue = [(self.id, 0,None),] #will be size same as number of workers
        
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
        parser.add_argument('--resume', '-r', action='store_true',
                            help='resume from checkpoint')
        args = parser.parse_args()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        best_acc = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        # Data
        print('==> Preparing data..')
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

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')

        # Model
        print('==> Building model..')
        self.net = ResNet18()
        self.net = self.net.to(self.device)
        if self.device == 'cuda:0':
            cudnn.benchmark = True

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        listen = threading.Thread(target=self.listen)
        listen.daemon = True
        listen.start()
        self.run()
        self.alive = False
        listen.join()

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

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


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

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def run(self):
        net, start_epoch, scheduler = self.net, self.start_epoch, self.scheduler
        train, test = self.train, self.test
        
        for rounds in range(20):
            if rounds!=0:
                """
                wait until self.par == rounds
                """
                # while True:
                #     if self.par[0]==rounds:
                #         break
                #     time.sleep(1)
                up_t = time.time()
                # self.parQueue.append(self.par[1])
                
                self.net.load_state_dict(merge_param(self.parQueue, self.device))
                
                self.net = self.net.to(torch.device(self.device))
                print("parameters loaded time:{:.2f}".format(time.time()-up_t))
                
            
            for epoch in range(start_epoch, start_epoch+1):
                train(epoch)
                test(epoch)
                scheduler.step()

            start_epoch+=1 
            
            self.push_param()

        
            
    def listen(self):
        """
        thread for listening
        """
        while self.alive:
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.socket, selectors.EVENT_READ, self.accept_handler)
            while self.alive:
                for (key,mask) in self.selector.select():
                    key: selectors.SelectorKey
                    srv_sock, callback = key.fileobj, key.data
                    callback(srv_sock, self.selector)
    
    def accept_handler(self, sock: socket.socket, sel: selectors.BaseSelector):
        """
        accept connection from other nodes
        """
        conn: socket.socket
        conn, addr = sock.accept()
        sel.register(conn, selectors.EVENT_READ, self.read_handler)


    def read_handler(self, conn: socket.socket, sel: selectors.BaseSelector):
        """
        read data from other nodes
        """
        message = "---- wait for recv[any other] from {}".format(conn.getpeername())
        #self.logger.debug(message)  
        data = b""
        recv_t = time.time()
        data_len = pickle.loads(conn.recv(1024))
        print(data_len)
        conn.send(b"ok")
        while True:
            datas = conn.recv(data_len)
            if not datas: break
            data += datas
        print("recv parameters time:{:.2f}".format(time.time()-recv_t))
        deseri_t = time.time()
        
        data = gzip.decompress(data)
        data = pickle.loads(data)
        #data = msgpack.unpackb(data)
        #data = torch.from_numpy(np.asarray(data))
        self.par = (self.par[0]+1, data)
        print("deserialize the model:{:.2f}".format(time.time()-deseri_t))
        #self._handle(data, conn)
        #data = pickle.loads(data)
        #self.logger.debug("[recv data : {}]".format(data))
        # threading.Thread(target=self._handle, args=((data,conn)), daemon=True).start()
        
        
        sel.unregister(conn)
    
    def push_param(self, rounds: int=0):
            seri_t = time.time()
            par = self.net.state_dict()
            self.parQueue[0]=(self.id, rounds, par)
            #print(par)
            par = pickle.dumps(par)
            #print size of par for set the buffer size
            #print('size of par:{}B'.format(len(par)))
            par = gzip.compress(par)
            #print('size of par:{}B'.format(len(par)))
            
            #print(par)
            #par = msgpack.packb(par.numpy())
            print('serialize the model',end=' ')
            print('time:{:.2f}'.format(time.time()-seri_t))
            send_t = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #sock.connect(('localhost',14141))
            for finger in set(self.finger_table):
                if(finger[0]==self.addr):
                    continue
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(finger[0])
                sock.send(pickle.dumps(('push_param', len(par), rounds, self.id)))
            #sock.send(pickle.dumps((len(par))))
            if sock.recv(1024).decode('utf-8')=='ok':
                sock.send(par)
            sock.close()
            print('send parameters time:{:.2f}'.format(time.time()-send_t))
    
    
def merge_param(P : tuple(int,int,list(OrderedDict([(str, torch.Tensor),]),)), device = 'cuda:0'):
    """
    merge and mean parameters from other nodes.
    
    input : list of OrderedDict([(str, torch.Tensor),])
    output : OrderedDict([(str, torch.Tensor),])
    """
    par = P[0][2]
    for i in range(1,len(P)):
        for key in par:
            par[key] += P[i][2][key]
    for key in par:
        par[key] = par[key]/len(P)
    par = par.to(device)
    return par
'''

if __name__ == '__main__':
    fingertable = [("localhost",12000)]
    fednode.learning2(14000,[])

    # fingertable = [("localhost",14000)]
    # fednode.learning2(12000,[])