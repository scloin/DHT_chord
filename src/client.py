import socket, pickle
import hashlib

__all__ = ['client']
NUM_OF_BITS = 6

class client:
    def __init__(self, addr=None):
    #for i in range(1,65):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(addr)
        #self.addr = '220.67.133.165'
        self.id = 1234
        self.sock.send(pickle.dumps(('hop_count', 0,self.id)))
        #self.sock.send(pickle.dumps(('find_id', i)))
        #self.data = pickle.loads(self.sock.recv(1024))
        #print(self.data)
        self.sock.close()

if __name__ == '__main__':
    client(('220.67.133.165', 12000))