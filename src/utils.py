import textwrap
import logging
import hashlib
import subprocess
import socket

__all__ = ['contain', 'MultiLineFormatter', 'hash', ]

def contain(id, begin, end):
    """
    check if id is between begin and end
    """
    if begin < end:
        return begin < id <= end
    elif begin > end:
        return begin < id or id <= end
    return False

class MultiLineFormatter(logging.Formatter):
    def format(self, record):
        message = record.msg
        record.msg = ''
        header = super().format(record)
        msg = textwrap.indent(message, ' ' * len(header)).lstrip()
        record.msg = message
        return header + msg

def hash(addr, NUM_OF_BITS=6):
    """
    hash function, that generated id of node by address, using sha1
    """
    return int(hashlib.sha1(str(addr).encode()).hexdigest(), 16) % (2**NUM_OF_BITS)


def get_global_ip():
    """
    get global ip address
    """
    return subprocess.check_output("wget http://ipecho.net/plain -O - -q ; echo", shell=True).decode().strip()

def get_self_ip():
    return socket.gethostbyname(socket.gethostname())

if __name__ == '__main__':
    ip : str
    #ip = input("ip")
    ip = 'localhost'
    port : int
    # port = input('port')
    slist = [0 for n in range(64)]
    for port in range(12000, 20000, 1):
        hashed = hash((ip, port))
        if slist[hashed]==0:
            slist[hashed] = port
    #slist.sort()
    print(slist)