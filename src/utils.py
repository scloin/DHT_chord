import textwrap
import logging
import hashlib
import subprocess
import socket
import logging
import argparse
import time
from time import strftime
import sys
import time
import os

__all__ = ['contain', 'MultiLineFormatter', 'hash', 'progress_bar','get_global_ip', 'get_self_ip', 'handle_args', 'handle_args_f']

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

def handle_args():
    this_ip=get_self_ip()
    parser = argparse.ArgumentParser()
    parser.add_argument("--port","-p", help="this peer's port number", type=int)
    parser.add_argument("--addr","-a", help="this peer's ip address", type=str, default=this_ip)
    parser.add_argument("--help_port","-P", help="help peer's port number", type=int, default=-1)
    parser.add_argument("--help_addr","-A", help="help peer's ip address", type=str, default=this_ip)
    parser.add_argument("--log", help="enable log", action="store_true", default=False)
    parser.add_argument("--debug", help="enable log(debug)", action="store_true", default=False)
    parser.add_argument("--container","-c", help="container", action="store_true", default=False)
    args = parser.parse_args()

    this_addr = (args.addr, args.port)
    if args.help_port == -1:
        help_addr = None
    else:
        help_addr = (args.help_addr, args.help_port)
        
    formatter = MultiLineFormatter(
    fmt='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S',
    )

    if args.log == True:
        logger = logging.getLogger()
        log_handler= logging.FileHandler("logs/%s.log" %(str(this_addr[1])), mode='w', encoding=None, delay=False)
        log_handler.setFormatter(formatter)
        if args.debug==True:
            log_handler.setLevel(logging.DEBUG)
        else:
            log_handler.setLevel(logging.INFO)
        logger.addHandler(log_handler)
        
        con_handler= logging.StreamHandler()
        con_handler.setFormatter(formatter)
        con_handler.setLevel(logging.CRITICAL)
        logger.addHandler(con_handler)
        
        logger.setLevel(logging.DEBUG)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        log_handler= logging.StreamHandler()
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)
    
    
    
    return this_addr, help_addr, logger, args.container

def handle_args_f():
    this_ip=get_self_ip()
    parser = argparse.ArgumentParser()
    parser.add_argument("--port","-p", help="this peer's port number", type=int)
    parser.add_argument("--addr","-a", help="this peer's ip address", type=str, default=this_ip)
    parser.add_argument("--help_port","-P", help="help peer's port number", type=int, default=-1)
    parser.add_argument("--help_addr","-A", help="help peer's ip address", type=str, default=this_ip)
    parser.add_argument("--log", help="enable log", action="store_true", default=False)
    parser.add_argument("--debug", help="enable log(debug)", action="store_true", default=False)
    parser.add_argument("--container","-c", help="container", action="store_true", default=False)
    parser.add_argument("--test", "-t", help="A number for test case", type=int, default=0)
    #add for federated learning
    parser.add_argument("--gpu","-g", help="GPU num (-1 if use cpu)", type=int, default=0)
    
    args = parser.parse_args()

    this_addr = (args.addr, args.port)
    if args.help_port == -1:
        help_addr = None
    else:
        help_addr = (args.help_addr, args.help_port)
        
    formatter = MultiLineFormatter(
    fmt='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S',
    )

    if args.log == True:
        logger = logging.getLogger()
        log_handler= logging.FileHandler("logs/%s.log" %(str(this_addr[1])), mode='w', encoding=None, delay=False)
        log_handler.setFormatter(formatter)
        if args.debug==True:
            log_handler.setLevel(logging.DEBUG)
        else:
            log_handler.setLevel(logging.INFO)
        logger.addHandler(log_handler)
        
        con_handler= logging.StreamHandler()
        con_handler.setFormatter(formatter)
        con_handler.setLevel(logging.CRITICAL)
        logger.addHandler(con_handler)
        
        logger.setLevel(logging.DEBUG)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        log_handler= logging.StreamHandler()
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)
    
    #add for federated learning
    if args.gpu == -1:
        device = 'cpu'
    else:
        device = 'cuda:' + str(args.gpu)
    
    return this_addr, help_addr, logger, args.container, device, args.test

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

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 10.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


if __name__ == '__main__':
    print('g:', get_global_ip())
    print('s:', get_self_ip())
    ip : str
    #ip = input("ip")
    ip = get_self_ip()
    port : int
    # port = input('port')
    slist = [0 for n in range(64)]
    for port in range(12000, 20000, 1):
        hashed = hash((ip, port))
        if slist[hashed]==0:
            slist[hashed] = port
    #slist.sort()
    print(slist)