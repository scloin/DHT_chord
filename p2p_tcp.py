"""
- run (must run root node first)
    root node  : python p2p_tcp.py 8000(this node's port)
    other node : python p2p_tcp.py 7000(this node's port) 8000(root node port)
"""
import logging
import argparse
import time
from time import strftime
from src.node import P2PNode
from src.utils import MultiLineFormatter

def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="this peer's port number", type=int)
    parser.add_argument("--addr", help="this peer's ip address", type=str, default='localhost')
    parser.add_argument("--help_port", help="help peer's port number", type=int, default=-1)
    parser.add_argument("--help_addr", help="this peer's ip address", type=str, default='localhost')
    parser.add_argument("--log", help="this peer's ip address", action="store_true", default=False)
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
        log_handler= logging.FileHandler("logs/%s.log" %(str(this_addr[1])), mode='a', encoding=None, delay=False)
        log_handler.setFormatter(formatter)
        log_handler.setLevel(logging.DEBUG)
        logger.addHandler(log_handler)
        
        con_handler= logging.StreamHandler()
        con_handler.setFormatter(formatter)
        con_handler.setLevel(logging.INFO)
        logger.addHandler(con_handler)
        
        logger.setLevel(logging.DEBUG)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        log_handler= logging.StreamHandler()
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)
    
    return this_addr, help_addr, logger

if __name__ == '__main__':
    this_addr, help_addr, logger = handle_args()
    node = P2PNode(logger, (this_addr), (help_addr))
