"""
This is implementation of DHT-chord protocol.

- run (must run root node first)
    root node  : python p2p_tcp.py 8000(this node's port)
    other node : python p2p_tcp.py 7000(this node's port) 8000(root node port)
"""
from src.node import P2Pnode
from src.utils import handle_args



if __name__ == '__main__':
    this_addr, help_addr, logger, container = handle_args()
    node = P2Pnode(logger, (this_addr), (help_addr), container)