"""
This is implementation of Federated learning with P2P communication with DHT-chord protocol.

- run (must run root node first)
    root node  : python p2p_tcp.py 8000(this node's port)
    other node : python p2p_tcp.py 7000(this node's port) 8000(root node port)
"""
from src.fednode import fednode
from src.utils import handle_args_f

if __name__ == '__main__':
    this_addr, help_addr, logger, container, device = handle_args_f()
    #node = P2PNode(logger, (this_addr), (help_addr))
    node = fednode(logger, (this_addr), (help_addr), container, device)