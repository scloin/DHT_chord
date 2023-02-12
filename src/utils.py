import textwrap
import logging
import hashlib

__all__ = ['contain', 'MultiLineFormatter', 'hash']

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