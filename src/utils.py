import textwrap
import logging

def contain(id, id1, id2):
    """
    check if id is between id1 and id2
    """
    if id1 < id2:
        return id1 < id <= id2
    else:
        return id1 < id or id <= id2

class MultiLineFormatter(logging.Formatter):
    def format(self, record):
        message = record.msg
        record.msg = ''
        header = super().format(record)
        msg = textwrap.indent(message, ' ' * len(header)).lstrip()
        record.msg = message
        return header + msg
