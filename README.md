# **P2P chord(DHT) protocal in python**

## **Usage**

### generate chord ring
basic example:
```bash
#root(start) node
python p2p_tcp --port <port of peer>

#peer node
python p2p_tcp --port <port of peer> --help_port <port of root>
```
- For other option, 
"python p2p_tcp -h" may help.
- you may chage log option in p2p_tcp.py