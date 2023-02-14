home=/home/deepl/sooho/chord/DHT_chord
python=/home/deepl/anaconda3/envs/pytorch/bin/python


#default value
addr='localhost'

#python3.7 $home/p2p_tcp.py --port $port --addr $addr --log

# str=$($python $home/test_chord.py 15)
# echo $str

str="12002 12008 12011 12013 12015 12024 12029 12032 12042 12057 12067 12070 12089 12174"
echo 12000 ready.. > logs/12000.log
for var in $str
do
    echo $var ready.. > logs/$var.log
done

$python $home/p2p_tcp.py --port 12000 --log &
echo "start 12000"
sleep 2
for var in $str
do
   $python $home/p2p_tcp.py -p $var -P 12000 --log >/dev/null&
    echo "start $var"
    #sleep 2 
done

# kill process after 10 sec

# sleep 10
# p0=$(netstat -tupln | grep -Ei 12000 | grep -oEi [0-9]+/ | grep -oEi [0-9]+)
# kill -9 $p0