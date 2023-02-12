home=/home/deepl/sooho/chord/DHT_chord
python=/home/deepl/anaconda3/envs/pytorch/bin/python


#default value
addr='localhost'

list="13221 12693 12594 13254 15135 13782 14607 12330 13584 14145 12726 19656 13089"
list2="13221 12594 15135 14607 13584 12726 13089 13782"
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
    if [ $var -eq 12011 ]
    then
        $python $home/p2p_tcp.py -p $var -P 12000 --log --debug >/dev/null&
    else
        $python $home/p2p_tcp.py -p $var -P 12000 --log >/dev/null&
    fi
    echo "start $var"
    #sleep 2
done

# kill process after 10 sec

# sleep 10
# p0=$(netstat -tupln | grep -Ei 12000 | grep -oEi [0-9]+/ | grep -oEi [0-9]+)
# kill -9 $p0