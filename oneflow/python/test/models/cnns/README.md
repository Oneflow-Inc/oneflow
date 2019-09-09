# Usage: 
## Test all cnn nets: 
```
python run_cnns_test.py
```
or:
```
python -m unittest run_cnns_test 
```

## Test a specific net:
```
python -m unittest cnns_test.TestAlexNet
```

## Test a specific case for a specific net: 
```
python -m unittest cnns_test.TestAlexNet.test_1n1c
```

## Print test report
```
python -m unittest cnns_test.TestAlexNet.test_report
```
The report is like this:
```
======================================================================
xx net loss report
======================================================================
iter     tf          of-1n1c      of-1n4c       of-2n4c
0        6.932688    6.932688     6.932688      6.932688
1        6.924820    ...          ...           ...
2        6.917069
3        6.909393
4        6.901904
5        6.894367
6        6.886764
7        6.879305
8        6.872003
9        6.864939
```

## Other run options:
use `-f` to stop the test when fisrt error or fail occured.
```
python -m -f unittest run_cnns_test
```
