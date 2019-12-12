# Usage: 
## Test all cnn nets: 
```
python run_cnns_test.py
```

## Test all cnn nets on current node: 
```
python 1node_run_cnns_test.py
```

## Test a specific net:
### alexnet
```
python run_cnns_test.py TestAlexNet
```

### resnet50
```
python run_cnns_test.py TestResNet50
```

### inceptionv3
```
python run_cnns_test.py TestInceptionV3
```

### vgg16
```
python run_cnns_test.py TestVgg16
```

## Test a specific case for a specific net: 

### test alexnet on 1 gpu, 1 machine(node)
```
python run_cnns_test.py TestAlexNet.test_1n1c
```

### test alexnet on 4 gpu, 1 machine(node)
```
python run_cnns_test.py TestAlexNet.test_1n4c
```

### test alexnet on 4 gpu, 8 machine(node)
```
python run_cnns_test.py TestAlexNet.test_2n8c

```

## Loss report format
```
======================================================================
xx net loss report
======================================================================
iter     tensorflow   oneflow-1n1c
0        6.932688     6.932688
1        6.924820     6.924820
2        6.917069     6.917069
3        6.909393     6.909393
4        6.901904     6.901904
5        6.894367     6.894367
6        6.886764     6.886764
7        6.879305     6.879305
8        6.872003     6.872003
9        6.864939     6.864939
```

## Test Bert on current node (1n1c, 1n4c):

```
python 1node_run_cnns_test.py test_bert
```

## Test Bert with specified distributed strategy
```
python 2node_run_cnns_test.py test_bert.test_2n8c
```
