import oneflow as flow
# import matplotlib.pyplot as plt
import transforms as transforms
from datasets.mnist import FashionMNIST
import time
import sys


mnist_train = FashionMNIST(root='./test/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = FashionMNIST(root='./test/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())


print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width
