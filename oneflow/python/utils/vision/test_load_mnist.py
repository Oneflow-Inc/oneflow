import oneflow as flow

from IPython import display
import matplotlib.pyplot as plt
import transforms as transforms
import oneflow.python.utils.data as data
from datasets.mnist import FashionMNIST
import time
import sys


mnist_train = FashionMNIST(root='./test/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = FashionMNIST(root='./test/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())


print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def use_svg_display():
    """Use svg format to display plot in jupyter."""
    display.set_matplotlib_formats('svg')

# 本函数已保存在d2lzh包中方便以后使用
def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        img = img.reshape(shape=[img.size(1), img.size(2)])
        f.imshow(img.numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.savefig("fashion-mnist.png")
    # plt.show()

X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))


batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter =  data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))