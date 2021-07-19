## Inference

测试平台：Nvidia GTX2080Ti单卡.  
CUDA版本：10.0  
CUDNN版本：7.5.0   
TensorRT版本：6.0.1  

Oneflow-Benchmark   
branch: of_dev_python_py3    
commit: 985dd3f03887d266e66573db0b31a4cf3051ff31   

Oneflow:   
branch: of_xrt_tensorrt   
commit: 726c3a12b9d97b57f9fb7e3d212b63564e20e755   

### CV

#### Speed

输入图片大小为224 (inception-v3为299)，预热5 batches，平均吞吐（img/s）为500个batches的平均值。

1. batch size为8

>| -            | Oneflow(fp32) | Oneflow(fp16) | TensorRT(fp32) | TensorRT(fp16) | TensorRT(int8) |
>| ------------ | ------------- | ------------- | -------------- | -------------- | -------------- |
>| alexnet      | 2637          | 1550          | 2540           | 2759           |                |
>| vgg16        | 371           | 332           | 377            | 1124           |                |
>| resnet50     | 657           | 541           | 729            | 940            |                |
>| inception-v3 | 433           | 434           | 489            | 999            |                |

2. batch size为50

>| -            | Oneflow(fp32) | Oneflow(fp16) | TensorRT(fp32) | TensorRT(fp16) | TensorRT(int8) |
>| ------------ | ------------- | ------------- | -------------- | -------------- | -------------- |
>| alexnet      | 6999          | 3219          | 4306           | 7704           |                |
>| vgg16        | 497           | 476           | 404            | 1482           |                |
>| resnet50     | 810           | 619           | 830            | 1285           |                |
>| inception-v3 | 544           | 531           | 717            | 1839           |                |


#### Precision

总共5w张图片, 统计Top1 accuracy和相对oneflow fp32的分类误差数量。

>|  -           | Oneflow(fp32) | Oneflow(fp16) | TensorRT(fp32) | TensorRT(fp16) | TensorRT(int8) |
>| ------------ | ------------- | ------------- | -------------- | -------------- | -------------- |
>| vgg16        | 0.495 / 0     | 0.495 / 61    | 0.495 / 0      | 0.495 / 101    |                |
>| alexnet      |               |               |                |                |                |
>| resnet50     | 0.613 / 0     | 0.613 / 59    | 0.613 / 0      | 0.613 / 130    |                |
>| inception-v3 |               |               |                |                |                |

