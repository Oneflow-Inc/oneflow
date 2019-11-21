## XRT (X-Runtime)

XRT是一个同时支持多个计算引擎的运行时加速库，目前已经集成了TensorFlow XLA和Nvidia TensorRT两个后端引擎。其中XLA全面支持训练和预测，TensorRT支持预测以及部分算子支持训练。对于同一个计算图，XRT允许多个计算引擎联合使用，以获得更好的加速效果。

对于任意后端引擎，XRT的执行过程均分成以下四个步骤：

1. 计算图的转换
2. 引擎无关优化
3. 生成引擎相关Executable
4. 执行Executable

### 引擎无关优化

- 划分子图

  根据计算图中每个计算节点是否可编译、device、sbp policy等一系列属性，对节点进行聚合，被聚合的节点被新的节点（Launch节点）折叠后并在节点内进行子图重建，同时确定子图的后端执行引擎。

  如果多个后端引擎被开启，则会按照优先级进行每个引擎的子图划分。目前各引擎的优先级如下：

  - 训练时，优先进行XLA的子图划分，之后进行TensorRT子图划分。

  - 预测时，优先进行TensorRT的子图划分，之后进行XLA子图划分。

  [子图划分](https://github.com/Oneflow-Inc/oneflow-issue/issues/44)是自动完成的，但可以通过设置以下环境变量来调整子图划分的结果。

  ```shell
  export FLAGS_clustering_minimum_nodes=1
  export FLAGS_clustering_maximum_nodes=100
  export FLAGS_strict_clustering=true
  ```

  - FLAGS_clustering_minimum_nodes

    设置每个子图合并的节点的最小数量。当子图包含的节点数量小于该值时，则该合并的子图会被释放。

  - FLAGS_clustering_maximum_nodes

    设置每个子图合并的节点的最大数量。在合并时XRT可以保证每个子图包含的节点数不大于该设定值。

  - FLAGS_strict_clustering

    节点在合并时可能会互相破坏依赖，导致节点的执行时机发生改变。可以设置环境变量FLAGS_strict_clustering=true来规避该行为，确保合并后节点的执行时机不变。

    同时FLAGS_strict_clustering=true时会导致合并的子图变小，可能导致后端引擎丧失一些优化机会。FLAGS_strict_clustering默认设为true。

- ...

### Executable的生成

在runtime阶段，每个子图都可以被编译成一个与引擎相关的Executable。

对于静态shape的子图，由于缓存机制，每个子图只需要在运行时编译一次。对于包含动态shape的子图，则可能每次运行时都需要编译一次，因此如果计算图中包含动态shape的节点，暂时不建议使用XRT。

### Executable的执行

Executable执行时会分别调用所属的后端引擎提供的执行接口，执行完成后返回计算结果。对于GPU，执行接口调用是异步的，而对于CPU，执行接口调用是同步的。

- 临时内存管理

  目前XLA是通过自动增长的buffer内存池来管理临时内存的，并支持复用输出的buffer，达到减少显存占用和in-place计算的效果。

  TensorRT可以通过环境变量来设置临时buffer的最大字节数。

  ```shell
  export FLAGS_max_workspace_bytes=10000
  ```

- Max batch size

  TensorRT在执行时需要设置最大支持的batch size，XRT支持用户通过环境变量来设置，

  ```shell
  export FLAGS_max_batch_size=10
  ```

  当然，如果在运行时实际的batch size超过了设置的最大batch size，则XRT允许TensorRT Executable自动调整max batch size并正确执行（自动调整max batch size会带来一定的开销）。

### 在OneFlow中如何使用XRT

首先要求在编译OneFlow时开启了WITH_XLA或WITH_TENSORRT选项。

OneFlow中XRT的使用默认是关闭的，可以通过前端的Python接口和设置环境变量的方法来配置开启或关闭XLA和TensorRT，并且通过Python接口配置的优先级高于通过环境变量配置的方法。

- Python接口配置

  ```python
  import oneflow as flow

  # 配置使用XLA
  # True开启XLA，False关闭XLA，默认为未定义状态
  flow.config.use_xla_jit(True)

  # 配置使用TensorRT
  # True开启TensorRT，False关闭TensorRT，默认为未定义状态
  flow.config.use_tensorrt(True)
  ```

- 从环境变量配置

  ```shell
  # 只在Python前端未定义状态下生效
  export FLAGS_use_xla_jit=true # true为开启，false为关闭
  export FLAGS_use_tensorrt=true # true为开启，false为关闭
  ```

### BenchMark

- Bert base (batch size = 60)

  >| RTX  2080Ti 单卡       | FP32        |             | FP16混合精度 |             |
  >| ---------------------- | ----------- | ----------- | ------------ | ----------- |
  >|                        | oneflow     | oneflow-xla | oneflow      | oneflow-xla |
  >| loss (100 batches)     | 8.85063839  | 8.850635529 | 8.850672722  | 8.850834847 |
  >| s/batch                | 0.57        | 0.45        | 0.31         | 0.19        |
  >| 显存占用               | 8669MiB     | 8685MiB     | 7009MiB      | 7041MiB     |
  >| 计算吞吐 (sentences/s) | 105.2631579 | 133.3333333 | 193.5483871  | 315.7894737 |
  >| 加速比                 | 1           | 1.266666667 | 1            | 1.631578947 |
  
  >| RTX  2080Ti 2卡        | FP32        |             | FP16混合精度 |             |
  >| ---------------------- | ----------- | ----------- | ------------ | ----------- |
  >|                        | oneflow     | oneflow-xla | oneflow      | oneflow-xla |
  >| loss (100 batche)      | 8.806107521 | 8.806109428 | 8.806120873  | 8.806238174 |
  >| s/batch                | 0.596       | 0.485       | 0.353        | 0.241       |
  >| 显存占用               | 9147MiB     | 9149MiB     | 7669MiB      | 7675MiB     |
  >| 计算吞吐 (sentences/s) | 201.3422819 | 247.4226804 | 339.9433428  | 497.9253112 |
  >| 加速比                 | 1           | 1.228865979 | 1            | 1.46473029  | 
  
  >| RTX  2080Ti 4卡        | FP32        |             | FP16混合精度 |             |
  >| ---------------------- | ----------- | ----------- | ------------ | ----------- |
  >|                        | oneflow     | oneflow-xla | oneflow      | oneflow-xla |
  >| loss (100 batches)     | 8.730175972 | 8.730184555 | 8.730111122  | 8.729899406 |
  >| s/batch                | 0.61        | 0.495       | 0.376        | 0.252       |
  >| 显存占用               | 9147MiB     | 9149MiB     | 7669MiB      | 7675MiB     |
  >| 计算吞吐 (sentences/s) | 393.442623  | 484.8484848 | 638.2978723  | 952.3809524 |
  >| 加速比                 | 1           | 1.232323232 | 1            | 1.492063492 |

- Bert base (batch size = 40)

  >| RTX  2080Ti 单卡       | FP32      |             |            |                | FP16混合精度 |             |            |                |
  >| ---------------------- | --------- | ----------- | ---------- | -------------- | ------------ | ----------- | ---------- | -------------- |
  >|                        | oneflow   | oneflow-xla | tensorflow | tensorflow-xla | oneflow      | oneflow-xla | tensorflow | tensorflow-xla |
  >| 计算吞吐 (sentences/s) | 99.276    | 125.708     | 91.4       | 119.1          | 170.731      | 288.511     | 202.2      | 309.5          |
  >| 加速比                 | 1         | 1.26625     | 1          | 1.30306        | 1            | 1.690       | 1          | 1.53066        |

  >| RTX  2080Ti 2卡        | FP32      |             |            |                | FP16混合精度 |             |            |                |
  >| ---------------------- | --------- | ----------- | ---------- | -------------- | ------------ | ----------- | ---------- | -------------- |
  >|                        | oneflow   | oneflow-xla | tensorflow | tensorflow-xla | oneflow      | oneflow-xla | tensorflow | tensorflow-xla |
  >| 计算吞吐 (sentences/s) | 188.476   | 223.643     | 173.6      | 196.2          | 290.946      | 431.241     | 307.8      | 376.1          |
  >| 加速比                 | 1         | 1.18659     | 1          | 1.13018        | 1            | 1.482       | 1          | 1.22190        |
