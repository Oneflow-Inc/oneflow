## XRT (X-Runtime)

XRT是一个同时支持多个计算引擎的运行时加速库，目前已经集成了TensorFlow XLA和Nvidia TensorRT两个后端引擎。其中XLA全面支持训练和预测，TensorRT支持预测以及部分算子支持训练。对于同一个计算图，XRT允许多个计算引擎联合使用，以获得更好的加速效果。

不同的后端引擎支持不同的后端硬件，比如XLA支持CPU和Nvidia GPU，但TensorRT仅支持Nvidia GPU。

对于任意后端引擎，XRT的执行过程均分成以下四个步骤：

1. 计算图的转换
2. 划分计算子图
3. 引擎无关优化
4. 生成引擎相关Executable
5. 执行Executable

### Build with XLA

- #### Install Bazel

  Download and install bazel from [here](https://docs.bazel.build/versions/1.0.0/bazel-overview.html) , and version 0.24.1 is recommended. You can confirm bazel is installed successfully by running the following command:

  ```shell
  bazel version
  ```

- #### Build Third Parties

  Inside directory `build`, run:

  ```shell
  cmake -DWITH_XLA=ON -DTHIRD_PARTY=ON -DCMAKE_BUILD_TYPE=Release ..
  make -j$(nproc)
  ```

  If the downloading error occurred, you should go back to the previous step to reinstall the cmake, then clean the file CMakeCache.txt and build the third-parties once again.

- #### Build OneFlow

  Inside directory `build`, run:
  ```shell
  cmake .. \
  -DWITH_XLA=ON \
  -DTHIRD_PARTY=OFF \
  -DCMAKE_BUILD_TYPE=Release
  
  make -j$(nproc)
  ```

### Build with TensorRT

- #### Build Third Parties

  1. Download TensorRT(>=6.0) .tgz and unzip the package.
  
  2. Inside directory `build`, run:
  
  ```shell
  cmake -DWITH_TENSORRT=ON -DTENSORRT_ROOT=your_tensorrt_path -DTHIRD_PARTY=ON ..
  make -j$(nproc)
  ```
- #### Build OneFlow

  Inside directory `build`, run:
  ```shell
  cmake .. \
  -DWITH_TENSORRT=ON \
  -DTENSORRT_ROOT=your_tensorrt_path \
  -DTHIRD_PARTY=OFF

  make -j$(nproc)
  ```

### 计算图的转换

  将OneFlow Job转换成XRT的计算流图 (XrtGraph)，该计算流图经过一序列变换后，最终被编译成后端引擎相关的Executable。

### 划分计算子图

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

### 引擎无关优化

暂未提供，后续可以加入一些图优化相关的pass。

### Executable的生成

在runtime阶段，每个计算子图都可以被编译成一个与引擎相关的Executable。

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

  config = flow.function_config()

  # 配置使用XLA
  config.use_xla_jit()

  # 配置使用TensorRT
  config.use_tensorrt()
  ```

- 从环境变量配置

  ```shell
  # 只在Python前端未定义状态下生效
  export FLAGS_use_xla_jit=true # true为开启，false为关闭
  export FLAGS_use_tensorrt=true # true为开启，false为关闭
  ```

- 低精度配置

  ```python
  # XLA自动混合精度(float16)
  config.enable_auto_mixed_precision()

  # TensorRT float16
  config.tensorrt.use_fp16()

  # TensorRT int8 (离线加载Calibration的方式)
  config.tensorrt.use_int8()
  # Set int8 calibration table path
  int8_calibration_path = "./int8_calibration"
  config.tensorrt.int8_calibration(int8_calibration_path)
  ```

#### 使用Int8量化计算

XRT支持离线加载和在线生成量化校准表两种方式来启动Int8的量化计算。离线加载的方式需要提前生成一个TensorRT格式的量化校准表，而且该量化校准表通常可以被重复使用，而在线生成的方式则在同一份脚本中，同时进行正常精度的计算和量化校准表的生成，一旦校准表生成后，则会在下一个迭代中自动切换到Int8精度的计算。

- 生成Int8量化校准表(Int8 Calibration Table)

  首先你需要为生成量化校准表准备一个校准数据集，通常可以是训练集或验证集的一个子集。然后按照正常的网络配置，开启TensorRT Int8。比如：

  ```python
  import oneflow as flow

  config = flow.function_config()

  config.use_tensorrt()
  config.tensorrt.use_int8()

  @flow.function(config)
  def Job(input):
      # define your network
      pass
  ```
  当开启Int8，但又没有指定对应的量化校准表时，XRT会自动进入量化表生成模式，之后feed的数据都会按照正常的精度（fp32或fp16）进行计算，计算的结果会被用于生成对应的Int8量化校准表。最后将生成的量化校准表保存到指定的目录，在该目录下，每一个子图都会生成一个对应的量化校准表文件。
  
  ```python
  # 使用10个batch的数据生成Int8量化校准表
  for _ in range(10):
      input = next_calibration_batch() # 加载校准数据集
      Job(input).get()

  # 保存量化校准表
  flow.tensorrt.write_int8_calibration("./int8_calibration") # int8_calibration目录需要手动创建
  ```
  当Int8量化校准表生成完成后，你就可以按照上面介绍的离线加载Calibration的方式启动TensorRT Int8的量化计算。

- 在线生成量化校准表并进行int8计算

  在线方式分成两个步骤，首先利用校准数据集生成量化校准表，然后直接利用生成的量化校准表进行Int8的构图和计算。同样以上面的Job为例，

  ```python
  # 使用10个batch的数据生成Int8量化校准表
  for _ in range(10):
    input = next_calibration_batch() # 加载校准数据集
    Job(input).get()

  # 缓存量化校准表
  flow.tensorrt.cache_int8_calibration()

  # 当量化校准表cache完成后，XRT会自动切换到int8的计算
  for _ in range(100):
    input = next_batch() # 加载数据
    Job(input).get()
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
