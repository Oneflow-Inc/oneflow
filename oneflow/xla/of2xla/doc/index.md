### XLA documents

- 如何使用XLA

  Oneflow中XLA默认是关闭的，可以通过设置以下环境变量来开启XLA进行计算加速，或关闭XLA。

  ```shell
  # 开启XLA
  export FLAGS_use_xla_jit=true
  # 关闭XLA
  # export FLAGS_use_xla_jit=false
  ```

  开启了XLA之后，Job编译阶段程序会自动对整个计算图进行[子图切分]()，每个子图会被XLA编译成一个Executable并执行。您可以通过设置以下环境变量来调整子图切分的结果。

  ```shell
  export FLAGS_clustering_minimum_nodes=1
  export FLAGS_clustering_minimum_nodes=100
  export FLAGS_strict_clustering=true
  ```

  - FLAGS_clustering_minimum_nodes

    设置每个子图合并的op的最小数量。当子图包含的op数量小于该值时，则子图内的op不会被XLA编译。

  - FLAGS_clustering_minimum_nodes

    设置每个子图合并的op的最大数量。当子图包含的op数量大于该值时，则子图内的op不会被XLA编译。

  - FLAGS_strict_clustering

    op在合并时可能会互相破坏依赖，导致op的执行时机发生改变。可以设置环境变量FLFLAGS_strict_clustering=true来规避该行为，确保合并后op的执行时机不变。

    同时FLAGS_strict_clustering=true时会导致合并的子图较小，可能导致XLA丧失一些优化机会。FLAGS_strict_clustering默认设为true。

  如果需要输出每个子图的结果，您只需要在当前目录创建一个名为dump_subgraph的目录，在编译阶段生成的子图都会输出在该目录下（该功能仅用于debug目的，后续版本可能不支持）。

- 如何添加XLA op

  当前所有已支持的op实现都放在oneflow/xla/of2xla/ops目录下，添加新的XLA op时可以参考一下。

