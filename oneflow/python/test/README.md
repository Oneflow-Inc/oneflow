# 测试工具使用简介

Created: Oct 4, 2020 10:52 AM

对op的测试代码进行了更新，主要解决的问题：

1. 旧代码在 python 自带的 unittest 上引入了一些还有点复杂的抽象，导致并行运行单元测试很难做到
2. 不能随意运行单一脚本，比如 `python3 oneflow/python/test/ops/test_add.py` 这样
3. 对于启动测试的配置信息，都要靠命令行传入，对 CI 不友好

新的编写规范：

```cpp
@flow.unittest.skip_unless_1n1d()
class TestAdd(flow.unittest.TestCase):
    def test_naive(test_case):
        ....

    def test_broadcast(test_case):
        ....

if __name__ == "__main__":
    unittest.main()
```

- 必须把`test__***` 函数写在一个继承 `flow.unittest.TestCase` 的类里面
- 必须加一个 `if __name__ == "__main__":` ，里面调用 `unittest.main()`
- 必须加上 skip decorator，比如 `@flow.unittest.skip_unless_1n1d()` 标记这个测试用例只在1 node 1 device 的情况下才能运行。注意：这里的 device 不仅要考虑到 oneflow 用了几个 gpu，还要考虑到这个脚本里面 tensorflow/pytorch 用到了几个 gpu
- skip decorator 可以放在 class 头上也可以放在 method 头上，放在 class 头上的话，不满足条件整个 class 内部所有 test method 都会跳过
- 在 python unit test 的规范上没有引入额外的抽象，了解更多：[https://docs.python.org/3/library/unittest.html](https://docs.python.org/3/library/unittest.html)

如何运行：

- 整体运行，进入 `oneflow/python/test/ops`目录，运行`python3 -m unittest`

    ```cpp
    oneflow/python/test/ops
    export ONEFLOW_TEST_DEVICE_NUM=1
    python3 -m unittest --failfast --verbose
    ```

    或者：

    ```cpp
    python3 -m unittest discover oneflow/python/test/ops
    ```

    更多用法请参考 [https://docs.python.org/3/library/unittest.html](https://docs.python.org/3/library/unittest.html)

- 通过设置环境变量 `ONEFLOW_TEST_DEVICE_NUM` 过滤要运行几卡的脚本，如果没有给，默认就是1
- 多机脚本需要设置 `ONEFLOW_TEST_NODE_LIST` 和`ONEFLOW_TEST_MASTER_PORT`环境变量来指定多机的 ip 地址和 control port
- 运行单一脚本，可以直接用 python3 二进制运行一个文件，接受 python unitest 的所有命令行参数，如 `--failfast` , `--verbose`

    ```cpp
    python3 oneflow/python/test/ops/test_add.py --verbose
    ```
