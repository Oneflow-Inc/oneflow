## 概述
描述 op 的功能、公式等。若参考了其它框架的接口，应列出超链接。

## 功能 CheckList
**注意** : 功能复选框均为可选项，若未选择，说明理由即可。例如：该 Op 由 Python 接口拼接而成，因此无 `SetBatchAxisInferFn` Op 注册；再比如：该 Op 无输入，因此无 `SetInputArgModifyFn`。

模板中自带的复选框可留空，但是不能删除。可根据实际情况增加复选框选项。

### Op
 - [ ] Op SetBatchAxisInferFn
 - [ ] Op SetGetSbpFn
 - [ ] Op SetInputArgModifyFn
 - [ ] Op 反向梯度注册

### Kernel
 - [ ] CPU in:float32
 - [ ] CPU in:float64
 - [ ] CPU in:int32
 - [ ] CPU in:int64
 - [ ] CPU in:int8

 - [ ] GPU in:float32
 - [ ] GPU in:float64
 - [ ] GPU in:int32
 - [ ] GPU in:int64
 - [ ] GPU in:float16
 - [ ] GPU in:int8


### Python Wrapper
 - [ ] Python API 参数检查及异常提示
 - [ ] 接口注释
 - [ ] Example 

### 测试
 - [ ] 单机单卡  CPU Test Case
 - [ ] 单机单卡  GPU Test Case
 - [ ] 单机多卡  CPU Test Case
 - [ ] 单机多卡  GPU Test Case
 - [ ] 分布式  CPU Test Case
 - [ ] 分布式  GPU Test Case

## GPU 有效带宽
带 GPU 的 Op，请参考 https://github.com/Oneflow-Inc/OneTeam/issues/167 测试有效带宽，并附带测试报告。
以下是报告样例：

理论带宽：
```text
 Device to Device Bandwidth, 1 Device(s)
 PINNED Memory Transfers
   Transfer Size (Bytes)	Bandwidth(MB/s)
   33554432			250798.5
```

实际带宽：
```
PROFILER::KERNEL::CUDA_MEMORY_BANDWIDTH op_name: sqrt_2 elapsed(ms): 0.196064 memory_size(Byte): 50331648 bandwidth(GB/s): 239.08
PROFILER::KERNEL::CUDA_MEMORY_BANDWIDTH op_name: sqrt_2_grad elapsed(ms): 0.29072 memory_size(Byte): 75497472 bandwidth(GB/s): 241.856
```


## PR Checklist
 - [ ] PR 标题语句通畅，明确表达 PR 内容，适合直接作为新版本发布时的 changelog
 - [ ] 代码格式化
 - [ ] 已经本地编译通过
 - [ ] 已本地针对改动测试
 - [ ] 已添加 type 标签:(填写 type 标签名，如 `bug, enhancement, purge, feature, documentation`)
 - [ ] 已添加 component 标签:(填写 component 标签名，如 `op, system, eager, build, xla, python, ci, test, tooling`)
 - [ ] Draft 转正式 PR 前已请人 Review
