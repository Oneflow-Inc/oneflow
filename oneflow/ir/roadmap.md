# Iree Roadmap

## 支持resnet的编译
### 1. Op转换添加
    - [困难][未做] variableOp
    - [困难][未做] NormalizationOp
        - TODO
    - [中等][未做] MatMulOp
        - TransposeOp
        - ReshapeOp
        - MatMulOp
    - [中等][已做] Conv2DOp
        - tosa::ConstOp
        - tosa::Conv2DOp
    - [简单][未做] FlattenOp
        - tosa::ReshapeOp
    - [简单][未做] Add2Op
        - tosa::AddOp
    - [简单][未做] BroadcastAddOp
        - tosa::AddOp
    - [简单][未做] MaxPool2DOp
        - MaxPool2DOp
    - [简单][已做] ReluOp
        - tosa::ReluNOp
    - [简单][已做] InputOp
    - [简单][已做] OutputOp
    - [简单][已做] JobOp
    - [简单][已做] ReturnOp

### 2. Op测试添加
添加上述Op的测试用例，用lit或unittest完成

### 3. VariableOp Conversion
    - 所有variableOp转换成Op的输出
    - 所有variableOp在-lower-oneflow-to-tosa转换为tosa::const, 通过pybind11获取graph.get_dict()['$model_name']['$param_name']的tensor实例值把它硬编码到tosa::const里面，如果无graph则生成truncated的const实例： %3 = "tosa.const"() {value = opaque<"elided_large_const">}
    - 通过iree—extension实现shared mem机制, 实现较难

### 4. NormailzationOp Conversion
TODO
#####
- 支持resnet的预测
TODO
- 支持resnet的训练
TODO

