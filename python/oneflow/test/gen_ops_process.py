"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import subprocess

from numpy import triu_indices
import oneflow as flow
import oneflow.nn as nn

api_list = [
    "Tensor",  # flow.xxx
    "BoolTensor",
    "ByteTensor",
    "CharTensor",
    "DoubleTensor",
    "FloatTensor",
    "HalfTensor",
    "IntTensor",
    "LongTensor",
    "Size",
    "abs",
    "acos",
    "acosh",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "add",
    "addmm",
    "any",
    "arange",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "argmax",
    "argmin",
    "argsort",
    "argwhere",
    "as_strided",
    "as_tensor",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "autograd",
    "batch_gather",
    "bernoulli",
    "bfloat16",
    "bmm",
    "bool",
    "boxing",
    "broadcast_like",
    "cast",
    "cat",
    "ceil",
    "char",
    "chunk",
    "clamp",
    "clamp_",
    "clip",
    "clip_",
    "concat",
    "constant_initializer",
    "convert_oneflow_dtype_to_numpy_dtype",
    "cos",
    "cosh",
    "cumprod",
    "cumsum",
    "device",
    "diag",
    "diagonal",
    "distributed_partial_fc_sample",
    "div",
    "div_",
    "dot",
    "double",
    "dtype",
    "dtypes",
    "einsum",
    "empty",
    "eq",
    "equal",
    "erf",
    "erfc",
    "erfinv",
    "erfinv_",
    "exp",
    "expand",
    "expm1",
    "eye",
    "flatten",
    "flip",
    "float",
    "float16",
    "float32",
    "float64",
    "floor",
    "floor_",
    "floor_divide",
    "fmod",
    "from_numpy",
    "full",
    "gather",
    "gather_nd",
    "ge",
    "gelu",
    "glorot_normal_initializer",
    "glorot_uniform_initializer",
    "grad_enable",
    "greater",
    "greater_equal",
    "gt",
    "half",
    "hsplit",
    "in_top_k",
    "index_select",
    "int",
    "int32",
    "int64",
    "int8",
    "is_floating_point",
    "is_grad_enabled",
    "is_nonzero",
    "is_tensor",
    "kaiming_initializer",
    "le",
    "linalg_flow",
    "linalg_matrix_norm",
    "linalg_norm",
    "linalg_vector_norm",
    "linspace",
    "log",
    "log1p",
    "log2",
    "log_softmax",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "long",
    "lt",
    "manual_seed",
    "masked_fill",
    "masked_select",
    "matmul",
    "max",
    "maximum",
    "mean",
    "meshgrid",
    "min",
    "minimum",
    "mish",
    "movedim",
    "mul",
    "narrow",
    "ne",
    "neg",
    "negative",
    "new_ones",
    "nms",
    "no_grad",
    "nonzero",
    "not_equal",
    "numel",
    "one_embedding",
    "ones",
    "ones_initializer",
    "ones_like",
    "pad",
    "permute",
    "placement",
    "pow",
    "prod",
    "randint",
    "randn",
    "random_normal_initializer",
    "random_uniform_initializer",
    "randperm",
    "reciprocal",
    "relu",
    "repeat",
    "reshape",
    "roi_align",
    "roll",
    "round",
    "rsqrt",
    "save",
    "sbp",
    "scatter",
    "scatter_add",
    "select",
    "selu",
    "set_num_threads",
    "set_printoptions",
    "set_rng_state",
    "sigmoid",
    "sign",
    "silu",
    "sin",
    "sin_",
    "sinh",
    "slice",
    "slice_update",
    "softmax",
    "softplus",
    "softshrink",
    "softsign",
    "sort",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "stateful_op",
    "std",
    "sub",
    "sum",
    "support",
    "swapaxes",
    "t",
    "tan",
    "tanh",
    "tensor_buffer",
    "tensor_buffer_to_list_of_tensors",
    "tensor_buffer_to_tensor",
    "tensor_scatter_nd_update",
    "tensor_split",
    "tensor_to_tensor_buffer",
    "tile",
    "to_global",
    "to_local",
    "topk",
    "transpose",
    "tril",
    "triu",
    "truncated_normal_initializer",
    "uint8",
    "unsqueeze",
    "var",
    "variance_scaling_initializer",
    "version",
    "view",
    "vsplit",
    "where",
    "xavier_normal_initializer",
    "xavier_uniform_initializer",
    "zero_",
    "zeros",
    "zeros_initializer",
    "zeros_like",
    "Adagrad",  # oneflow.optim.xxx
    "Adam",
    "AdamW",
    "LAMB",
    "RMSprop",
    "SGD",
    "ChainedScheduler",  # oneflow.optim.lr_scheduler.xxx
    "ConstantLR",
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "CosineDecayLR",
    "ExponentialLR",
    "LambdaLR",
    "LinearLR",
    "MultiStepLR",
    "PolynomialLR",
    "ReduceLROnPlateau",
    "SequentialLR",
    "StepLR",
    "WarmUpLR",
    "AdaptiveAvgPool1d",  # oneflow.nn.xxx
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AllReduce",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "BCELoss",
    "BCEWithLogitsLoss",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "CELU",
    "COCOReader",
    "CTCLoss",
    "CoinFlip",
    "CombinedMarginLoss",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "CropMirrorNormalize",
    "CrossEntropyLoss",
    "DistributedPariticalFCSample",
    "Dropout",
    "ELU",
    "Embedding",
    "FakeQuantization",
    "Flatten",
    "Fold",
    "FusedBatchNorm1d",
    "FusedBatchNorm2d",
    "FusedBatchNorm3d",
    "FusedMLP",
    "GELU",
    "GLU",
    "GPTIndexedBinDataReader",
    "GRU",
    "GroupNorm",
    "Hardsigmoid",
    "Hardswish",
    "Hardtanh",
    "Identity",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "KLDivLoss",
    "L1Loss",
    "LSTM",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "LogSigmoid",
    "LogSoftmax",
    "MSELoss",
    "MarginRankingLoss",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "MinMaxObserver",
    "Mish",
    "Module",
    "ModuleDict",
    "ModuleList",
    "MovingAverageMinMaxObserver",
    "NLLLoss",
    "PReLU",
    "Parameter",
    "ParameterDict",
    "ParameterList",
    "PixelShuffle",
    "Quantization",
    "RNN",
    "ReLU",
    "ReLU6",
    "ReflectionPad2d",
    "ReplicationPad2d",
    "SELU",
    "Sequential",
    "SiLU",
    "Sigmoid",
    "SmoothL1Loss",
    "Softmax",
    "Softplus",
    "Softshrink",
    "Softsign",
    "Tanh",
    "TripletMarginLoss",
    "Unfold",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
    "ZeroPad2d",
    "adaptive_avg_pool1d",  # oneflow.nn.functional.xxx
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "affine_grid",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "celu",
    "conv1d",
    "conv2d",
    "conv3d",
    "cross_entropy",
    "ctc_greedy_decoder",
    "dropout",
    "elu",
    "embedding",
    "functional_maxpool",
    "gelu",
    "glu",
    "grid_sample",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "interpolate",
    "layer_norm",
    "leaky_relu",
    "linear",
    "log_softmax",
    "logsigmoid",
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "mish",
    "normalize",
    "one_hot",
    "pad",
    "prelu",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "silu",
    "smooth_l1_loss",
    "softmax",
    "softplus",
    "softshrink",
    "softsign",
    "sparse_softmax_cross_entropy",
    "tanh",
    "triplet_margin_loss",
    "upsample",
    "CalcGain",  # flow.nn.init.xxx
    "calculate_gain",
    "constant_",
    "flow",
    "kaiming_normal_",
    "kaiming_uniform_",
    "normal_",
    "ones_",
    "os",
    "trunc_normal_",
    "uniform_",
    "xavier_normal_",
    "xavier_uniform_",
    "zeros_",
    "adagrad",  # flow.nn.optimizer.xxx
    "adam",
    "adamw",
    "chained_scheduler",
    "constant_lr",
    "cosine_annealing_lr",
    "cosine_annealing_warm_restarts",
    "cosine_decay_lr",
    "exponential_lr",
    "lamb",
    "lambda_lr",
    "linear_lr",
    "lr_scheduler",
    "multistep_lr",
    "polynomial_lr",
    "reduce_lr_on_plateau",
    "rmsprop",
    "sequential_lr",
    "sgd",
    "step_lr",
    "warmup_lr",
]

dir_list = [
    ["../../../python/oneflow/framework/docstr"],
    ["../../../python/oneflow/test/modules", "../../../python/oneflow/test/tensor"],
    ["../../../python/oneflow/test/exceptions"],
]
num_cols = 4

test_func_list = list()
file_func_map = dict()
file_func_map_list = []


def get_test_func(path):
    files = os.listdir(path)
    commit_bytes = subprocess.check_output(["git", "rev-parse", "HEAD"])
    commit_str = commit_bytes.decode("utf-8").replace("\n", "")
    result_func_list = []
    for file in files:
        if not os.path.isdir(file) and file.find("__pycache__") == -1:
            f = open(path + "/" + file)
            last_line = ""
            iter_f = iter(f)
            line_num = 1
            for line in iter_f:
                line = line.strip()
                if line.startswith("def test_") and line.endswith("(test_case):"):
                    result_func_list.append(line[9:-12])
                    file_func_map[line[9:-12]] = (
                        f" [{line[9:-12]}]("
                        + "https://github.com/Oneflow-Inc/oneflow/blob/"
                        + commit_str
                        + "/python/oneflow/test/"
                        + path
                        + "/"
                        + file
                        + f"#L{line_num}) "
                    )
                elif last_line.startswith("add_docstr"):
                    result_func_list.append(line[0:-1])
                    file_func_map[line[0:-1]] = (
                        f" [{line[0:-1]}]("
                        + "https://github.com/Oneflow-Inc/oneflow/blob/"
                        + commit_str
                        + "/python/oneflow/test/"
                        + path
                        + "/"
                        + file
                        + f"#L{line_num}) "
                    )
                last_line = line
                line_num += 1
    return result_func_list


for i in range(0, len(dir_list)):
    tmp_func_list = list()
    file_func_map = dict()
    for path in dir_list[i]:
        tmp_func_list.extend(get_test_func(path))
    test_func_list.append(tmp_func_list)
    file_func_map_list.append(file_func_map)


def pure_match(x, y):
    x = x.lower()
    x = x.split("_")[0]
    y = y.lower()
    pos = x.find(y)
    if pos != -1:
        return True
    else:
        return False


def match_test_func(func, func_list):
    match_res = ""
    for i in range(len(func_list)):
        if pure_match(func_list[i], func):
            match_res = func_list[i]
            break
    return match_res


result_list = []
result_list.append(f"## Ops Version : Alpha")
result_list.append(f"")
result_list.append(f"")
table_head = f"|op name   | Doc Test | Compatiable/Completeness Test | Exception |"
result_list.append(table_head)
result_list.append(
    f"| ------------------------- | ------------- | ----------------------------- | --------- |"
)

cnt0 = 0
cnt1 = 0
cnt2 = 0

pre = ""

for name in api_list:
    if name == "Tensor":
        pre = "oneflow."
    elif name == "Adagrad":
        pre = "oneflow.optim."
    elif name == "ChainedScheduler":
        pre = "oneflow.optim.lr_scheduler."
    elif name == "AdaptiveAvgPool1d":
        pre = "oneflow.nn."
    elif name == "adaptive_avg_pool1d" and pre == "oneflow.nn.":
        pre = "oneflow.nn.functional."
    elif name == "CalcGain":
        pre = "oneflow.nn.init."
    table_line = f"| {pre+name} |"
    for i in range(3):
        match_name = match_test_func(name, test_func_list[i])
        if match_name != "":
            if i == 0:
                cnt0 += 1
            elif i == 1:
                cnt1 += 1
            else:
                cnt2 += 1
            table_line += file_func_map_list[i][match_name]
        table_line += "  |"
    result_list.append(table_line)

doc_test_ratio = cnt0 * 1.0 / len(api_list)
compatiable_completeness_test_ratio = cnt1 * 1.0 / len(api_list)
exception_test_ratio = cnt2 * 1.0 / len(api_list)

result_list.append(f"## Test Data Summary")

result_list.append(f"- OneFlow Total API Number: ====================>{len(api_list)}")
result_list.append(
    f"- Doc Test Ratio: ====================>{100*doc_test_ratio:.2f}% = {cnt0} / {len(api_list)}"
)
result_list.append(
    f"- Compatiable/Completeness Test Ratio: ====================>{100*compatiable_completeness_test_ratio:.2f}% = {cnt1} / {len(api_list)}"
)
result_list.append(
    f"- Exception Test Ratio: ====================>{100*exception_test_ratio:.2f}% = {cnt2} / {len(api_list)}"
)

f = open("./README.md", "w")
for line in result_list:
    f.write(line + "\n")
f.close()
