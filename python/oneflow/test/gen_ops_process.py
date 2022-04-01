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

from numpy import triu_indices
import oneflow as flow
import oneflow.nn as nn

api_dict = {
    "tensor": flow.Tensor,
    "bool_tensor": flow.BoolTensor,
    "byte_tensor": flow.ByteTensor,
    "char_tensor": flow.CharTensor,
    "double_tensor": flow.DoubleTensor,
    "float_tensor": flow.FloatTensor,
    "half_tensor": flow.HalfTensor,
    "int_tensor": flow.IntTensor,
    "long_tensor": flow.LongTensor,
    "size": flow.Size,
    "abs": flow.abs,
    "acos": flow.acos,
    "acosh": flow.acosh,
}
dir_list = [
    [],
    [],
    ["../../../python/oneflow/test/modules", "../../../python/oneflow/test/tensor"],
    ["../../../python/oneflow/test/exceptions"],
]
num_cols = 4

test_func_list = list()
file_func_map = dict()
file_func_map_list = []


def get_test_func(path):
    files = os.listdir(path)
    result_func_list = []
    for file in files:
        if not os.path.isdir(file) and file.startswith("test_"):
            f = open(path + "/" + file)
            iter_f = iter(f)
            for line in iter_f:
                line = line.strip()
                if line.startswith("def test_") and line.endswith("(test_case):"):
                    result_func_list.append(line[9:-12])
                    file_func_map[line[9:-12]] = path + "/" + file
    return result_func_list


for i in range(1, len(dir_list)):
    tmp_func_list = list()
    file_func_map = dict()
    for path in dir_list[i]:
        tmp_func_list.extend(get_test_func(path))
    test_func_list.append(tmp_func_list)
    file_func_map_list.append(file_func_map)


def match_test_func(func, func_list):
    match_res = ""
    for i in range(len(func_list)):
        if func_list[i].find(func) != -1:
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


for name, func in api_dict.items():
    table_line = f"| {name} |"
    for i in range(3):
        match_name = match_test_func(name, test_func_list[i])
        if match_name != "":
            table_line += file_func_map_list[i][match_name]
        table_line += "  |"
    result_list.append(table_line)

f = open("./README.md", "w")
for line in result_list:
    f.write(line + "\n")
f.close()
