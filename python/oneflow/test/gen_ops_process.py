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
import glob


op_files = glob.glob("../../../docs/source/*.rst")
op_files.remove("../../../docs/source/graph.rst")
op_prefix = ""
api_list = []
api_str = ""
for op_file in op_files:
    with open(op_file, "r") as f:
        line = True
        while line:
            line = f.readline()
            if ":members:" in line:
                api_str += line.replace(":members:", "")
                line = f.readline()
                while line and ":" not in line:
                    api_str += line
                    line = f.readline()
            if ".. autofunction::" in line:
                api_str += line.replace(".. autofunction::", "")
            if ".. currentmodule::" in line:
                api_str += line.split(".")[-1]
            if ".. autoclass::" in line:
                api_str += line.split(".")[-1]

api_list = api_str.strip().replace(" ", "").replace(",", "").split("\n")  

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
