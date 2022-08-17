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
import re


def get_api(rst_dir):
    """
    Extract operator names from rst files.

    `currentmodule` is not regarded as operators.
    `autoclass` and `automodule` are regarded as operators in the absence of `members`.
    """
    op_files = glob.glob(rst_dir + "/*.rst")
    op_files.remove(rst_dir + "/graph.rst")
    op_files.remove(rst_dir + "/index.rst")
    api_list = []
    api_str = ""
    for op_file in op_files:
        with open(op_file, "r") as f:
            line = f.readline()
            pre = ""
            while line:
                skip = False
                if ".. currentmodule::" in line:
                    pre = line.strip().replace(".. currentmodule::", "") + "."
                elif ".. autofunction::" in line:
                    if "oneflow" not in line:
                        api_str += pre
                    api_str += line.replace(".. autofunction::", "")
                elif (
                    ".. autosummary::" in line
                    or ".. autoclass::" in line
                    or ":toctree:" in line
                    or ":nosignatures:" in line
                    or ":template:" in line
                ):
                    if ":nosignatures:" in line:
                        line = f.readline()
                        if ":template:" in line:
                            line = f.readline()
                        line = f.readline()
                        while line and len(line.replace(" ", "")) > 1:
                            if "oneflow" not in line:
                                api_str += pre
                            api_str += line
                            line = f.readline()
                elif ".. automodule::" in line:
                    pre_a = line.replace(".. automodule:: ", "")
                    line = f.readline()
                    skip = True
                    if ":members:" in line and len(line) > 14:
                        pre_a = pre_a.strip() + "."
                        api_str += pre_a + line.replace(":members:", "")
                        line = f.readline()
                        while (
                            line and ":" not in line and len(line.replace(" ", "")) > 1
                        ):
                            api_str += pre_a + line
                            line = f.readline()
                if not skip:
                    line = f.readline()

    api_list = api_str.strip().replace(" ", "").replace(",", "").split("\n")
    return api_list


def get_profile_func(path):
    """
    Iterate through files under `path` to find out all operator names,
    and update code links to file_func_map_list by file_func_map.
    """
    files = os.listdir(path)
    commit_bytes = subprocess.check_output(["git", "rev-parse", "HEAD"])
    commit_str = commit_bytes.decode("utf-8").replace("\n", "")
    result_profile_func_list = []
    for file in files:
        if file != "log" and not os.path.isdir(file) and file.find("__pycache__") == -1:
            f = open(os.path.join(path, file))
            last_line = ""
            iter_f = iter(f)
            line_num = 1
            for line in iter_f:
                line = line.strip()
                match = re.fullmatch(r"^@profile\((.+)\)$", line)
                if match:
                    tem_profile = match.group(1)
                    tem_profile_name = tem_profile.split(".")[-1]
                    result_profile_func_list.append(tem_profile_name)

    return result_profile_func_list


def get_test_func(path):
    """
    Iterate through files under `path` to find out all operator names,
    and update code links to file_func_map_list by file_func_map.
    """
    files = os.listdir(path)
    commit_bytes = subprocess.check_output(["git", "rev-parse", "HEAD"])
    commit_str = commit_bytes.decode("utf-8").replace("\n", "")
    result_func_list = []
    for file in files:
        if file != "log" and not os.path.isdir(file) and file.find("__pycache__") == -1:
            f = open(os.path.join(path, file))
            last_line = ""
            iter_f = iter(f)
            line_num = 1
            for line in iter_f:
                line = line.strip()
                rem = re.match("def .*?(test_.*)\(test_case.*", line)
                if rem and "#" not in line:
                    func_name = rem.group(1).replace("_test_", "").replace("test_", "")
                    result_func_list.append(func_name)
                    file_func_map[func_name] = (
                        f" [{func_name}]("
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


def pure_match(x, y):
    """
    Check whether x contains y.

    The purpose of identifying "." is to accurately match operator documents.
    For example, if we make pos = x.find(y) while y = clip_, either oneflow.Tensor.clip or oneflow.Tensor.clip_ is right.

    Besides, identifying "_" is important.
    For example, if we make pos = x.find(y) while y = squeeze, either test of squeeze or unsqueeze is right.
    """
    x = x.lower()
    y = y.lower()
    pos = -1
    if "." in x:
        x = x.split(".")
        for i in x:
            if i == y:
                pos = 1
                break
    elif "_" in y:
        pos = x.find(y)
    else:
        x = x.split("_")
        for i in x:
            if i == y:
                pos = 1
                break
    return pos != -1


def match_test_func(func, func_list):
    """
    func: operator name
    func_list: names of all operators

    Check whether func_list contains func. If yes, return matching content, or else return "".
    """
    match_res = ""
    for i in range(len(func_list)):
        if pure_match(func_list[i], func):
            match_res = func_list[i]
            break
    return match_res


if __name__ == "__main__":
    api_list = get_api("../../../docs/source")
    dir_list = [
        ["../../../python/oneflow/framework/docstr"],
        ["../../../python/oneflow/test/modules", "../../../python/oneflow/test/tensor"],
        ["../../../python/oneflow/test/exceptions"],
    ]
    num_cols = 4
    test_func_list = list()
    test_profile_list = list()
    file_func_map = dict()
    file_func_map_list = []

    for i in range(0, len(dir_list)):
        tmp_func_list = list()
        tmp_profile_list = list()
        file_func_map = dict()
        for path in dir_list[i]:
            tmp_func_list.extend(get_test_func(path))
            tmp_profile_list.extend(get_profile_func(path))
        test_func_list.append(tmp_func_list)
        test_profile_list.extend(tmp_profile_list)
        file_func_map_list.append(file_func_map)

    result_list = []
    result_list.append(f"## Ops Version : Alpha")
    result_list.append(f"")
    result_list.append(f"")
    table_head = f"| Op Name | Doc Test | Compatiable/Completeness Test | Exception | Performance Test |"
    result_list.append(table_head)
    result_list.append(
        f"| ------------------------- | ------------- | ----------------------------- | --------- | ---------------- |"
    )

    cnt0 = 0  # the number of doc_test
    cnt1 = 0  # the number of compatiable_completeness_test
    cnt2 = 0  # the number of exception_test
    cnt3 = 0  # the number of profile_test

    for name in api_list:
        table_line = f"| {name} |"
        name = name.split(".")[-1]
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
        if name in test_profile_list:
            table_line += " done "
            cnt3 += 1
        table_line += "  |"

        result_list.append(table_line)

    doc_test_ratio = cnt0 / len(api_list)
    compatiable_completeness_test_ratio = cnt1 / len(api_list)
    exception_test_ratio = cnt2 / len(api_list)
    performance_test_ratio = cnt3 / len(api_list)

    result_list.append(f"## Test Data Summary")
    result_list.append(f"- OneFlow Total API Number: {len(api_list)}")
    result_list.append(
        f"- Doc Test Ratio: {100*doc_test_ratio:.2f}% ({cnt0} / {len(api_list)})"
    )
    result_list.append(
        f"- Compatiable/Completeness Test Ratio: {100*compatiable_completeness_test_ratio:.2f}% ({cnt1} / {len(api_list)})"
    )
    result_list.append(
        f"- Exception Test Ratio: {100*exception_test_ratio:.2f}% ({cnt2} / {len(api_list)})"
    )
    result_list.append(
        f"- Performance Test Ratio: {100*performance_test_ratio:.2f}% ({cnt3} / {len(api_list)})"
    )
    f = open("./README.md", "w")
    for line in result_list:
        f.write(line + "\n")
    f.close()
