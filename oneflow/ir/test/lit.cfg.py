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
# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "ONEFLOW"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.oneflow_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
    "Examples",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
    "networks",
    "test_fuse_cast_scale.mlir.py",
    "test_util.py",
    "test_mlir_opt.mlir.py",
    "lit.cfg.py",
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.oneflow_obj_root, "test")
config.oneflow_tools_dir = os.path.join(config.oneflow_ir_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

# TODO: these two should be unnecessary
llvm_config.with_environment(
    "LD_LIBRARY_PATH",
    os.path.join(config.oneflow_obj_root, "third_party_install/protobuf/lib"),
    append_path=True,
)
llvm_config.with_environment(
    "LD_LIBRARY_PATH",
    os.path.join(config.oneflow_obj_root, "_deps/glog-build"),
    append_path=True,
)

llvm_config.with_environment("ONEFLOW_MLIR_STDOUT", "1")
llvm_config.with_environment("ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS", "1")
llvm_config.with_environment("ONEFLOW_MLIR_ENABLE_ROUND_TRIP", "1")
llvm_config.with_environment("ONEFLOW_MLIR_CSE", "1")
llvm_config.with_environment("ONEFLOW_MLIR_FUSE_FORWARD_OPS", "1")
llvm_config.with_environment(
    "PYTHONPATH", os.path.join(config.oneflow_src_root, "python"), append_path=True,
)

tool_dirs = [config.oneflow_tools_dir, config.llvm_tools_dir]
tools = ["oneflow-opt", "oneflow-translate", "oneflow-runner"]
tools.extend(
    [
        ToolSubst("%with_cuda", config.BUILD_CUDA, unresolved="ignore"),
        ToolSubst("%linalg_test_lib_dir", config.llvm_lib_dir, unresolved="ignore"),
        ToolSubst("%test_exec_root", config.test_exec_root, unresolved="ignore"),
    ]
)
llvm_config.add_tool_substitutions(tools, tool_dirs)

try:
    from iree import runtime as ireert
    from iree.compiler import compile_str

    config.WITH_ONEFLOW_IREE = True
except ImportError:
    config.WITH_ONEFLOW_IREE = False
