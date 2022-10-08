/*
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
*/
#include "oneflow/core/job/graph_scope_vars.h"
#include <vector>

namespace oneflow {

namespace {

std::vector<std::string>* GetPythonPathsToBeFilteredForDebuggingVar() {
  static thread_local std::vector<std::string> filtered_paths;
  return &filtered_paths;
}

std::vector<std::string>* GetPythonPathsToBeKeptForDebuggingVar() {
  static thread_local std::vector<std::string> kept_paths;
  return &kept_paths;
}

bool* GetGraphVerboseStepLr() {
  static thread_local bool graph_verbose_step_lr = false;
  return &graph_verbose_step_lr;
}

int32_t* GetGraphDebugMaxPyStackDepthVar() {
  static thread_local int32_t graph_debug_max_py_stack_depth = 2;
  return &graph_debug_max_py_stack_depth;
}

bool* GetGraphDebugModeFlag() {
  static thread_local bool graph_debug_mode_flag = false;
  return &graph_debug_mode_flag;
}

bool* GetGraphDebugOnlyUserPyStackFlag() {
  static thread_local bool graph_debug_only_user_py_stack = true;
  return &graph_debug_only_user_py_stack;
}
}  // namespace

bool IsOpenGraphVerboseStepLr() {
  auto* graph_verbose_step_lr = GetGraphVerboseStepLr();
  bool is_graph_verbose_step_lr = *graph_verbose_step_lr;
  return is_graph_verbose_step_lr;
}

void SetGraphVerboseStepLr(bool verbose) {
  auto* graph_verbose_step_lr = GetGraphVerboseStepLr();
  *graph_verbose_step_lr = verbose;
}

void InitPythonPathsToBeKeptAndFilteredForDebugging(const std::string& python_base_dir) {
  std::vector<std::string>* kept_paths = GetPythonPathsToBeKeptForDebuggingVar();
  kept_paths->clear();
  kept_paths->push_back(python_base_dir + "/test");
  kept_paths->push_back(python_base_dir + "/nn/modules");

  std::vector<std::string>* filtered_paths = GetPythonPathsToBeFilteredForDebuggingVar();
  filtered_paths->clear();
  filtered_paths->push_back(python_base_dir);
}

const std::vector<std::string>& GetPythonPathsToBeFilteredForDebugging() {
  return *GetPythonPathsToBeFilteredForDebuggingVar();
}
const std::vector<std::string>& GetPythonPathsToBeKeptForDebugging() {
  return *GetPythonPathsToBeKeptForDebuggingVar();
}

void SetGraphDebugMaxPyStackDepth(int32_t depth) { *GetGraphDebugMaxPyStackDepthVar() = depth; }
int32_t GetGraphDebugMaxPyStackDepth() { return *GetGraphDebugMaxPyStackDepthVar(); }

void SetGraphDebugMode(bool mode) { *GetGraphDebugModeFlag() = mode; }
bool GetGraphDebugMode() { return *GetGraphDebugModeFlag(); }

void SetGraphDebugOnlyUserPyStack(bool flag) { *GetGraphDebugOnlyUserPyStackFlag() = flag; }
bool GetGraphDebugOnlyUserPyStack() { return *GetGraphDebugOnlyUserPyStackFlag(); }
}  // namespace oneflow
