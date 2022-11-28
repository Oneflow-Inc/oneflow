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
#ifndef ONEFLOW_CORE_JOB_GRAPH_SCOPE_VARS_H_
#define ONEFLOW_CORE_JOB_GRAPH_SCOPE_VARS_H_

#include <cstdint>
#include <string>
#include <vector>

namespace oneflow {

bool IsOpenGraphVerboseStepLr();
void SetGraphVerboseStepLr(bool verbose);

void SetGraphDebugMaxPyStackDepth(int32_t depth);
int32_t GetGraphDebugMaxPyStackDepth();
void SetGraphDebugMode(bool mode);
bool GetGraphDebugMode();
void SetGraphDebugOnlyUserPyStack(bool flag);
bool GetGraphDebugOnlyUserPyStack();
void InitPythonPathsToBeKeptAndFilteredForDebugging(const std::string& python_base_dir);
const std::vector<std::string>& GetPythonPathsToBeFilteredForDebugging();
const std::vector<std::string>& GetPythonPathsToBeKeptForDebugging();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_GRAPH_SCOPE_VARS_H_
