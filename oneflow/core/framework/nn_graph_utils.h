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
#ifndef ONEFLOW_CORE_FRAMEWORK_NN_GRAPH_UTILS_H_
#define ONEFLOW_CORE_FRAMEWORK_NN_GRAPH_UTILS_H_

#include <set>
#include <string>

namespace oneflow {
// A templated function that broadcasts data from the master process to worker processes in a
// multi-threaded manner. Return push/pull keys only in master process.
template<typename X, typename Y>
std::set<std::string> MultiThreadBroadcastFromMasterToWorkers(size_t world_size,
                                                              const std::string& prefix,
                                                              const X& master_data, Y* worker_data);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_NN_GRAPH_UTILS_H_