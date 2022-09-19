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
#ifndef ONEFLOW_CORE_AUTO_PARALLEL_SBP_UTIL_H_
#define ONEFLOW_CORE_AUTO_PARALLEL_SBP_UTIL_H_

#include "oneflow/core/graph/op_graph.h"

namespace oneflow {
namespace auto_parallel {

// Judge whether we need the same SBP for both producer and consumer
bool RequireSameSbp(const OpNode* consumer, const std::string& ibn);

}  // namespace auto_parallel
}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PARALLEL_SBP_UTIL_H_
