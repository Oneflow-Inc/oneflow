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
#ifndef ONEFLOW_CORE_GRAPH_SUB_TASK_GRAPH_BUILDER_STATUS_UTIL_H_
#define ONEFLOW_CORE_GRAPH_SUB_TASK_GRAPH_BUILDER_STATUS_UTIL_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class SubTskGphBuilderStatus;

Maybe<SubTskGphBuilderStatus> BuildSubTskGphBuilderStatus(const std::string& builder_name,
                                                          const std::string& comment);

Maybe<SubTskGphBuilderStatus> MakeComposedSubTskGphBuilderStatus(
    const std::vector<SubTskGphBuilderStatus>& status);

class SubTskGphBuilderStatus final {
 public:
  SubTskGphBuilderStatus(const std::string& builder_name, const std::string& comment)
      : builder_name_(builder_name), comment_(comment){};
  ~SubTskGphBuilderStatus() = default;

  // Getters
  const std::string& builder_name() const { return builder_name_; }
  const std::string& comment() const { return comment_; }

 private:
  std::string builder_name_;
  std::string comment_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_SUB_TASK_GRAPH_BUILDER_STATUS_UTIL_H_
