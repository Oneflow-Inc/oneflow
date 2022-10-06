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
#include "oneflow/core/graph/boxing/sub_task_graph_builder_status_util.h"

namespace oneflow {

Maybe<SubTskGphBuilderStatus> BuildSubTskGphBuilderStatus(const std::string& builder_name,
                                                          const std::string& comment) {
  SubTskGphBuilderStatus status(builder_name, comment);
  return status;
}

Maybe<SubTskGphBuilderStatus> MakeComposedSubTskGphBuilderStatus(
    const std::vector<SubTskGphBuilderStatus>& status_vec) {
  std::string builder_name = "ComposedBuilder:";
  std::string comment = "ComposedComment:";
  for (auto status : status_vec) {
    builder_name += " ";
    builder_name += status.builder_name();
    comment += " ";
    if (status.comment().empty()) {
      comment += "None";
    } else {
      comment += status.comment();
    }
  }
  SubTskGphBuilderStatus composed_status(builder_name, comment);
  return composed_status;
}

}  // namespace oneflow
