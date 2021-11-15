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

#ifndef ONEFLOW_API_CPP_GRAPH_H_
#define ONEFLOW_API_CPP_GRAPH_H_

#include <string>
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/api/cpp/device.h"

namespace oneflow_api {

class Graph {
 public:
  void Load(const std::string& model_path, const Device& device);

 private:
  void CreateVariableOp(oneflow::HashMap<std::string, std::shared_ptr<oneflow::one::Tensor>>&
                            variable_op_name_to_tensor,
                        const Device& target_device, bool is_mirrored);

  oneflow::Job job_;
};

// TODO(zzk0): model_path is a single file or a directory, it depends on how parameters are stored
inline Graph load(const std::string& model_path, const Device& device) {
  Graph graph;
  graph.Load(model_path, device);
  return graph;
}

inline Graph load(const std::string& model_path) {
  Device device = Device("cpu");
  return load(model_path, device);
}

}  // namespace oneflow_api

#endif  // ONEFLOW_API_CPP_GRAPH_H_
