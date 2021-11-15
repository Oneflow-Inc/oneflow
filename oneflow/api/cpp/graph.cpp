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

#include "oneflow/api/cpp/graph.h"
#include <cstdio>
#include <fstream>
#include <istream>
#include <memory>
#include <sstream>
#include <string>
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/serving/saved_model.pb.h"

namespace oneflow_api {

void Graph::Load(const std::string& model_path, const Device& target_device) {
  // load from local directory, binary text format
  std::ifstream input(model_path.c_str());
  std::stringstream buffer;
  buffer << input.rdbuf();
  job_.ParseFromString(buffer.str());
  input.close();

  // create variable conf op
  oneflow::HashMap<std::string, std::shared_ptr<oneflow::one::Tensor>> variable_op_name_to_tensor;
  CreateVariableOp(variable_op_name_to_tensor, target_device, true);
}

void Graph::CreateVariableOp(oneflow::HashMap<std::string, std::shared_ptr<oneflow::one::Tensor>>&
                                 variable_op_name_to_tensor,
                             const Device& target_device, bool is_mirrored) {
  oneflow::OpGraph op_graph(job_);
  op_graph.ForEachNode([&](oneflow::OpNode* node) -> oneflow::Maybe<void> {
    std::cout << node->op().op_name() << std::endl;
    if (!node->op().op_conf().has_variable_conf()) { return oneflow::Maybe<void>::Ok(); }

    std::shared_ptr<oneflow::Shape> shape =
        std::make_shared<oneflow::Shape>(node->op().op_conf().variable_conf().shape());
    oneflow::DataType dtype = node->op().op_conf().variable_conf().data_type();
    bool is_lazy = false;        // TODO(zzk0): not very sure?
    bool requires_grad = false;  // Inference mode
    bool is_leaf = true;         // all parameters are leaf node

    if (is_mirrored) {
      // To create a MirroredTensor: shape, dtype, device, is_lazy, requires_grad, is_leaf
      auto device = target_device.GetDeviceSymbolSharedPtr();
      std::shared_ptr<oneflow::one::MirroredTensor> tensor =
          CHECK_JUST(oneflow::one::MirroredTensor::MakeTensor(shape, dtype, *device, is_lazy,
                                                              requires_grad, is_leaf));
      variable_op_name_to_tensor[node->op().op_name()] = tensor;
    }
    else {
      // To create a ConsistentTensor: shape, dtype, sbp + parallel desc, is_lazy, requires_grad,
    }
    return oneflow::Maybe<void>::Ok();
  });
}

}  // namespace oneflow_api
