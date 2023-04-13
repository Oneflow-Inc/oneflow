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
#ifndef ONEFLOW_CORE_FRAMEWORK_MULTI_CLIENT_SESSION_CONTEXT_H_
#define ONEFLOW_CORE_FRAMEWORK_MULTI_CLIENT_SESSION_CONTEXT_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/job/env_global_objects_scope.h"

namespace oneflow {

class MultiClientSessionContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultiClientSessionContext);
  explicit MultiClientSessionContext(const std::shared_ptr<EnvGlobalObjectsScope>&);
  ~MultiClientSessionContext();

  Maybe<void> TryInit(const ConfigProto& config_proto);
  Maybe<void> TryInit(const std::string& config_proto_str);
  Maybe<void> UpdateResource(const Resource& reso_proto);
  Maybe<void> UpdateResource(const std::string& reso_proto_str);

  Maybe<void> TryClose();

  // NOTE(chengcheng): for nn.Graph catch free EagerTensor in Graph.build().
  //   NNGraph should NOT hold ANY shared_ptr<Tensor> because NNGraph will send to VM stream in
  //   RunLazyNNGraphInstruction, the tensor in NNGraph will Never be released for hold in VM
  //   instrunction and compute stream. So we store free EagerTensor in MultiClientSessionContext,
  //   and will be release in NNGraph destructor.
  void StoreFreeEagerTensorWithNameByGraphName(const std::string& graph_name,
                                               const std::shared_ptr<one::Tensor>& tensor,
                                               const std::string& tensor_name);
  const std::vector<std::pair<std::string, std::shared_ptr<one::Tensor>>>&
  GetFreeEagerTensorNamePairByGraphName(const std::string& graph_name);
  void RemoveGraphFreeEagerTensors(const std::string& graph_name);

 private:
  bool is_inited_ = false;
  std::shared_ptr<EnvGlobalObjectsScope> env_ctx_;
  HashMap<std::string, std::vector<std::pair<std::string, std::shared_ptr<one::Tensor>>>>
      graph_name2free_eager_tensors_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_MULTI_CLIENT_SESSION_CONTEXT_H_
