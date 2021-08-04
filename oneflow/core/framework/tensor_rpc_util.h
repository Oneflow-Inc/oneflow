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
#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_RPC_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_RPC_UTIL_H_

#include "oneflow/core/framework/rpc_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/common/optional.h"

namespace oneflow {

class FlatTensorConsistency;

class CheckConsistencyAsyncRpcCtx : public AsyncRpcCtx {
 public:
  CheckConsistencyAsyncRpcCtx(
      const RpcToken& rpc_token, Symbol<one::ConsistentTensorMeta> tensor_meta,
      const Optional<Symbol<cfg::ParallelDistribution>>& consumer_parallel_distribution_constraint,
      const RpcToken& tensor_rpc_token)
      : AsyncRpcCtx(rpc_token),
        tensor_meta_(tensor_meta),
        consumer_parallel_distribution_constraint_(consumer_parallel_distribution_constraint),
        tensor_rpc_token_(tensor_rpc_token) {}

  ~CheckConsistencyAsyncRpcCtx() override;

  Maybe<void> PrepareSendBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                           std::function<void()>* Callback) override;

  Maybe<void> PrepareRecvBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                           std::function<void()>* Callback) override;

  Maybe<void> Check() const;

 private:
  Symbol<one::ConsistentTensorMeta> tensor_meta_;
  Optional<Symbol<cfg::ParallelDistribution>> consumer_parallel_distribution_constraint_;
  RpcToken tensor_rpc_token_;
  std::shared_ptr<FlatTensorConsistency> flat_tensor_consistency_;
};

Maybe<CheckConsistencyAsyncRpcCtx> LaunchTensorMetaConsistencyCheck(const one::Tensor& tensor);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_RPC_UTIL_H_
