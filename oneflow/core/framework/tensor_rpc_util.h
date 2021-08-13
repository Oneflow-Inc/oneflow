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

#include "oneflow/core/framework/transport_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

struct FlatTensorConsistency;

class CheckConsistencyAsyncTransportCtx : public AsyncTransportCtx {
 public:
  CheckConsistencyAsyncTransportCtx(
      const TransportToken& transport_token, Symbol<one::ConsistentTensorMeta> tensor_meta,
      const Optional<Symbol<cfg::ParallelDistribution>>& consumer_nd_sbp_constraint,
      const TransportToken& tensor_transport_token)
      : AsyncTransportCtx(transport_token),
        tensor_meta_(tensor_meta),
        consumer_nd_sbp_constraint_(consumer_nd_sbp_constraint),
        tensor_transport_token_(tensor_transport_token) {}

  ~CheckConsistencyAsyncTransportCtx() override;

  Maybe<void> PrepareSendBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                           std::function<void()>* Callback) override;

  Maybe<void> PrepareRecvBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                           std::function<void()>* Callback) override;

  Maybe<void> Check() const;

 private:
  Symbol<one::ConsistentTensorMeta> tensor_meta_;
  Optional<Symbol<cfg::ParallelDistribution>> consumer_nd_sbp_constraint_;
  TransportToken tensor_transport_token_;
  std::shared_ptr<FlatTensorConsistency> flat_tensor_consistency_;
};

Maybe<CheckConsistencyAsyncTransportCtx> LaunchTensorMetaConsistencyCheck(
    const one::Tensor& tensor);

template<typename... Args>
struct CheckConsistentTensorMeta;

template<typename RetT, typename... Args>
struct CheckConsistentTensorMeta<RetT, const one::Tensor&, Args...> {
  static_assert(is_maybe<RetT>::value, "returned value type must be Maybe<T>.");
  template<RetT (*func)(const one::Tensor&, Args...)>
  static RetT Call(const one::Tensor& tensor, Args... args) {
    const auto& ctx = JUST(LaunchTensorMetaConsistencyCheck(tensor));
    RetT&& ret = func(tensor, args...);
    // Always synchronize consistent tensor meta even if `func` failed.
    JUST(TransportUtil::WaitUntilDoneOrTimeout(*ctx, TransportUtil::TimeoutSeconds()));
    JUST(ctx->Check());
    return ret;
  }
};

template<typename RetT, typename... Args>
struct CheckConsistentTensorMeta<RetT, const std::shared_ptr<one::Tensor>&, Args...> {
  static_assert(is_maybe<RetT>::value, "returned value type must be Maybe<T>.");
  template<RetT (*func)(const std::shared_ptr<one::Tensor>&, Args...)>
  static RetT Call(const std::shared_ptr<one::Tensor>& tensor, Args... args) {
    const auto& ctx = JUST(LaunchTensorMetaConsistencyCheck(*tensor));
    LOG(ERROR) << "rank: " << GlobalProcessCtx::Rank()
               << "\ntransport_token:" << static_cast<int64_t>(ctx->transport_token());
    RetT&& ret = func(tensor, args...);
    // Always synchronize consistent tensor meta even if `func` failed.
    JUST(TransportUtil::WaitUntilDoneOrTimeout(*ctx, TransportUtil::TimeoutSeconds()));
    JUST(ctx->Check());
    return ret;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_RPC_UTIL_H_
