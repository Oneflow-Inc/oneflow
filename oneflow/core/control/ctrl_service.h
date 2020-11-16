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
#ifndef ONEFLOW_CORE_CONTROL_CTRL_SERVICE_H_
#define ONEFLOW_CORE_CONTROL_CTRL_SERVICE_H_

#include <grpc++/grpc++.h>
#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/proto_utils.h>
#include <grpc++/impl/codegen/rpc_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/status.h>
#include <grpc++/impl/codegen/stub_options.h>
#include <grpc++/impl/codegen/sync_stream.h>
#include <grpc++/impl/codegen/client_unary_call.h>
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/control/control.pb.h"

namespace oneflow {

#define CTRL_METHOD_SEQ               \
  OF_PP_MAKE_TUPLE_SEQ(LoadServer)    \
  OF_PP_MAKE_TUPLE_SEQ(Barrier)       \
  OF_PP_MAKE_TUPLE_SEQ(TryLock)       \
  OF_PP_MAKE_TUPLE_SEQ(NotifyDone)    \
  OF_PP_MAKE_TUPLE_SEQ(WaitUntilDone) \
  OF_PP_MAKE_TUPLE_SEQ(PushKV)        \
  OF_PP_MAKE_TUPLE_SEQ(ClearKV)       \
  OF_PP_MAKE_TUPLE_SEQ(PullKV)        \
  OF_PP_MAKE_TUPLE_SEQ(PushActEvent)  \
  OF_PP_MAKE_TUPLE_SEQ(Clear)         \
  OF_PP_MAKE_TUPLE_SEQ(IncreaseCount) \
  OF_PP_MAKE_TUPLE_SEQ(EraseCount)

#define CatRequest(method) method##Request,
#define CatReqponse(method) method##Response,
#define CatEnum(method) k##method,
#define CatName(method) "/oneflow.CtrlService/" OF_PP_STRINGIZE(method),

#define MAKE_META_DATA()                                                                       \
  enum class CtrlMethod { OF_PP_FOR_EACH_TUPLE(CatEnum, CTRL_METHOD_SEQ) };                    \
  static const char* g_method_name[] = {OF_PP_FOR_EACH_TUPLE(CatName, CTRL_METHOD_SEQ)};       \
  using CtrlRequestTuple = std::tuple<OF_PP_FOR_EACH_TUPLE(CatRequest, CTRL_METHOD_SEQ) void>; \
  using CtrlResponseTuple = std::tuple<OF_PP_FOR_EACH_TUPLE(CatReqponse, CTRL_METHOD_SEQ) void>;

MAKE_META_DATA()

constexpr const size_t kCtrlMethodNum = OF_PP_SEQ_SIZE(CTRL_METHOD_SEQ);

template<CtrlMethod ctrl_method>
using CtrlRequest =
    typename std::tuple_element<static_cast<size_t>(ctrl_method), CtrlRequestTuple>::type;

template<CtrlMethod ctrl_method>
using CtrlResponse =
    typename std::tuple_element<static_cast<size_t>(ctrl_method), CtrlResponseTuple>::type;

inline const char* GetMethodName(CtrlMethod method) {
  return g_method_name[static_cast<int32_t>(method)];
}

class CtrlService final {
 public:
  class Stub final {
   public:
    Stub(std::shared_ptr<grpc::ChannelInterface> channel);

    template<CtrlMethod ctrl_method>
    grpc::Status CallMethod(grpc::ClientContext* context, const CtrlRequest<ctrl_method>& request,
                            CtrlResponse<ctrl_method>* response) {
      return grpc::internal::BlockingUnaryCall(channel_.get(),
                                               rpcmethods_.at(static_cast<size_t>(ctrl_method)),
                                               context, request, response);
    }

   private:
    std::array<const grpc::internal::RpcMethod, kCtrlMethodNum> rpcmethods_;

    std::shared_ptr<grpc::ChannelInterface> channel_;
  };

  static std::unique_ptr<Stub> NewStub(const std::string& addr);

  class AsyncService final : public grpc::Service {
   public:
    AsyncService();
    ~AsyncService() = default;
    using grpc::Service::RequestAsyncUnary;
  };
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CONTROL_CTRL_SERVICE_H_
