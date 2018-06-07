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
  OF_PP_MAKE_TUPLE_SEQ(EraseCount)    \
  OF_PP_MAKE_TUPLE_SEQ(PushAvgActInterval)

enum class CtrlMethod {
#define MAKE_ENTRY(method) k##method,
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, CTRL_METHOD_SEQ)
};
#undef MAKE_ENTRY

const int32_t kCtrlMethodNum = OF_PP_SEQ_SIZE(CTRL_METHOD_SEQ);

using CtrlRequestTuple = std::tuple<
#define MAKE_ENTRY(method) method##Request,
    OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, CTRL_METHOD_SEQ) void>;
#undef MAKE_ENTRY

using CtrlResponseTuple = std::tuple<
#define MAKE_ENTRY(method) method##Response,
    OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, CTRL_METHOD_SEQ) void>;
#undef MAKE_ENTRY

template<CtrlMethod ctrl_method>
using CtrlRequest =
    typename std::tuple_element<static_cast<size_t>(ctrl_method), CtrlRequestTuple>::type;
template<CtrlMethod ctrl_method>
using CtrlResponse =
    typename std::tuple_element<static_cast<size_t>(ctrl_method), CtrlResponseTuple>::type;

class CtrlService final {
 public:
  class Stub final {
   public:
    Stub(std::shared_ptr<grpc::ChannelInterface> channel);

    template<CtrlMethod ctrl_method>
    grpc::Status CallMethod(grpc::ClientContext* context, const CtrlRequest<ctrl_method>& request,
                            CtrlResponse<ctrl_method>* response) {
      return grpc::BlockingUnaryCall(channel_.get(),
                                     rpcmethods_.at(static_cast<size_t>(ctrl_method)), context,
                                     request, response);
    }

   private:
    std::array<const grpc::RpcMethod, kCtrlMethodNum> rpcmethods_;

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
