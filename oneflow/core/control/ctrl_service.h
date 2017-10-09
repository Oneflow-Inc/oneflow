#ifndef ONEFLOW_CORE_CONTROL_CTRL_SERVICE_H_
#define ONEFLOW_CORE_CONTROL_CTRL_SERVICE_H_

#include "grpc++/grpc++.h"
#include "grpc++/impl/codegen/async_stream.h"
#include "grpc++/impl/codegen/async_unary_call.h"
#include "grpc++/impl/codegen/proto_utils.h"
#include "grpc++/impl/codegen/rpc_method.h"
#include "grpc++/impl/codegen/service_type.h"
#include "grpc++/impl/codegen/status.h"
#include "grpc++/impl/codegen/stub_options.h"
#include "grpc++/impl/codegen/sync_stream.h"
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
  OF_PP_MAKE_TUPLE_SEQ(PushPlan)      \
  OF_PP_MAKE_TUPLE_SEQ(ClearPlan)     \
  OF_PP_MAKE_TUPLE_SEQ(PullPlan)      \
  OF_PP_MAKE_TUPLE_SEQ(PushPort)      \
  OF_PP_MAKE_TUPLE_SEQ(ClearPort)     \
  OF_PP_MAKE_TUPLE_SEQ(PullPort)

enum class CtrlMethod {
#define MAKE_ENTRY(method) k##method,
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, CTRL_METHOD_SEQ)
#undef MAKE_ENTRY
};

const int32_t kCtrlMethodNum = OF_PP_INTERNAL_SEQ_SIZE(CTRL_METHOD_SEQ);

class CtrlService final {
 public:
  class Stub final {
   public:
    Stub(std::shared_ptr<grpc::ChannelInterface> channel);
#define DECLARE_STUB_METHOD(method)                   \
  grpc::Status method(grpc::ClientContext* context,   \
                      const method##Request& request, \
                      method##Response* response);

    OF_PP_FOR_EACH_TUPLE(DECLARE_STUB_METHOD, CTRL_METHOD_SEQ);

#undef DECLARE_STUB_METHOD

   private:
#define DECLARE_RPC_METHOD(method) const grpc::RpcMethod rpcmethod_##method##_;
    OF_PP_FOR_EACH_TUPLE(DECLARE_RPC_METHOD, CTRL_METHOD_SEQ);
#undef DECLARE_RPC_METHOD

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
