#ifndef ONEFLOW_CORE_COMM_NETWORK_CTRL_SERVICE_H_
#define ONEFLOW_CORE_COMM_NETWORK_CTRL_SERVICE_H_

#include "grpc++/impl/codegen/async_stream.h"
#include "grpc++/impl/codegen/async_unary_call.h"
#include "grpc++/impl/codegen/proto_utils.h"
#include "grpc++/impl/codegen/rpc_method.h"
#include "grpc++/impl/codegen/service_type.h"
#include "grpc++/impl/codegen/status.h"
#include "grpc++/impl/codegen/stub_options.h"
#include "grpc++/impl/codegen/sync_stream.h"
#include "oneflow/core/comm_network/control.pb.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

#define CTRL_METHOD_SEQ           \
  OF_PP_MAKE_TUPLE_SEQ(AddWorker) \
  OF_PP_MAKE_TUPLE_SEQ(Barrier)

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
    grpc::Status AddWorker(grpc::ClientContext* context,
                           const AddWorkerRequest& request,
                           AddWorkerResponse* response);
    grpc::Status Barrier(grpc::ClientContext* context,
                         const BarrierRequest& request,
                         BarrierResponse* response);

   private:
    const grpc::RpcMethod rpcmethod_AddWorker_;
    const grpc::RpcMethod rpcmethod_Barrier_;
    std::shared_ptr<grpc::ChannelInterface> channel_;
  };

  class AsyncService final : public grpc::Service {
    AsyncService();
    ~AsyncService() = default;
  };
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_CTRL_SERVICE_H_
