#include "oneflow/core/control/ctrl_service.h"
#include "grpc++/impl/codegen/client_unary_call.h"

namespace oneflow {

namespace {

const char* g_method_name[] = {
#define DEFINE_METHOD_NAME(method) \
  "/oneflow.CtrlService/" OF_PP_STRINGIZE(method),
    OF_PP_FOR_EACH_TUPLE(DEFINE_METHOD_NAME, CTRL_METHOD_SEQ)};

const char* GetMethodName(CtrlMethod method) {
  return g_method_name[static_cast<int32_t>(method)];
}

}  // namespace

CtrlService::Stub::Stub(std::shared_ptr<grpc::ChannelInterface> channel)
    :
#define INIT_RPC_METHOD_OBJ(method)                           \
  rpcmethod_##method##_(GetMethodName(CtrlMethod::k##method), \
                        grpc::RpcMethod::NORMAL_RPC, channel),
      OF_PP_FOR_EACH_TUPLE(INIT_RPC_METHOD_OBJ, CTRL_METHOD_SEQ)
          channel_(channel) {
}

#define DEFINE_STUB_METHOD(method)                                        \
  grpc::Status CtrlService::Stub::method(grpc::ClientContext* context,    \
                                         const method##Request& request,  \
                                         method##Response* response) {    \
    return grpc::BlockingUnaryCall(channel_.get(), rpcmethod_##method##_, \
                                   context, request, response);           \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_STUB_METHOD, CTRL_METHOD_SEQ)

std::unique_ptr<CtrlService::Stub> CtrlService::NewStub(
    const std::string& addr) {
  return of_make_unique<Stub>(
      grpc::CreateChannel(addr, grpc::InsecureChannelCredentials()));
}

CtrlService::AsyncService::AsyncService() {
  for (int32_t i = 0; i < kCtrlMethodNum; ++i) {
    AddMethod(
        new grpc::RpcServiceMethod(GetMethodName(static_cast<CtrlMethod>(i)),
                                   grpc::RpcMethod::NORMAL_RPC, nullptr));
    grpc::Service::MarkMethodAsync(i);
  }
}

}  // namespace oneflow
