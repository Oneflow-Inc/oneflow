#include <iostream>
#include <memory>
#include <string>

#include "grpc++/alarm.h"
#include "grpc++/server_builder.h"

#include "async_service_interface.h"
#include "grpc_master_service.h"
#include "grpc_call.h"
#include "grpc_master_service_impl.h"
#include "master.pb.h"
//#include "master.grpc.pb.h"

namespace oneflow{

class GrpcMasterService : public AsyncServiceInterface {
 public:
  GrpcMasterService(::grpc::ServerBuilder* builder){
    builder->RegisterService(&master_service_);
    cq_ = builder->AddCompletionQueue().release();
  }
  
  ~GrpcMasterService() {
    delete cq_;
  }
  
#define ENQUEUE_REQUEST(method, supports_cancel)                              \
  do {                                                                        \
    if (!is_shutdown_) {                                                      \
      Call<GrpcMasterService, grpc::MasterService::AsyncService,              \
           method##Request, method##Response>::                               \
          EnqueueRequest(&master_service_, cq_,                               \
                         &grpc::MasterService::AsyncService::Request##method, \
                         &GrpcMasterService::method##Handler,                 \
                         (supports_cancel));                                  \
    }                                                                         \
  } while (0)

  void HandleRPCsLoop() override {
    ENQUEUE_REQUEST(CreateSession, true);
  }

  template <class RequestMessage, class ResponseMessage>
  using MasterCall = Call<GrpcMasterService, grpc::MasterService::AsyncService,
                          RequestMessage, ResponseMessage>;

  void CreateSessionHandler(
      MasterCall<CreateSessionRequest, CreateSessionResponse>* call) {
    ENQUEUE_REQUEST(CreateSession, true);
  }
 private:
  ::grpc::ServerCompletionQueue* cq_;
  grpc::MasterService::AsyncService master_service_;
  bool is_shutdown_;

#undef ENQUEUE_REQUEST
};

AsyncServiceInterface* NewGrpcMasterService(::grpc::ServerBuilder* builder){
  return new GrpcMasterService(builder);
}

}
