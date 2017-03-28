#include <iostream>
#include <memory>
#include <string>

#include "grpc++/alarm.h"
#include "grpc++/server_builder.h"

#include "net/async_service_interface.h"
#include "net/grpc_master_service.h"
#include "net/grpc_call.h"
#include "net/grpc_master_service_impl.h"
#include "proto/master.pb.h"
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

  void Shutdown() override {
    if(is_shutdown_) {
      shutdown_alarm_ = 
          new ::grpc::Alarm(cq_, gpr_now(GPR_CLOCK_MONOTONIC), nullptr);
    }
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
  void ExtendSessionHandler(
      MasterCall<ExtendSessionRequest, ExtendSessionResponse>* call) {
    ENQUEUE_REQUEST(ExtendSession, false);
  }

  void RunStepHandler(
      MasterCall<RunStepRequest, RunStepResponse>* call) {
    ENQUEUE_REQUEST(RunStep, true);
  }
  
  void CloseSessionHandler(
      MasterCall<CloseSessionRequest, CloseSessionResponse>* call){
    ENQUEUE_REQUEST(CloseSession, false);
  }

  void ListDevicesHandler(
      MasterCall<ListDevicesRequest, ListDevicesResponse>* call) {
    ENQUEUE_REQUEST(ListDevices, false);
  } 

  void ResetHandler(MasterCall<ResetRequest, ResetResponse>* call) {
    ENQUEUE_REQUEST(Reset, false);
  }
 private:
  ::grpc::ServerCompletionQueue* cq_;
  grpc::MasterService::AsyncService master_service_;
  bool is_shutdown_;
  ::grpc::Alarm* shutdown_alarm_;

#undef ENQUEUE_REQUEST
};

AsyncServiceInterface* NewGrpcMasterService(::grpc::ServerBuilder* builder){
  return new GrpcMasterService(builder);
}

}
