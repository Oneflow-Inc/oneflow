#include <iostream>
#include <memory>
#include <string>

#include <grpc++/grpc++.h>

#include "grpc++/server_builder.h"

#include "net/async_service_interface.h"
#include "net/grpc_worker_service.h"
#include "net/grpc_worker_service_impl.h"
#include "net/grpc_call.h"
#include "proto/worker.pb.h"


namespace oneflow{

class GrpcWorkerService : public AsyncServiceInterface{
 public:
  GrpcWorkerService(::grpc::ServerBuilder* builder){
    builder->RegisterService(&worker_service_);  
    cq_ = builder->AddCompletionQueue().release();
  }

  ~GrpcWorkerService() {
    delete cq_;
  }
  
  void Shutdown() override {
    if(is_shutdown_){
      shutdown_alarm_ = 
          new ::grpc::Alarm(cq_, gpr_now(GPR_CLOCK_MONOTONIC), nullptr);      
    }
  }

#define  ENQUEUE_REQUEST(method, supports_cancel)                              \
  do {                                                                        \
    if (!is_shutdown_) {                                                      \
      Call<GrpcWorkerService, grpc::WorkerService::AsyncService,              \
           method##Request, method##Response>::                               \
          EnqueueRequest(&worker_service_, cq_,                               \
                         &grpc::WorkerService::AsyncService::Request##method, \
                         &GrpcWorkerService::method##Handler,                 \
                         (supports_cancel));                                  \
    }                                                                         \
  } while (0)


  void HandleRPCsLoop() {
    ENQUEUE_REQUEST(GetStatus, false);
  }

  ::grpc::ServerCompletionQueue* cq_;
  grpc::WorkerService::AsyncService worker_service_;
  bool is_shutdown_;
  ::grpc::Alarm* shutdown_alarm_;

  template <class RequestMessage, class ResponseMessage>
  using WorkerCall = Call<GrpcWorkerService, grpc::WorkerService::AsyncService,
                          RequestMessage, ResponseMessage>;

  void GetStatusHandler(WorkerCall<GetStatusRequest, GetStatusResponse>* call) {
    ENQUEUE_REQUEST(GetStatus, false);
  }

  void CleanupAllHandler(WorkerCall<CleanupAllRequest, CleanupAllResponse>* call) {
    ENQUEUE_REQUEST(CleanupAll, false);
  }
 
  void RegisterGraphHandler(WorkerCall<RegisterGraphRequest, RegisterGraphResponse>* call) {
    ENQUEUE_REQUEST(RegisterGraph, false);
  }

  void DeregisterGraphHandler(WorkerCall<DeregisterGraphRequest, DeregisterGraphResponse>* call) {
    ENQUEUE_REQUEST(DeregisterGraph, false);
  }

  void RunGraphHandler(WorkerCall<RunGraphRequest, RunGraphResponse>* call) {
    ENQUEUE_REQUEST(RunGraph, true);
  }

  void CleanupGraphHandler(WorkerCall<CleanupGraphRequest, CleanupGraphResponse>* call) {
    ENQUEUE_REQUEST(CleanupGraph, false);
  }
 
  void LoggingHandler(WorkerCall<LoggingRequest, LoggingResponse>* call) {
    ENQUEUE_REQUEST(Logging, false);
  }

  void TracingHandler(WorkerCall<TracingRequest, TracingResponse>* call) {
    ENQUEUE_REQUEST(Tracing, false);
  }
};

AsyncServiceInterface* NewGrpcWorkerService(::grpc::ServerBuilder* builder){
  return new GrpcWorkerService(builder);
}

}
