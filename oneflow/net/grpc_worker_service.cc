#include <iostream>
#include <memory>
#include <string>

//#include "grpc_worker_service_impl.h"

#include <grpc++/grpc++.h>
#include "grpc++/server_builder.h"

#include "async_service_interface.h"
#include "grpc_worker_service.h"
#include "grpc_worker_service_impl.h"
#include "grpc_call.h"
#include "worker.pb.h"


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

  template <class RequestMessage, class ResponseMessage>
  using WorkerCall = Call<GrpcWorkerService, grpc::WorkerService::AsyncService,
                          RequestMessage, ResponseMessage>;

  void GetStatusHandler(WorkerCall<GetStatusRequest, GetStatusResponse>* call) {
    std::cout<<"hello, I am worker"<<std::endl;
    ENQUEUE_REQUEST(GetStatus, false);
  }

};

AsyncServiceInterface* NewGrpcWorkerService(::grpc::ServerBuilder* builder){
  return new GrpcWorkerService(builder);
}

}
