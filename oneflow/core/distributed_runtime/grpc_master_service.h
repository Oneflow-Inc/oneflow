#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_MASTER_SERVICE_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_MASTER_SERVICE_H_

#include <thread>
#include <grpc++/grpc++.h>
#include "grpc++/alarm.h"
#include "grpc++/server_builder.h"

#include "oneflow/core/distributed_runtime/grpc_call.h"
#include "oneflow/core/distributed_runtime/grpc_master_service_impl.h"
#include "oneflow/core/distributed_runtime/master.h"

#include "tensorflow/core/platform/mutex.h"

#include "oneflow/core/distributed_runtime/master_service.pb.h"

#include "tensorflow/core/lib/core/threadpool.h"

namespace oneflow {

class GrpcMasterService {
 public:
  GrpcMasterService(Master* master, ::grpc::ServerBuilder* builder)
      : master_(master), is_shutdown_(false) {
    builder->RegisterService(&master_service_);
    cq_ = builder->AddCompletionQueue();
    //core_num_ = std::thread::hardware_concurrency();
    //compute_pool_ = new ::tensorflow::thread::ThreadPool(tensorflow::Env::Default(), "master_service", core_num_);
  }

  ~GrpcMasterService() {
    delete shutdown_alarm_;
  }

  void Shutdown() {
    bool did_shutdown = false;
    {
      ::tensorflow::mutex_lock l(mu_);
      if (!is_shutdown_) {
        is_shutdown_ = true;
        did_shutdown = true;
      }
    }
    if (did_shutdown) {
      shutdown_alarm_ = 
        new ::grpc::Alarm(cq_.get(), gpr_now(GPR_CLOCK_MONOTONIC), nullptr);
    }
  }  // Shutdown

/*
#define ENQUEUE_REQUEST(method, supports_cancel)                          \
  do {                                                                    \
    ::tensorflow::mutex_lock l(mu_);                                      \
    if (!is_shutdown_) {                                                  \
      Call<GrpcMasterService, grpc::MasterService::AsyncService,          \
        method##Request, method##Response>::EnqueueRequest(               \
          &master_service_, cq_.get(),                                    \
          &grpc::MasterService::AsyncService::Request##method,            \
          &GrpcMasterService::method##Handler,                            \
          (supports_cancel));                                             \
    }                                                                     \
  } while (0)
*/
#define ENQUEUE_REQUEST(method, supports_cancel)                          \
  do {                                                                    \
    ::tensorflow::mutex_lock l(mu_);                                      \
    if (!is_shutdown_) {                                                  \
      Call<GrpcMasterService, grpc::MasterService::AsyncService,          \
        method##Request, method##Response>::EnqueueRequestForMethod(      \
          &master_service_, cq_.get(),                                    \
          static_cast<int16_t>(GrpcMasterMethod::k##method),              \
          &GrpcMasterService::method##Handler,                            \
          (supports_cancel));                                             \
    }                                                                     \
  } while (0)

  void EnqueueSendGraphMethod() {
    ENQUEUE_REQUEST(SendGraph, true);
    std::cout<<"hi~"<<std::endl;
  }

  void test() {
    std::cout<<"hehe"<<std::endl;
  }

 public:
  Master* master_ = nullptr;
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
  grpc::MasterService::AsyncService master_service_;

  ::tensorflow::mutex mu_;
  bool is_shutdown_ GUARDED_BY(mu_);
  ::grpc::Alarm* shutdown_alarm_ = nullptr;

  size_t core_num_;
  tensorflow::thread::ThreadPool* compute_pool_ = nullptr;

  void Schedule(std::function<void()> f) {
    compute_pool_->Schedule(std::move(f));
  }

  template<class RequestMessage, class ResponseMessage>
  using MasterCall = Call<GrpcMasterService, grpc::MasterService::AsyncService,
                          RequestMessage, ResponseMessage>;

  void SendGraphHandler(MasterCall<SendGraphRequest,
                        SendGraphResponse>* call) {
    //Schedule([this, call] {
       ::tensorflow::Status s = master_->SendGraph(&call->request, &call->response);
       call->SendResponse(ToGrpcStatus(s));
    //});
    //ENQUEUE_REQUEST(SendGraph, true);
  }  // Sendgraphhandler

#undef ENQUEUE_REQUEST
};  // GrpcMasterService

}  // oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_MASTER_SERVICE_H_
