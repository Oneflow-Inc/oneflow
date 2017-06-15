#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_MASTER_SERVICE_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_MASTER_SERVICE_H_

#include <grpc++/grpc++.h>

#include "oneflow/core/distributed_runtime/grpc_call.h"
#include "oneflow/core/distributed_runtime/grpc_master_service_impl.h"
#include "oneflow/core/distributed_runtime/master.h"

#include "tensorflow/core/platform/default/mutex.h"

#include "oneflow/core/distributed_runtime/master_service.pb.h"

#include "tensorflow/core/lib/core/threadpool.h"

namespace oneflow {

using ::grpc::ServerBuilder;

class GrpcMasterService {
 public:
  GrpcMasterService(Master* master, ::grpc::ServerBuilder* builder)
      : master_(master) {
    builder->RegisterService(&master_service_);
    cq_ = builder->AddCompletionQueue();
    core_num_ = std::thread::hardware_concurrency();
    compute_pool_ = new tensorflow::thread::ThreadPool(core_num_);
  }

  ~GrpcMasterService() {
    delete shutdown_alarm_;
  }

  void Shutdown() override {
    bool did_shutdown = false;
    {
      mutex_lock l(mu_);
      if (!is_shutdown_) {
        is_shutdown_ = true;
        did_shutdown = true;
      }
    }
    if (did_shutdown) {
      shutdown_alarm_ = 
        new ::grpc::Alam(cq_.get(), gpr_now(GPR_CLOCK_MONOTONIC), nullptr);
    }
  }  // Shutdown

#define ENQUEUE_REQUEST(method)                                           \
  do {                                                                    \
    mutex_lock l(mu_);                                                    \
    Call<GrpcMasterService, grpc::MasterService::AsyncService,            \
        method##Request, method##Response>::                              \
    EnqueueRequestForMethod(                                              \
        &master_service_, cq_.get(),                                      \
        static_cast<int>(GrpcMasterMethod::k##method),                    \
        &GrpcMasterService::method##Handler);                             \
  } while (0)

 private:
  Master* master_;
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
  grpc::MasterService::AsyncService master_service_;

  mutex mu_;
  bool is_shutdown_ GUARDED_BY(mu_);
  ::grpc::Alam* shutdown_alarm_ = nullptr;

  size_t core_num_;
  Tensorflow::thread::ThreadPool* compute_pool_ = nullptr;

  void Schedule(std::function<void()> f) {
    compute_pool_->Schedule(std::move(f));
  }

  template<class RequestMessage, class ResponseMessage>
  using MasterCall = Call<GrpcMasterService, grpc::MasterService::AsyncService,
                          RequestMessage, ResponseMessage>;

  void SendGraphHandler(MasterCall<SendGraphRequest,
                        SendGraphResponse>* call) {
    Schedule([this, call] {
      Status s = master_->SendGraph(&call->request, &call->response);
      call->SendResponse(s);
    });
  }  // Sendgraphhandler
};  // GrpcMasterService

}  // oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_MASTER_SERVICE_H_
