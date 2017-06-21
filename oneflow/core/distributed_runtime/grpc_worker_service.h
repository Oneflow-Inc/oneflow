#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_SERVER_SERVICE_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_SERVER_SERVICE_H_

#include <thread>
#include "grpc++/alarm.h"
#include "grpc++/server_builder.h"

#include "oneflow/core/distributed_runtime/grpc_call.h"
#include "oneflow/core/distributed_runtime/grpc_worker_service_impl.h"
#include "oneflow/core/distributed_runtime/worker.h"
#include "oneflow/core/distributed_runtime/worker_service.pb.h"

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace oneflow {

class GrpcWorkerService {
 public:
  GrpcWorkerService(::grpc::ServerBuilder* builder) {
    builder->RegisterService(&worker_service_);
    cq_ = builder->AddCompletionQueue();
    core_num_ = std::thread::hardware_concurrency();
    compute_pool_ = 
      new ::tensorflow::thread::ThreadPool(
          ::tensorflow::Env::Default(), 
          "worker_service", core_num_);
  }

  ~GrpcWorkerService() {
    delete shutdown_alarm_;
  }

  void Shutdown() {
    bool did_shutdown = false;
    {
      ::tensorflow::mutext_lock l(mu_);
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

#define ENQUEUE_REQUEST(method, supports_cancel)                          \
  do {                                                                    \
    ::tensorflow::mutext_lock l(mu_);                                     \
    if (!is_shutdown_) {                                                  \
      Call<GrpcWorkerService, grpc::WorkerService::AsyncService,          \
           method##Request, method##Response>::                           \
      EnqueueRequestForMethod(                                            \
          &worker_service_, cq_.get(),                                    \
          static_cast<int32_t>(GrpcWorkerMethod::k##method),              \
          &GrpcWorkerService::method##Handler,                            \
          (supports_cancel));                                             \
    }                                                                     \
  } while (0)

  void EnqueueGetStatusMethod() {
    ENQUEUE_REQUEST(GetStatus, false);
  }

 public:
  Worker* worker_;
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
  grpc::WorkerService::AsyncService worker_service_;
  GrpcWorker* grpc_worker_;

  ::tensorflow::mutext mu_;
  bool is_shutdown_ GUARDED_BY(mu_);
  ::grpc::Alarm* shutdown_alarm_ = nullptr;

  size_t core_num_;
  ::tensorflow::thread::ThreadPool* compute_pool_ = nullptr;
  void Schedule(std::function<void()> f) {
    compute_pool_->Schedule(std::move(f));
  }

 private:
  template <class RequestMessage, class ResponseMessage>
  using WorkerCall = Call<GrpcWorkerService, grpc::WorkerService::AsyncService,
                          RequestMessage, ResponseMessage>;

  void GetStatusHandler(WorkerCall<GetStatusRequest,
                                   GetStatusResponse>* call) {
    pool_->enqueue([this, call] {
      ::tensorflow::Status status = worker_->GetStatus();
      call->SendResponse(status);
    });
  }

  void GetMachineDescHandler(WorkerCall<GetMachineDescRequest,
                                        GetMachineDescResponse>* call) {
    pool_->enqueue([this, call] {
      ::tensorflow::Status status 
        = worker_->GetMachineDesc(&call->request, &call->response);
      call->SendResponse(status);
    });
  }

  void GetMemoryDescHandler(WorkerCall<GetMemoryDescRequest, 
                                        GetMemoryDescResponse>* call) {
    pool_->enqueue([this, call] {
      ::tensorflow::Status status 
        = worker_->GetMemoryDesc(&call->request, &call->response);
      call->SendResponse(status);
    });
  }

  void SendTaskGraphHandler(WorkerCall<SendTaskGraphRequest,
                                       SendTaskGraphResponse>* call) {
    pool_->enqueue([this, call] {
      ::tensorflow::Status status 
        = worker_->SendTaskGraph(&call->request, &call->response);
      call->SendResponse(status);
    });
  }
#undef ENQUEUE_REQUEST

  void SendMessageHandler(WorkerCall<SendMessageRequest, 
                                     SendMessageResponse>* call) {
    pool_->enqueue([this, call] {
      ::tensorflow::Status status 
        = worker_->SendMessage(&call->request, &call->response);
      call->SendResponse(status);
    });
  }

  void EnqueueReadDataRaw() {
    Call<GrpcWorkerService, grpc::WorkerService::AsyncService,
         ReadDataRequest, ::grpc::ByteBuffer>::
         EnqueueRequestForMethod(
           &worker_service_, cq_.get(),
           static_cast<int>(GrpcWorkerMethod::kReadData),
           &GrpcWorkerService::ReadDataRaw);
  }

  void ReadDataRaw(WorkerCall<ReadDataRequest, ::grpc::ByteBuffer>* call) {
    pool_->enqueue([this, call] {
      ::tensorflow::Status status 
        = grpc_worker_->ReadData(&call->request, &call->response);
      call->SendResponse(status);
      EnqueueReadDataRaw();
    });
  }  // Readdataraw

};  // Grpcworkerservice

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_SERVER_SERVICE_H_
