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

namespace grpc {
class ByteBuffer;
}

namespace oneflow {

class GrpcWorkerService {
 public:
  GrpcWorkerService(Worker* worker, ::grpc::ServerBuilder* builder) 
    : worker_(worker), is_shutdown_(false) {
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

#define ENQUEUE_REQUEST(method, supports_cancel)                          \
  do {                                                                    \
    ::tensorflow::mutex_lock l(mu_);                                     \
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

  ::tensorflow::mutex mu_;
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
    //Schedule([this, call] {
      ::tensorflow::Status status = worker_->GetStatus(&call->request, &call->response);
      call->SendResponse(ToGrpcStatus(status));
    //});
    ENQUEUE_REQUEST(GetStatus, false);
  }

  void GetMachineDescHandler(WorkerCall<GetMachineDescRequest,
                                        GetMachineDescResponse>* call) {
    Schedule([this, call] {
      ::tensorflow::Status status 
        = worker_->GetMachineDesc(&call->request, &call->response);
      call->SendResponse(ToGrpcStatus(status));
    });
  }

  void GetMemoryDescHandler(WorkerCall<GetMemoryDescRequest, 
                                       GetMemoryDescResponse>* call) {
    Schedule([this, call] {
      ::tensorflow::Status status 
        = worker_->GetMemoryDesc(&call->request, &call->response);
      call->SendResponse(ToGrpcStatus(status));
    });
  }

  void SendTaskGraphHandler(WorkerCall<SendTaskGraphRequest,
                                       SendTaskGraphResponse>* call) {
    Schedule([this, call] {
      ::tensorflow::Status status 
        = worker_->SendTaskGraph(&call->request, &call->response);
      call->SendResponse(ToGrpcStatus(status));
    });
  }

  void SendMessageHandler(WorkerCall<SendMessageRequest,
                                     SendMessageResponse>* call) {
    Schedule([this, call] {
      ::tensorflow::Status status 
        = worker_->SendMessageAsync(&call->request, &call->response);
      call->SendResponse(ToGrpcStatus(status));
    });
  }

  void ReadDataHandleRaw(
      WorkerCall<ReadDataRequest, ::grpc::ByteBuffer>* call) {
    Schedule([this, call] {
      worker_->ReadDataAsync(&call->request, &call->response,
                             [this, call](const ::tensorflow::Status& status) {
                               call->SendResponse(ToGrpcStatus(status));
                             });
    });
    EnqueueReadDataRaw();
  }

#undef ENQUEUE_REQUEST
  void EnqueueReadDataRaw() {
    ::tensorflow::mutex_lock l(mu_);
    if (!is_shutdown_) {
      Call<GrpcWorkerService, grpc::WorkerService::AsyncService,
        ReadDataRequest, ::grpc::ByteBuffer>::
        EnqueueRequestForMethod(
          &worker_service_, cq_.get(),
          static_cast<int>(GrpcWorkerMethod::kReadData),
          &GrpcWorkerService::ReadDataHandleRaw, true);
    }
  }

};  // Grpcworkerservice

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_SERVER_SERVICE_H_
