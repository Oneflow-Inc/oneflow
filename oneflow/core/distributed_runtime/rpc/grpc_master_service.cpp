/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// GrpcMasterService implements the RPC service MasterSerivce.
//
// A GrpcMasterService maintains the state of live graph computation
// sessions, each session orchestrates both local and remote devices
// to carry out the graph computation.
//
// A GrpcMasterService knows ahead of time local devices available as
// client devices.
//
// A GrpcMasterService discovers remote devices in the background and
// keeps track of statistics of those remote devices.
//
// Each session analyzes the graph, places nodes across available
// devices, and ultimately drives the graph computation by initiating
// RunGraph on workers.
#include "oneflow/core/distributed_runtime/rpc/grpc_master_service.h"

#include "grpc++/alarm.h"
#include "grpc++/server_builder.h"

#include "oneflow/core/distributed_runtime/master.h"
#include "oneflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_call.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_util.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace oneflow {

class GrpcMasterService : public AsyncServiceInterface {
 public:
  GrpcMasterService(Master* master, ::grpc::ServerBuilder* builder)
      : master_impl_(master), is_shutdown_(false) {
    builder->RegisterService(&master_service_);
    cq_ = builder->AddCompletionQueue();
  }

  ~GrpcMasterService() override { delete shutdown_alarm_; }

  void Shutdown() override {
    bool did_shutdown = false;
    {
      ::tensorflow::mutex_lock l(mu_);
      if (!is_shutdown_) {
        LOG(INFO) << "Shutting down GrpcMasterService.";
        is_shutdown_ = true;
        did_shutdown = true;
      }
    }
    if (did_shutdown) {
      // NOTE(mrry): This enqueues a special event (with a null tag)
      // that causes the completion queue to be shut down on the
      // polling thread.
      shutdown_alarm_ =
          new ::grpc::Alarm(cq_.get(), gpr_now(GPR_CLOCK_MONOTONIC), nullptr);
    }
  }

// This macro creates a new request for the given RPC method name
// (e.g., `ENQUEUE_REQUEST(GetStatus, false);`), and enqueues it on
// `this->cq_`.
//
// This macro is invoked one or more times for each RPC method to
// ensure that there are sufficient completion queue entries to
// handle incoming requests without blocking.
//
// The implementation of the request handler for each RPC method
// must ensure that it calls ENQUEUE_REQUEST() for that RPC method,
// to keep accepting new requests.
#define ENQUEUE_REQUEST(method, supports_cancel)                       \
  do {                                                                 \
    ::tensorflow::mutex_lock l(mu_);                                   \
    if (!is_shutdown_) {                                               \
      Call<GrpcMasterService, grpc::MasterService::AsyncService,       \
           method##Request, method##Response>::                        \
          EnqueueRequestForMethod(                                     \
              &master_service_, cq_.get(),                             \
              static_cast<int>(GrpcMasterMethod::k##method),           \
              &GrpcMasterService::method##Handler, (supports_cancel)); \
    }                                                                  \
  } while (0)

  void HandleRPCsLoop() override {
    ENQUEUE_REQUEST(SendJob, false);

    void* tag;
    bool ok;
    while (cq_->Next(&tag, &ok)) {
      UntypedCall<GrpcMasterService>::Tag* callback_tag =
          static_cast<UntypedCall<GrpcMasterService>::Tag*>(tag);
      if (callback_tag) {
        callback_tag->OnCompleted(this, ok);
      } else {
        // NOTE(mrry): A null `callback_tag` indicates that this is
        // the shutdown alarm.
        cq_->Shutdown();
      }
    }
  }

 private:
  Master* master_impl_ = nullptr;  // Not owned.
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
  grpc::MasterService::AsyncService master_service_;

  tensorflow::mutex mu_;
  bool is_shutdown_ GUARDED_BY(mu_);
  ::grpc::Alarm* shutdown_alarm_ = nullptr;

  template<class RequestMessage, class ResponseMessage>
  using MasterCall = Call<GrpcMasterService, grpc::MasterService::AsyncService,
                          RequestMessage, ResponseMessage>;

  // RPC handler for sending job.
  void SendJobHandler(MasterCall<SendJobRequest, SendJobResponse>* call) {
    master_impl_->SendJob(&call->request, &call->response,
                          [call](const ::tensorflow::Status& status) {
                            call->SendResponse(ToGrpcStatus(status));
                          });
    ENQUEUE_REQUEST(SendJob, true);
  }

#undef ENQUEUE_REQUEST

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcMasterService);
};

AsyncServiceInterface* NewGrpcMasterService(Master* master,
                                            ::grpc::ServerBuilder* builder) {
  return new GrpcMasterService(master, builder);
}

}  // namespace oneflow
