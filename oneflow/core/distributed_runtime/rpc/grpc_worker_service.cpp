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

#include "oneflow/core/distributed_runtime/rpc/grpc_worker_service.h"

namespace oneflow {

void GrpcWorkerService::Shutdown() {
  bool did_shutdown = false;
  {
    ::tensorflow::mutex_lock l(mu_);
    if (!is_shutdown_) {
      LOG(INFO) << "Shutting down GrpcWorkerService.";
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
    cpu_stream_->CloseReceiveEnd();
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
      Call<GrpcWorkerService, grpc::WorkerService::AsyncService,       \
           method##Request, method##Response>::                        \
          EnqueueRequestForMethod(                                     \
              &worker_service_, cq_.get(),                             \
              static_cast<int>(GrpcWorkerMethod::k##method),           \
              &GrpcWorkerService::method##Handler, (supports_cancel)); \
    }                                                                  \
  } while (0)

void GrpcWorkerService::HandleRPCsLoop() {
  ENQUEUE_REQUEST(SendPlan, false);
  ENQUEUE_REQUEST(WorkerConnectDataPlane, false);
  ENQUEUE_REQUEST(WorkerInitRuntime, false);
  ENQUEUE_REQUEST(WorkerInitModel, false);
  ENQUEUE_REQUEST(WorkerActivateActor, false);
  ENQUEUE_REQUEST(WorkerSendRemoteRegst, false);
  ENQUEUE_REQUEST(WorkerSendRemoteRegstToConsumer, false);
  ENQUEUE_REQUEST(WorkerStartActor, false);
  ENQUEUE_REQUEST(WorkerInitDataPlane, false);

  void* tag;
  bool ok;
  while (cq_->Next(&tag, &ok)) {
    UntypedCall<GrpcWorkerService>::Tag* callback_tag =
        static_cast<UntypedCall<GrpcWorkerService>::Tag*>(tag);
    if (callback_tag) {
      callback_tag->OnCompleted(this, ok);
    } else {
      // NOTE(mrry): A null `callback_tag` indicates that this is
      // the shutdown alarm.
      cq_->Shutdown();
    }
  }
}

void GrpcWorkerService::DoWorkLoop() {
  std::function<void()> work;
  while (cpu_stream_->ReceiveWork(&work) == 0) { work(); }
}

// RPC handler for sending job.
void GrpcWorkerService::SendPlanHandler(
    WorkerCall<SendPlanRequest, SendPlanResponse>* call) {
  cpu_stream_->SendWork([this, call]() {
    worker_impl_->SendPlan(&call->request, &call->response,
                           [call](const ::tensorflow::Status& status) {
                             call->SendResponse(ToGrpcStatus(status));
                           });
  });
  ENQUEUE_REQUEST(SendPlan, true);
}

void GrpcWorkerService::WorkerConnectDataPlaneHandler(
    WorkerCall<WorkerConnectDataPlaneRequest, WorkerConnectDataPlaneResponse>*
        call) {
  cpu_stream_->SendWork([this, call]() {
    worker_impl_->WorkerConnectDataPlane(
        &call->request, &call->response,
        [call](const ::tensorflow::Status& status) {
          call->SendResponse(ToGrpcStatus(status));
        });
  });
  ENQUEUE_REQUEST(WorkerConnectDataPlane, true);
}

void GrpcWorkerService::WorkerInitRuntimeHandler(
    WorkerCall<WorkerInitRuntimeRequest, WorkerInitRuntimeResponse>* call) {
  cpu_stream_->SendWork([this, call]() {
    worker_impl_->WorkerInitRuntime(&call->request, &call->response,
                                    [call](const ::tensorflow::Status& status) {
                                      call->SendResponse(ToGrpcStatus(status));
                                    });
  });
  ENQUEUE_REQUEST(WorkerInitRuntime, true);
}

void GrpcWorkerService::WorkerInitModelHandler(
    WorkerCall<WorkerInitModelRequest, WorkerInitModelResponse>* call) {
  cpu_stream_->SendWork([this, call]() {
    worker_impl_->WorkerInitModel(&call->request, &call->response,
                                  [call](const ::tensorflow::Status& status) {
                                    call->SendResponse(ToGrpcStatus(status));
                                  });
  });
  ENQUEUE_REQUEST(WorkerInitModel, true);
}

void GrpcWorkerService::WorkerActivateActorHandler(
    WorkerCall<WorkerActivateActorRequest, WorkerActivateActorResponse>* call) {
  cpu_stream_->SendWork([this, call]() {
    worker_impl_->WorkerActivateActor(
        &call->request, &call->response,
        [call](const ::tensorflow::Status& status) {
          call->SendResponse(ToGrpcStatus(status));
        });
  });
  ENQUEUE_REQUEST(WorkerActivateActor, true);
}

void GrpcWorkerService::WorkerSendRemoteRegstHandler(
    WorkerCall<WorkerSendRemoteRegstRequest, WorkerSendRemoteRegstResponse>*
        call) {
  cpu_stream_->SendWork([this, call]() {
    worker_impl_->WorkerSendRemoteRegst(
        &call->request, &call->response,
        [call](const ::tensorflow::Status& status) {
          call->SendResponse(ToGrpcStatus(status));
        });
  });
  ENQUEUE_REQUEST(WorkerSendRemoteRegst, true);
}

void GrpcWorkerService::WorkerSendRemoteRegstToConsumerHandler(
    WorkerCall<WorkerSendRemoteRegstToConsumerRequest,
               WorkerSendRemoteRegstToConsumerResponse>* call) {
  cpu_stream_->SendWork([this, call]() {
    worker_impl_->WorkerSendRemoteRegstToConsumer(
        &call->request, &call->response,
        [call](const ::tensorflow::Status& status) {
          call->SendResponse(ToGrpcStatus(status));
        });
  });
  ENQUEUE_REQUEST(WorkerSendRemoteRegstToConsumer, true);
}

void GrpcWorkerService::WorkerStartActorHandler(
    WorkerCall<WorkerStartActorRequest, WorkerStartActorResponse>* call) {
  cpu_stream_->SendWork([this, call]() {
    worker_impl_->WorkerStartActor(&call->request, &call->response,
                                   [call](const ::tensorflow::Status& status) {
                                     call->SendResponse(ToGrpcStatus(status));
                                   });
  });
  ENQUEUE_REQUEST(WorkerStartActor, true);
}

void GrpcWorkerService::WorkerInitDataPlaneHandler(
    WorkerCall<WorkerInitDataPlaneRequest, WorkerInitDataPlaneResponse>* call) {
  cpu_stream_->SendWork([this, call]() {
    worker_impl_->WorkerInitDataPlane(
        &call->request, &call->response,
        [call](const ::tensorflow::Status& status) {
          call->SendResponse(ToGrpcStatus(status));
        });
  });
  ENQUEUE_REQUEST(WorkerInitDataPlane, true);
}

#undef ENQUEUE_REQUEST

AsyncServiceInterface* NewGrpcWorkerService(Worker* master,
                                            ::grpc::ServerBuilder* builder) {
  return new GrpcWorkerService(master, builder);
}

}  // namespace oneflow
