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

#include "oneflow/core/distributed_runtime/rpc/grpc_master_service.h"

namespace oneflow {

void GrpcMasterService::Shutdown() {
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
      Call<GrpcMasterService, grpc::MasterService::AsyncService,       \
           method##Request, method##Response>::                        \
          EnqueueRequestForMethod(                                     \
              &master_service_, cq_.get(),                             \
              static_cast<int>(GrpcMasterMethod::k##method),           \
              &GrpcMasterService::method##Handler, (supports_cancel)); \
    }                                                                  \
  } while (0)

void GrpcMasterService::HandleRPCsLoop() {
  ENQUEUE_REQUEST(SendJob, false);
  ENQUEUE_REQUEST(MasterConnectDataPlane, false);
  ENQUEUE_REQUEST(MasterInitRuntime, false);
  ENQUEUE_REQUEST(MasterInitModel, false);
  ENQUEUE_REQUEST(MasterActivateActor, false);
  ENQUEUE_REQUEST(MasterSendRemoteRegst, false);
  ENQUEUE_REQUEST(MasterStartActor, false);
  ENQUEUE_REQUEST(MasterInitDataPlane, false);

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

void GrpcMasterService::DoWorkLoop() {
  std::function<void()> work;
  while (cpu_stream_->ReceiveWork(&work) == 0) { work(); }
}

// RPC handler for sending job.
void GrpcMasterService::SendJobHandler(
    MasterCall<SendJobRequest, SendJobResponse>* call) {
  cpu_stream_->SendWork([this, call]() {
    master_impl_->SendJob(&call->request, &call->response,
                          [call](const ::tensorflow::Status& status) {
                            call->SendResponse(ToGrpcStatus(status));
                          });
  });
  ENQUEUE_REQUEST(SendJob, true);
}

void GrpcMasterService::MasterConnectDataPlaneHandler(
    MasterCall<MasterConnectDataPlaneRequest, MasterConnectDataPlaneResponse>*
        call) {
  cpu_stream_->SendWork([this, call]() {
    master_impl_->MasterConnectDataPlane(
        &call->request, &call->response,
        [call](const ::tensorflow::Status& status) {
          call->SendResponse(ToGrpcStatus(status));
        });
  });
  ENQUEUE_REQUEST(MasterConnectDataPlane, true);
}

void GrpcMasterService::MasterInitRuntimeHandler(
    MasterCall<MasterInitRuntimeRequest, MasterInitRuntimeResponse>* call) {
  cpu_stream_->SendWork([this, call]() {
    master_impl_->MasterInitRuntime(&call->request, &call->response,
                                    [call](const ::tensorflow::Status& status) {
                                      call->SendResponse(ToGrpcStatus(status));
                                    });
  });
  ENQUEUE_REQUEST(MasterInitRuntime, true);
}

void GrpcMasterService::MasterInitModelHandler(
    MasterCall<MasterInitModelRequest, MasterInitModelResponse>* call) {
  cpu_stream_->SendWork([this, call]() {
    master_impl_->MasterInitModel(&call->request, &call->response,
                                  [call](const ::tensorflow::Status& status) {
                                    call->SendResponse(ToGrpcStatus(status));
                                  });
  });
  ENQUEUE_REQUEST(MasterInitModel, true);
}

void GrpcMasterService::MasterActivateActorHandler(
    MasterCall<MasterActivateActorRequest, MasterActivateActorResponse>* call) {
  cpu_stream_->SendWork([this, call]() {
    master_impl_->MasterActivateActor(
        &call->request, &call->response,
        [call](const ::tensorflow::Status& status) {
          call->SendResponse(ToGrpcStatus(status));
        });
  });
  ENQUEUE_REQUEST(MasterActivateActor, true);
}

void GrpcMasterService::MasterSendRemoteRegstHandler(
    MasterCall<MasterSendRemoteRegstRequest, MasterSendRemoteRegstResponse>*
        call) {
  cpu_stream_->SendWork([this, call]() {
    master_impl_->MasterSendRemoteRegst(
        &call->request, &call->response,
        [call](const ::tensorflow::Status& status) {
          call->SendResponse(ToGrpcStatus(status));
        });
  });
  ENQUEUE_REQUEST(MasterSendRemoteRegst, true);
}

void GrpcMasterService::MasterStartActorHandler(
    MasterCall<MasterStartActorRequest, MasterStartActorResponse>* call) {
  cpu_stream_->SendWork([this, call]() {
    master_impl_->MasterStartActor(&call->request, &call->response,
                                   [call](const ::tensorflow::Status& status) {
                                     call->SendResponse(ToGrpcStatus(status));
                                   });
  });
  ENQUEUE_REQUEST(MasterStartActor, true);
}

void GrpcMasterService::MasterInitDataPlaneHandler(
    MasterCall<MasterInitDataPlaneRequest, MasterInitDataPlaneResponse>* call) {
  cpu_stream_->SendWork([this, call]() {
    master_impl_->MasterInitDataPlane(
        &call->request, &call->response,
        [call](const ::tensorflow::Status& status) {
          call->SendResponse(ToGrpcStatus(status));
        });
  });
  ENQUEUE_REQUEST(MasterInitDataPlane, true);
}

#undef ENQUEUE_REQUEST

AsyncServiceInterface* NewGrpcMasterService(Master* master,
                                            ::grpc::ServerBuilder* builder) {
  return new GrpcMasterService(master, builder);
}

}  // namespace oneflow
