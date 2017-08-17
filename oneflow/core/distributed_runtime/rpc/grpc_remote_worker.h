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
#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_REMOTE_WORKER_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_REMOTE_WORKER_H_

#include <memory>
#include "grpc++/grpc++.h"
#include "oneflow/core/distributed_runtime/call_options.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_util.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h"
#include "oneflow/core/distributed_runtime/worker.pb.h"
#include "oneflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/lib/core/status.h"

namespace oneflow {

class GrpcRemoteWorker : public WorkerInterface {
 public:
  explicit GrpcRemoteWorker(
      const std::shared_ptr<::grpc::Channel>& client_channel,
      ::grpc::CompletionQueue* completion_queue)
      : stub_(grpc::WorkerService::NewStub(client_channel)),
        channel_(client_channel),
        cq_(completion_queue),
        sendplan_(Method(GrpcWorkerMethod::kSendPlan)),
        worker_connect_data_plane_(
            Method(GrpcWorkerMethod::kWorkerConnectDataPlane)),
        worker_init_runtime_(Method(GrpcWorkerMethod::kWorkerInitRuntime)),
        worker_init_model_(Method(GrpcWorkerMethod::kWorkerInitModel)),
        worker_activate_actor_(Method(GrpcWorkerMethod::kWorkerActivateActor)),
        worker_send_remote_regst_to_inc_(
            Method(GrpcWorkerMethod::kWorkerSendRemoteRegstToInc)),
        worker_send_remote_regst_to_dec_(
            Method(GrpcWorkerMethod::kWorkerSendRemoteRegstToDec)),
        worker_start_actor_(Method(GrpcWorkerMethod::kWorkerStartActor)),
        worker_init_data_plane_(
            Method(GrpcWorkerMethod::kWorkerInitDataPlane)) {}

  ~GrpcRemoteWorker() {}

  ::tensorflow::Status SendPlan(const SendPlanRequest* request,
                                SendPlanResponse* response) override;
  void SendPlanAsync(const SendPlanRequest* request, SendPlanResponse* response,
                     ::tensorflow::StatusCallback done) override;

  ::tensorflow::Status WorkerConnectDataPlane(
      const WorkerConnectDataPlaneRequest* request,
      WorkerConnectDataPlaneResponse* response) override;

  void WorkerConnectDataPlaneAsync(const WorkerConnectDataPlaneRequest* request,
                                   WorkerConnectDataPlaneResponse* response,
                                   ::tensorflow::StatusCallback done) override;

  void WorkerInitRuntimeAsync(const WorkerInitRuntimeRequest* request,
                              WorkerInitRuntimeResponse* response,
                              ::tensorflow::StatusCallback done) override;

  void WorkerInitModelAsync(const WorkerInitModelRequest* request,
                            WorkerInitModelResponse* response,
                            ::tensorflow::StatusCallback done) override;

  void WorkerActivateActorAsync(const WorkerActivateActorRequest* request,
                                WorkerActivateActorResponse* response,
                                ::tensorflow::StatusCallback done) override;

  void WorkerSendRemoteRegstToIncAsync(
      const WorkerSendRemoteRegstToIncRequest* request,
      WorkerSendRemoteRegstToIncResponse* response,
      ::tensorflow::StatusCallback done) override;

  void WorkerSendRemoteRegstToDecAsync(
      const WorkerSendRemoteRegstToDecRequest* request,
      WorkerSendRemoteRegstToDecResponse* response,
      ::tensorflow::StatusCallback done) override;

  void WorkerStartActorAsync(const WorkerStartActorRequest* request,
                             WorkerStartActorResponse* response,
                             ::tensorflow::StatusCallback done) override;

  ::tensorflow::Status WorkerInitDataPlane(
      const WorkerInitDataPlaneRequest* request,
      WorkerInitDataPlaneResponse* response) override;

  void WorkerInitDataPlaneAsync(const WorkerInitDataPlaneRequest* request,
                                WorkerInitDataPlaneResponse* response,
                                ::tensorflow::StatusCallback done) override;

 private:
  std::unique_ptr<grpc::WorkerService::Stub> stub_;

  // Object allocated per active RPC.
  template<class RequestMessage, class ResponseMessage>
  class RPCState final : public GrpcClientCQTag {
   public:
    RPCState(::grpc::ChannelInterface* channel, ::grpc::CompletionQueue* cq,
             const ::grpc::RpcMethod& method, const RequestMessage& request,
             ::tensorflow::StatusCallback done, CallOptions* call_opts)
        : call_opts_(call_opts),
          reader_(channel, cq, method, InitContext(call_opts), request),
          done_(std::move(done)) {}

    ~RPCState() override {}

    void StartRPC(ResponseMessage* response) {
      reader_.Finish(response, &status_, this);
    }

    void OnCompleted(bool ok) override {
      if (!ok) {
        VLOG(2) << "Call returned with non-ok status: "
                << status_.error_message();
      }
      if (call_opts_) { call_opts_->ClearCancelCallback(); }
      done_(FromGrpcStatus(status_));
      delete this;
    }

   private:
    CallOptions* call_opts_;
    ::grpc::ClientContext context_;
    ::grpc::ClientAsyncResponseReader<ResponseMessage> reader_;
    ::grpc::Status status_;
    ::tensorflow::StatusCallback done_;

    ::grpc::ClientContext* InitContext(CallOptions* call_opts) {
      // The initialization and recovery protocols rely on blocking
      // until we get a response.
      context_.set_fail_fast(false);
      if (call_opts) {
        call_opts->SetCancelCallback([this]() { context_.TryCancel(); });
      }
      return &context_;
    }
  };

  // Utility method for issuing a generic asynchronous request. The
  // given callback, `done`, will be called when the RPC completes.
  template<class RequestMessage, class ResponseMessage>
  void IssueRequest(const RequestMessage* request, ResponseMessage* response,
                    const ::grpc::RpcMethod& method,
                    ::tensorflow::StatusCallback done,
                    CallOptions* call_opts = nullptr) {
    auto state = new RPCState<RequestMessage, ResponseMessage>(
        channel_.get(), cq_, method, *request, std::move(done), call_opts);
    state->StartRPC(response);
  }

  // Helper function for initializing the RpcMethod objects below.
  ::grpc::RpcMethod Method(GrpcWorkerMethod id) {
    return ::grpc::RpcMethod(GrpcWorkerMethodName(id),
                             ::grpc::RpcMethod::NORMAL_RPC, channel_);
  }

  std::shared_ptr<::grpc::Channel> channel_;
  ::grpc::CompletionQueue* cq_;

  const ::grpc::RpcMethod sendplan_;
  const ::grpc::RpcMethod worker_connect_data_plane_;
  const ::grpc::RpcMethod worker_init_runtime_;
  const ::grpc::RpcMethod worker_init_model_;
  const ::grpc::RpcMethod worker_activate_actor_;
  const ::grpc::RpcMethod worker_send_remote_regst_to_inc_;
  const ::grpc::RpcMethod worker_send_remote_regst_to_dec_;
  const ::grpc::RpcMethod worker_start_actor_;
  const ::grpc::RpcMethod worker_init_data_plane_;
};  // GrpcRemoteWorker

}  // namespace oneflow
#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_REMOTE_WORKER_H_
