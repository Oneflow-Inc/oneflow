#include "oneflow/core/comm_network/ctrl_server.h"
#include "grpc++/alarm.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

#define ENQUEUE_REQUEST(method)                                              \
  do {                                                                       \
    auto call = new CtrlCall<method##Request, method##Response>();           \
    call->set_request_handler(                                               \
        std::bind(&CtrlServer::method##Handler, this, call));                \
    grpc_service_->RequestAsyncUnary(                                        \
        static_cast<int32_t>(CtrlMethod::k##method), call->mut_server_ctx(), \
        call->mut_request(), call->mut_responder(), cq_.get(), cq_.get(),    \
        call);                                                               \
  } while (0);

CtrlServer::~CtrlServer() {
  grpc::Alarm alarm(cq_.get(), gpr_now(GPR_CLOCK_MONOTONIC), nullptr);
  loop_thread_.join();
  grpc_server_->Shutdown();
  cq_->Shutdown();
}

CtrlServer::CtrlServer(const std::string& server_addr) {
  grpc::ServerBuilder server_builder;
  server_builder.AddListeningPort(server_addr,
                                  grpc::InsecureServerCredentials());
  grpc_service_.reset(new CtrlService::AsyncService);
  server_builder.RegisterService(grpc_service_.get());
  cq_ = server_builder.AddCompletionQueue();
  grpc_server_ = server_builder.BuildAndStart();
  LOG(INFO) << "Server listening on " << server_addr;
  added_worker_calls_.clear();
  loop_thread_ = std::thread(&CtrlServer::HandleRpcs, this);
}

void CtrlServer::HandleRpcs() {
  OF_PP_FOR_EACH_TUPLE(ENQUEUE_REQUEST, CTRL_METHOD_SEQ);

  void* tag = nullptr;
  bool ok = false;
  while (true) {
    CHECK(cq_->Next(&tag, &ok));
    CHECK(ok);
    auto call = static_cast<CtrlCallIf*>(tag);
    if (call) {
      call->Process();
    } else {
      break;
    }
  }
}

void CtrlServer::AddWorkerHandler(CtrlCallIf* call) {
  using AddWorkerCtrlCall = CtrlCall<AddWorkerRequest, AddWorkerResponse>;
  CHECK(RuntimeCtx::Singleton()->IsThisMachineMaster());
  added_worker_calls_.push_back(call);
  auto addworker_call = static_cast<AddWorkerCtrlCall*>(call);
  LOG(INFO) << "Add Worker " << addworker_call->request().worker_ctrl_addr();
  if (added_worker_calls_.size() == JobDesc::Singleton()->TotalMachineNum()) {
    for (CtrlCallIf* pending_call : added_worker_calls_) {
      pending_call->SendResponse();
    }
    added_worker_calls_.clear();
  }
  ENQUEUE_REQUEST(AddWorker);
}

void CtrlServer::BarrierHandler(CtrlCallIf* call) {
  using BarrierCtrlCall = CtrlCall<BarrierRequest, BarrierResponse>;
  auto barrier_call = static_cast<BarrierCtrlCall*>(call);
  const std::string& barrier_name = barrier_call->request().name();
  int32_t barrier_num = barrier_call->request().num();
  auto barrier_call_it = barrier_calls_.find(barrier_name);
  if (barrier_call_it == barrier_calls_.end()) {
    barrier_call_it =
        barrier_calls_
            .emplace(barrier_name,
                     std::make_pair(std::list<CtrlCallIf*>{}, barrier_num))
            .first;
  }
  CHECK_EQ(barrier_num, barrier_call_it->second.second);
  barrier_call_it->second.first.push_back(call);
  if (barrier_call_it->second.first.size() == barrier_call_it->second.second) {
    for (CtrlCallIf* pending_call : barrier_call_it->second.first) {
      pending_call->SendResponse();
    }
    barrier_calls_.erase(barrier_call_it);
  }
  ENQUEUE_REQUEST(Barrier);
}

}  // namespace oneflow
