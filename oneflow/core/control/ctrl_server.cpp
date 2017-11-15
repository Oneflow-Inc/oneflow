#include "oneflow/core/control/ctrl_server.h"

namespace oneflow {

namespace {

int ExtractPortFromAddr(const std::string& addr) {
  size_t pos = addr.find(':');
  return oneflow_cast<int>(addr.substr(pos + 1));
}

}  // namespace

CtrlServer::~CtrlServer() {
  grpc::Alarm alarm(cq_.get(), gpr_now(GPR_CLOCK_MONOTONIC), nullptr);
  loop_thread_.join();
  grpc_server_->Shutdown();
  cq_->Shutdown();
}

CtrlServer::CtrlServer(const std::string& server_addr) {
  int port = ExtractPortFromAddr(server_addr);
  grpc::ServerBuilder server_builder;
  int bound_port = 0;
  server_builder.AddListeningPort(
      server_addr, grpc::InsecureServerCredentials(), &bound_port);
  grpc_service_.reset(new CtrlService::AsyncService);
  server_builder.RegisterService(grpc_service_.get());
  cq_ = server_builder.AddCompletionQueue();
  grpc_server_ = server_builder.BuildAndStart();
  CHECK_EQ(port, bound_port) << "Port " << port << " is unavailable";
  LOG(INFO) << "CtrlServer listening on " << server_addr;
  plan_ = nullptr;
  port_ = -1;
  loop_thread_ = std::thread(&CtrlServer::HandleRpcs, this);
}

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

void CtrlServer::LoadServerHandler(
    CtrlCall<LoadServerRequest, LoadServerResponse>* call) {
  call->SendResponse();
  ENQUEUE_REQUEST(LoadServer);
}

void CtrlServer::BarrierHandler(
    CtrlCall<BarrierRequest, BarrierResponse>* call) {
  const std::string& barrier_name = call->request().name();
  int32_t barrier_num = call->request().num();
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

void CtrlServer::TryLockHandler(
    CtrlCall<TryLockRequest, TryLockResponse>* call) {
  const std::string& lock_name = call->request().name();
  auto name2lock_status_it = name2lock_status_.find(lock_name);
  if (name2lock_status_it == name2lock_status_.end()) {
    call->mut_response()->set_result(TryLockResult::kLocked);
    auto waiting_until_done_calls = new std::list<CtrlCallIf*>;
    CHECK(
        name2lock_status_.emplace(lock_name, waiting_until_done_calls).second);
  } else {
    if (name2lock_status_it->second) {
      call->mut_response()->set_result(TryLockResult::kDoing);
    } else {
      call->mut_response()->set_result(TryLockResult::kDone);
    }
  }
  call->SendResponse();
  ENQUEUE_REQUEST(TryLock);
}

void CtrlServer::NotifyDoneHandler(
    CtrlCall<NotifyDoneRequest, NotifyDoneResponse>* call) {
  const std::string& lock_name = call->request().name();
  auto name2lock_status_it = name2lock_status_.find(lock_name);
  auto waiting_calls =
      static_cast<std::list<CtrlCallIf*>*>(name2lock_status_it->second);
  for (CtrlCallIf* waiting_call : *waiting_calls) {
    waiting_call->SendResponse();
  }
  delete waiting_calls;
  name2lock_status_it->second = nullptr;
  call->SendResponse();
  ENQUEUE_REQUEST(NotifyDone);
}

void CtrlServer::WaitUntilDoneHandler(
    CtrlCall<WaitUntilDoneRequest, WaitUntilDoneResponse>* call) {
  const std::string& lock_name = call->request().name();
  void* lock_status = name2lock_status_.at(lock_name);
  if (lock_status) {
    auto waiting_calls = static_cast<std::list<CtrlCallIf*>*>(lock_status);
    waiting_calls->push_back(call);
  } else {
    call->SendResponse();
  }
  ENQUEUE_REQUEST(WaitUntilDone);
}

void CtrlServer::PushPlanHandler(
    CtrlCall<PushPlanRequest, PushPlanResponse>* call) {
  plan_.reset(new Plan(call->request().plan()));
  for (auto pending_call : pending_plan_calls_) {
    *(pending_call->mut_response()->mutable_plan()) = *plan_;
    pending_call->SendResponse();
  }
  call->SendResponse();
  ENQUEUE_REQUEST(PushPlan);
}

void CtrlServer::ClearPlanHandler(
    CtrlCall<ClearPlanRequest, ClearPlanResponse>* call) {
  plan_.reset();
  call->SendResponse();
  ENQUEUE_REQUEST(ClearPlan);
}

void CtrlServer::PullPlanHandler(
    CtrlCall<PullPlanRequest, PullPlanResponse>* call) {
  if (plan_) {
    *(call->mut_response()->mutable_plan()) = *plan_;
    call->SendResponse();
  } else {
    pending_plan_calls_.push_back(call);
  }
  ENQUEUE_REQUEST(PullPlan);
}

void CtrlServer::PushPortHandler(
    CtrlCall<PushPortRequest, PushPortResponse>* call) {
  port_ = call->request().port();
  for (auto pending_call : pending_port_calls_) {
    pending_call->mut_response()->set_port(port_);
    pending_call->SendResponse();
  }
  call->SendResponse();
  ENQUEUE_REQUEST(PushPort);
}

void CtrlServer::ClearPortHandler(
    CtrlCall<ClearPortRequest, ClearPortResponse>* call) {
  port_ = -1;
  call->SendResponse();
  ENQUEUE_REQUEST(ClearPort);
}

void CtrlServer::PullPortHandler(
    CtrlCall<PullPortRequest, PullPortResponse>* call) {
  if (port_ != -1) {
    call->mut_response()->set_port(port_);
    call->SendResponse();
  } else {
    pending_port_calls_.push_back(call);
  }
  ENQUEUE_REQUEST(PullPort);
}

}  // namespace oneflow
