#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

namespace {

const int32_t max_retry_num = 60;
const int64_t sleep_seconds = 10;

}  // namespace

void CtrlClient::Barrier(const std::string& barrier_name) {
  Barrier(barrier_name, JobDesc::Singleton()->TotalMachineNum());
}

void CtrlClient::Barrier(const std::string& barrier_name, int32_t barrier_num) {
  grpc::ClientContext client_ctx;
  BarrierRequest request;
  request.set_name(barrier_name);
  request.set_num(barrier_num);
  BarrierResponse response;
  GetMasterStub()->Barrier(&client_ctx, request, &response);
}

TryLockResult CtrlClient::TryLock(const std::string& name) {
  if (done_names_.find(name) != done_names_.end()) {
    return TryLockResult::kDone;
  }
  grpc::ClientContext client_ctx;
  TryLockRequest request;
  request.set_name(name);
  TryLockResponse response;
  GetResponsibleStub(name)->TryLock(&client_ctx, request, &response);
  if (response.result() == TryLockResult::kDone) {
    CHECK(done_names_.insert(name).second);
  }
  return response.result();
}

void CtrlClient::NotifyDone(const std::string& name) {
  grpc::ClientContext client_ctx;
  NotifyDoneRequest request;
  request.set_name(name);
  NotifyDoneResponse response;
  GetResponsibleStub(name)->NotifyDone(&client_ctx, request, &response);
}

void CtrlClient::WaitUntilDone(const std::string& name) {
  grpc::ClientContext client_ctx;
  WaitUntilDoneRequest request;
  request.set_name(name);
  WaitUntilDoneResponse response;
  GetResponsibleStub(name)->WaitUntilDone(&client_ctx, request, &response);
}

void CtrlClient::PushPlan(const Plan& plan) {
  grpc::ClientContext client_ctx;
  PushPlanRequest request;
  *(request.mutable_plan()) = plan;
  PushPlanResponse response;
  GetMasterStub()->PushPlan(&client_ctx, request, &response);
}

void CtrlClient::ClearPlan() {
  grpc::ClientContext client_ctx;
  ClearPlanRequest request;
  ClearPlanResponse response;
  GetMasterStub()->ClearPlan(&client_ctx, request, &response);
}

void CtrlClient::PullPlan(Plan* plan) {
  grpc::ClientContext client_ctx;
  PullPlanRequest request;
  PullPlanResponse response;
  GetMasterStub()->PullPlan(&client_ctx, request, &response);
  *plan = response.plan();
}

void CtrlClient::PushPort(uint16_t port) {
  grpc::ClientContext client_ctx;
  PushPortRequest request;
  request.set_port(port);
  PushPortResponse response;
  GetThisStub()->PushPort(&client_ctx, request, &response);
}

void CtrlClient::ClearPort() {
  grpc::ClientContext client_ctx;
  ClearPortRequest request;
  ClearPortResponse response;
  GetThisStub()->ClearPort(&client_ctx, request, &response);
}

uint16_t CtrlClient::PullPort(uint64_t machine_id) {
  grpc::ClientContext client_ctx;
  PullPortRequest request;
  PullPortResponse response;
  stubs_[machine_id]->PullPort(&client_ctx, request, &response);
  return response.port();
}

void CtrlClient::PushConnectionInfo(const ConnectionInfo& conn_info) {
  grpc::ClientContext client_ctx;
  PushConnectionInfoRequest request;
  *(request.mutable_conn_info()) = conn_info;
  PushConnectionInfoResponse response;
  GetThisStub()->PushConnectionInfo(&client_ctx, request, &response);
}

void CtrlClient::ClearConnectionInfo() {
  grpc::ClientContext client_ctx;
  ClearConnectionInfoRequest request;
  ClearConnectionInfoResponse response;
  GetThisStub()->ClearConnectionInfo(&client_ctx, request, &response);
}

ConnectionInfo& CtrlClient::PullConnectionInfo(uint64_t machine_id) {
  grpc::ClientContext client_ctx;
  PullConnectionInfoRequest request;
  PullConnectionInfoResponse response;
  stubs_[machine_id]->PullConnectionInfo(&client_ctx, request, &response);
  return *(response.mutable_conn_info());
}

void CtrlClient::PushTokenMsgs(const TokenMsgs& token_msgs) {
  grpc::ClientContext client_ctx;
  PushTokenMsgsRequest request;
  *(request.mutable_token_msgs()) = token_msgs;
  PushTokenMsgsResponse response;
  GetThisStub()->PushTokenMsgs(&client_ctx, request, &response);
}

void CtrlClient::ClearTokenMsgs() {
  grpc::ClientContext client_ctx;
  ClearTokenMsgsRequest request;
  ClearTokenMsgsResponse response;
  GetThisStub()->ClearTokenMsgs(&client_ctx, request, &response);
}

void CtrlClient::PullTokenMsgs(int64_t machine_id, TokenMsgs* token_msgs) {
  grpc::ClientContext client_ctx;
  PullTokenMsgsRequest request;
  PullTokenMsgsResponse response;
  stubs_[machine_id]->PullTokenMsgs(&client_ctx, request, &response);
  *token_msgs = response.token_msgs();
}

CtrlClient::CtrlClient() {
  stubs_.reserve(JobDesc::Singleton()->TotalMachineNum());
  for (int64_t i = 0; i < JobDesc::Singleton()->TotalMachineNum(); ++i) {
    std::string addr = RuntimeCtx::Singleton()->GetCtrlAddr(i);
    stubs_.push_back(CtrlService::NewStub(addr));
    LoadServer(addr, stubs_[i].get());
  }
}

void CtrlClient::LoadServer(const std::string& server_addr,
                            CtrlService::Stub* stub) {
  int32_t retry_idx = 0;
  for (; retry_idx < max_retry_num; ++retry_idx) {
    grpc::ClientContext client_ctx;
    LoadServerRequest request;
    LoadServerResponse response;
    grpc::Status st = stub->LoadServer(&client_ctx, request, &response);
    if (st.error_code() == grpc::StatusCode::OK) {
      LOG(INFO) << "LoadServer " << server_addr << " Successful at "
                << retry_idx << " times";
      break;
    } else if (st.error_code() == grpc::StatusCode::UNAVAILABLE) {
      LOG(INFO) << "LoadServer " << server_addr << " Failed at " << retry_idx
                << " times";
      std::this_thread::sleep_for(std::chrono::seconds(sleep_seconds));
      continue;
    } else {
      LOG(FATAL) << st.error_message();
    }
  }
  CHECK_LT(retry_idx, max_retry_num);
}

CtrlService::Stub* CtrlClient::GetThisStub() {
  return stubs_[RuntimeCtx::Singleton()->this_machine_id()].get();
}

CtrlService::Stub* CtrlClient::GetResponsibleStub(const std::string& key) {
  int64_t machine_id =
      (std::hash<std::string>{}(key)) % JobDesc::Singleton()->TotalMachineNum();
  return stubs_[machine_id].get();
}

}  // namespace oneflow
