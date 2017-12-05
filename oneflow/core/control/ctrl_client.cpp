#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"

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

void CtrlClient::PushKV(const std::string& k,
                        std::function<void(std::string*)> VSetter) {
  grpc::ClientContext client_ctx;
  PushKVRequest request;
  request.set_key(k);
  VSetter(request.mutable_val());
  PushKVResponse response;
  GetResponsibleStub(k)->PushKV(&client_ctx, request, &response);
}

void CtrlClient::PushKV(const std::string& k, const std::string& v) {
  PushKV(k, [&](std::string* o) { *o = v; });
}

void CtrlClient::PushKV(const std::string& k, const PbMessage& msg) {
  PushKV(k, [&](std::string* o) { msg.SerializeToString(o); });
}

void CtrlClient::ClearKV(const std::string& k) {
  grpc::ClientContext client_ctx;
  ClearKVRequest request;
  request.set_key(k);
  ClearKVResponse response;
  GetResponsibleStub(k)->ClearKV(&client_ctx, request, &response);
}

void CtrlClient::PullKV(const std::string& k,
                        std::function<void(const std::string&)> VGetter) {
  grpc::ClientContext client_ctx;
  PullKVRequest request;
  request.set_key(k);
  PullKVResponse response;
  GetResponsibleStub(k)->PullKV(&client_ctx, request, &response);
  VGetter(response.val());
}

void CtrlClient::PullKV(const std::string& k, std::string* v) {
  PullKV(k, [&](const std::string& i) { *v = i; });
}

void CtrlClient::PullKV(const std::string& k, PbMessage* msg) {
  PullKV(k, [&](const std::string& i) { msg->ParseFromString(i); });
}

CtrlClient::CtrlClient() {
  stubs_.reserve(JobDesc::Singleton()->TotalMachineNum());
  for (int64_t i = 0; i < JobDesc::Singleton()->TotalMachineNum(); ++i) {
    std::string addr = MachineCtx::Singleton()->GetCtrlAddr(i);
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
  return stubs_[MachineCtx::Singleton()->this_machine_id()].get();
}

CtrlService::Stub* CtrlClient::GetResponsibleStub(const std::string& key) {
  int64_t machine_id =
      (std::hash<std::string>{}(key)) % JobDesc::Singleton()->TotalMachineNum();
  return stubs_[machine_id].get();
}

}  // namespace oneflow
