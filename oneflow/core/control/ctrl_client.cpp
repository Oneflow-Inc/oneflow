#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"

namespace oneflow {

namespace {

const int32_t max_retry_num = 60;
const int64_t sleep_seconds = 10;

}  // namespace

OF_DEFINE_ENUM_TO_OSTREAM_FUNC(grpc::StatusCode);

#define GRPC_CHECK(x) CHECK_EQ(x.error_code(), grpc::StatusCode::OK)

CtrlClient::~CtrlClient() {
  {
    std::unique_lock<std::mutex> lck(need_heartbeat_thread_stop_mtx_);
    need_heartbeat_thread_stop_ = true;
  }
  heartbeat_thread_.join();
  OF_BARRIER();
}

void CtrlClient::Barrier(const std::string& barrier_name) {
  Barrier(barrier_name, Global<JobDesc>::Get()->TotalMachineNum());
}

void CtrlClient::Barrier(const std::string& barrier_name, int32_t barrier_num) {
  grpc::ClientContext client_ctx;
  BarrierRequest request;
  request.set_name(barrier_name);
  request.set_num(barrier_num);
  BarrierResponse response;
  GRPC_CHECK(GetMasterStub()->Barrier(&client_ctx, request, &response));
}

TryLockResult CtrlClient::TryLock(const std::string& name) {
  {
    std::unique_lock<std::mutex> lck(done_names_mtx_);
    if (done_names_.find(name) != done_names_.end()) {
      return TryLockResult::kDone;
    }
  }
  grpc::ClientContext client_ctx;
  TryLockRequest request;
  request.set_name(name);
  TryLockResponse response;
  GRPC_CHECK(
      GetResponsibleStub(name)->TryLock(&client_ctx, request, &response));
  if (response.result() == TryLockResult::kDone) {
    std::unique_lock<std::mutex> lck(done_names_mtx_);
    done_names_.insert(name);
  }
  return response.result();
}

void CtrlClient::NotifyDone(const std::string& name) {
  grpc::ClientContext client_ctx;
  NotifyDoneRequest request;
  request.set_name(name);
  NotifyDoneResponse response;
  GRPC_CHECK(
      GetResponsibleStub(name)->NotifyDone(&client_ctx, request, &response));
}

void CtrlClient::WaitUntilDone(const std::string& name) {
  grpc::ClientContext client_ctx;
  WaitUntilDoneRequest request;
  request.set_name(name);
  WaitUntilDoneResponse response;
  GRPC_CHECK(
      GetResponsibleStub(name)->WaitUntilDone(&client_ctx, request, &response));
}

void CtrlClient::PushKV(const std::string& k,
                        std::function<void(std::string*)> VSetter) {
  grpc::ClientContext client_ctx;
  PushKVRequest request;
  request.set_key(k);
  VSetter(request.mutable_val());
  PushKVResponse response;
  GRPC_CHECK(GetResponsibleStub(k)->PushKV(&client_ctx, request, &response));
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
  GRPC_CHECK(GetResponsibleStub(k)->ClearKV(&client_ctx, request, &response));
}

void CtrlClient::PullKV(const std::string& k,
                        std::function<void(const std::string&)> VGetter) {
  grpc::ClientContext client_ctx;
  PullKVRequest request;
  request.set_key(k);
  PullKVResponse response;
  GRPC_CHECK(GetResponsibleStub(k)->PullKV(&client_ctx, request, &response));
  VGetter(response.val());
}

void CtrlClient::PullKV(const std::string& k, std::string* v) {
  PullKV(k, [&](const std::string& i) { *v = i; });
}

void CtrlClient::PullKV(const std::string& k, PbMessage* msg) {
  PullKV(k, [&](const std::string& i) { msg->ParseFromString(i); });
}

void CtrlClient::PushActEvent(const ActEvent& act_event) {
  grpc::ClientContext client_ctx;
  PushActEventRequest request;
  *(request.mutable_act_event()) = act_event;
  PushActEventResponse response;
  GRPC_CHECK(GetMasterStub()->PushActEvent(&client_ctx, request, &response));
}

void CtrlClient::Clear() {
  grpc::ClientContext client_ctx;
  ClearRequest request;
  ClearResponse response;
  GRPC_CHECK(GetThisStub()->Clear(&client_ctx, request, &response));
  std::unique_lock<std::mutex> lck(done_names_mtx_);
  done_names_.clear();
}

int32_t CtrlClient::IncreaseCount(const std::string& k, int32_t v) {
  grpc::ClientContext client_ctx;
  IncreaseCountRequest request;
  request.set_key(k);
  request.set_val(v);
  IncreaseCountResponse response;
  GRPC_CHECK(
      GetResponsibleStub(k)->IncreaseCount(&client_ctx, request, &response));
  return response.val();
}

void CtrlClient::EraseCount(const std::string& k) {
  grpc::ClientContext client_ctx;
  EraseCountRequest request;
  request.set_key(k);
  EraseCountResponse response;
  GRPC_CHECK(
      GetResponsibleStub(k)->EraseCount(&client_ctx, request, &response));
}

CtrlClient::CtrlClient() {
  stubs_.reserve(Global<JobDesc>::Get()->TotalMachineNum());
  for (int64_t i = 0; i < Global<JobDesc>::Get()->TotalMachineNum(); ++i) {
    std::string addr = Global<MachineCtx>::Get()->GetCtrlAddr(i);
    stubs_.push_back(CtrlService::NewStub(addr));
    LoadServer(addr, stubs_[i].get());
  }
  need_heartbeat_thread_stop_ = false;
  heartbeat_thread_ = std::thread([this]() {
    std::mt19937 gen(NewRandomSeed());
    std::uniform_int_distribution<int32_t> sleep_second_dis(7, 13);
    LoadServerRequest request;
    LoadServerResponse response;
    while (true) {
      {
        std::unique_lock<std::mutex> lck(need_heartbeat_thread_stop_mtx_);
        if (need_heartbeat_thread_stop_) { break; }
      }
      for (size_t i = 0; i < stubs_.size(); ++i) {
        grpc::ClientContext client_ctx;
        GRPC_CHECK(stubs_[i]->LoadServer(&client_ctx, request, &response))
            << "Machine " << i << " lost";
      }
      std::this_thread::sleep_for(std::chrono::seconds(sleep_second_dis(gen)));
    }
  });
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
  return stubs_[Global<MachineCtx>::Get()->this_machine_id()].get();
}

CtrlService::Stub* CtrlClient::GetResponsibleStub(const std::string& key) {
  int64_t machine_id = (std::hash<std::string>{}(key))
                       % Global<JobDesc>::Get()->TotalMachineNum();
  return stubs_[machine_id].get();
}

}  // namespace oneflow
