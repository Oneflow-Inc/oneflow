#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"

namespace oneflow {

namespace {

const int32_t max_retry_num = 60;
const int64_t sleep_seconds = 10;

OF_DEFINE_ENUM_TO_OSTREAM_FUNC(grpc::StatusCode);

#define GRPC_CHECK(x) CHECK_EQ(x.error_code(), grpc::StatusCode::OK)

#define DEFINE_CLIENT_CALL(method)                                 \
  class method##ClientCall final {                                 \
   public:                                                         \
    OF_DISALLOW_COPY_AND_MOVE(method##ClientCall);                 \
    method##ClientCall() = default;                                \
    ~method##ClientCall() = default;                               \
    method##Request* mut_request() { return &request_; }           \
    const method##Response& response() const { return response_; } \
    void operator()(CtrlService::Stub* stub) {                     \
      grpc::ClientContext client_ctx;                              \
      GRPC_CHECK(stub->method(&client_ctx, request_, &response_)); \
    }                                                              \
                                                                   \
   private:                                                        \
    method##Request request_;                                      \
    method##Response response_;                                    \
  };

OF_PP_FOR_EACH_TUPLE(DEFINE_CLIENT_CALL, CTRL_METHOD_SEQ);

}  // namespace

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
  BarrierClientCall call;
  call.mut_request()->set_name(barrier_name);
  call.mut_request()->set_num(barrier_num);
  call(GetMasterStub());
}

TryLockResult CtrlClient::TryLock(const std::string& name) {
  {
    std::unique_lock<std::mutex> lck(done_names_mtx_);
    if (done_names_.find(name) != done_names_.end()) {
      return TryLockResult::kDone;
    }
  }
  TryLockClientCall call;
  call.mut_request()->set_name(name);
  call(GetResponsibleStub(name));
  if (call.response().result() == TryLockResult::kDone) {
    std::unique_lock<std::mutex> lck(done_names_mtx_);
    done_names_.insert(name);
  }
  return call.response().result();
}

void CtrlClient::NotifyDone(const std::string& name) {
  NotifyDoneClientCall call;
  call.mut_request()->set_name(name);
  call(GetResponsibleStub(name));
}

void CtrlClient::WaitUntilDone(const std::string& name) {
  WaitUntilDoneClientCall call;
  call.mut_request()->set_name(name);
  call(GetResponsibleStub(name));
}

void CtrlClient::PushKV(const std::string& k,
                        std::function<void(std::string*)> VSetter) {
  PushKVClientCall call;
  call.mut_request()->set_key(k);
  VSetter(call.mut_request()->mutable_val());
  call(GetResponsibleStub(k));
}

void CtrlClient::PushKV(const std::string& k, const std::string& v) {
  PushKV(k, [&](std::string* o) { *o = v; });
}

void CtrlClient::PushKV(const std::string& k, const PbMessage& msg) {
  PushKV(k, [&](std::string* o) { msg.SerializeToString(o); });
}

void CtrlClient::ClearKV(const std::string& k) {
  ClearKVClientCall call;
  call.mut_request()->set_key(k);
  call(GetResponsibleStub(k));
}

void CtrlClient::PullKV(const std::string& k,
                        std::function<void(const std::string&)> VGetter) {
  PullKVClientCall call;
  call.mut_request()->set_key(k);
  call(GetResponsibleStub(k));
  VGetter(call.response().val());
}

void CtrlClient::PullKV(const std::string& k, std::string* v) {
  PullKV(k, [&](const std::string& i) { *v = i; });
}

void CtrlClient::PullKV(const std::string& k, PbMessage* msg) {
  PullKV(k, [&](const std::string& i) { msg->ParseFromString(i); });
}

void CtrlClient::PushActEvent(const ActEvent& act_event) {
  PushActEventClientCall call;
  *(call.mut_request()->mutable_act_event()) = act_event;
  call(GetMasterStub());
}

void CtrlClient::Clear() {
  ClearClientCall call;
  call(GetThisStub());
  std::unique_lock<std::mutex> lck(done_names_mtx_);
  done_names_.clear();
}

int32_t CtrlClient::IncreaseCount(const std::string& k, int32_t v) {
  IncreaseCountClientCall call;
  call.mut_request()->set_key(k);
  call.mut_request()->set_val(v);
  call(GetResponsibleStub(k));
  return call.response().val();
}

void CtrlClient::EraseCount(const std::string& k) {
  EraseCountClientCall call;
  call.mut_request()->set_key(k);
  call(GetResponsibleStub(k));
}

void CtrlClient::PushAvgActInterval(int64_t actor_id, double avg_act_interval) {
  PushAvgActIntervalClientCall call;
  call.mut_request()->set_actor_id(actor_id);
  call.mut_request()->set_avg_act_interval(avg_act_interval);
  call(GetMasterStub());
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
