#include "oneflow/core/comm_network/ctrl_comm_network.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

namespace {

const int32_t max_retry_num = 60;
const int64_t sleep_seconds = 10;

}  // namespace

CtrlCommNet::CtrlCommNet(uint16_t port) {
  ctrl_server_.reset(new CtrlServer(RuntimeCtx::Singleton()->GetThisAddr() + ":"
                                    + std::to_string(port)));
  stubs_.reserve(JobDesc::Singleton()->TotalMachineNum());
  for (int64_t i = 0; i < JobDesc::Singleton()->TotalMachineNum(); ++i) {
    stubs_.push_back(CtrlService::NewStub(RuntimeCtx::Singleton()->GetAddr(i)
                                          + ":" + std::to_string(port)));
  }
  int32_t retry_idx = 0;
  for (; retry_idx < max_retry_num; ++retry_idx) {
    grpc::ClientContext client_ctx;
    AddWorkerRequest request;
    request.set_worker_addr(RuntimeCtx::Singleton()->GetThisAddr());
    AddWorkerResponse response;
    grpc::Status st =
        GetMasterStub()->AddWorker(&client_ctx, request, &response);
    if (st.error_code() == grpc::StatusCode::OK) {
      LOG(INFO) << "AddWorker Successful at " << retry_idx << " times";
      break;
    } else if (st.error_code() == grpc::StatusCode::UNAVAILABLE) {
      LOG(INFO) << "AddWorker Failed at " << retry_idx << " times";
      std::this_thread::sleep_for(std::chrono::seconds(sleep_seconds));
      continue;
    } else {
      LOG(FATAL) << st.error_message();
    }
  }
  CHECK_LT(retry_idx, max_retry_num);
}

void CtrlCommNet::Barrier(const std::string& barrier_name) {
  Barrier(barrier_name, JobDesc::Singleton()->TotalMachineNum());
}

void CtrlCommNet::Barrier(const std::string& barrier_name,
                          int32_t barrier_num) {
  grpc::ClientContext client_ctx;
  BarrierRequest request;
  request.set_name(barrier_name);
  request.set_num(barrier_num);
  BarrierResponse response;
  GetMasterStub()->Barrier(&client_ctx, request, &response);
}

TryLockResult CtrlCommNet::TryLock(const std::string& name) {
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

void CtrlCommNet::NotifyDone(const std::string& name) {
  grpc::ClientContext client_ctx;
  NotifyDoneRequest request;
  request.set_name(name);
  NotifyDoneResponse response;
  GetResponsibleStub(name)->NotifyDone(&client_ctx, request, &response);
}

void CtrlCommNet::WaitUntilDone(const std::string& name) {
  grpc::ClientContext client_ctx;
  WaitUntilDoneRequest request;
  request.set_name(name);
  WaitUntilDoneResponse response;
  GetResponsibleStub(name)->WaitUntilDone(&client_ctx, request, &response);
}

void CtrlCommNet::PublishPlan(const Plan* plan) {
  ctrl_server_->PublishPlan(plan);
}

void CtrlCommNet::FetchPlan(Plan* plan) {
  grpc::ClientContext client_ctx;
  FetchPlanRequest request;
  FetchPlanResponse response;
  GetMasterStub()->FetchPlan(&client_ctx, request, &response);
  *plan = response.plan();
}

CtrlService::Stub* CtrlCommNet::GetResponsibleStub(const std::string& key) {
  int64_t machine_id =
      (std::hash<std::string>{}(key)) % JobDesc::Singleton()->TotalMachineNum();
  return stubs_[machine_id].get();
}

}  // namespace oneflow
