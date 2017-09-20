#include "oneflow/core/comm_network/ctrl_comm_network.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

namespace {

const int32_t max_retry_num = 60;
const int64_t sleep_seconds = 10;

}  // namespace

void CtrlCommNet::Init() {
  ctrl_server_.reset(
      new CtrlServer(RuntimeCtx::Singleton()->GetThisCtrlAddr()));
  stubs_.reserve(JobDesc::Singleton()->TotalMachineNum());
  for (int64_t i = 0; i < JobDesc::Singleton()->TotalMachineNum(); ++i) {
    stubs_.push_back(
        CtrlService::NewStub(RuntimeCtx::Singleton()->GetCtrlAddr(i)));
  }
  int32_t retry_idx = 0;
  for (; retry_idx < max_retry_num; ++retry_idx) {
    grpc::ClientContext client_ctx;
    AddWorkerRequest request;
    request.set_worker_ctrl_addr(RuntimeCtx::Singleton()->GetThisCtrlAddr());
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

}  // namespace oneflow
