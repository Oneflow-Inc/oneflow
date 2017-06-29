#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

void RuntimeCtx::set_this_machine_name(const std::string& name) {
  this_machine_name_ = name;
  this_machine_id_ = IDMgr::Singleton().MachineID4MachineName(name);
  LOG(INFO) << "this machine name: " << this_machine_name_;
  LOG(INFO) << "this machine id: " << this_machine_id_;
}

void RuntimeCtx::SetModelInitCnt(int32_t val) {
  std::unique_lock<std::mutex> lck(model_init_cnt_mtx_);
  model_init_cnt_ = val;
}

void RuntimeCtx::OneModelInitDone() {
  std::unique_lock<std::mutex> lck(model_init_cnt_mtx_);
  model_init_cnt_ -= 1;
  if (model_init_cnt_ == 0) {
    model_init_cnt_cond_.notify_one();
  }
}

void RuntimeCtx::WaitUnitlAllModelInitDone() {
  std::unique_lock<std::mutex> lck(model_init_cnt_mtx_);
  model_init_cnt_cond_.wait(lck, [this]() {
    return model_init_cnt_ == 0;
  });
}

void RuntimeCtx::InitDataReader(const std::string& filepath) {
  data_reader_.reset(new PersistentCircularLineReader(filepath));
}

} // namespace oneflow
