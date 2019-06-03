#ifndef ONEFLOW_CORE_JOB_RESOURCE_DESC_H_
#define ONEFLOW_CORE_JOB_RESOURCE_DESC_H_

#include "oneflow/core/job/resource.pb.h"

namespace oneflow {

class ResourceDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ResourceDesc);
  ~ResourceDesc() = default;

  size_t TotalMachineNum() const { return resource_.machine().size(); }
  const Machine& machine(int32_t idx) const { return resource_.machine(idx); }
  int32_t ctrl_port() const { return resource_.ctrl_port(); }
  int32_t data_port() const { return resource_.data_port(); }
  size_t CommNetWorkerNum() const { return resource_.comm_net_worker_num(); }
  size_t rdma_mem_block_byte() const { return config_.rdma_mem_block_mbyte() * 1024 * 1024; }
  size_t rdma_recv_msg_buf_byte() const { return config_.rdma_recv_msg_buf_mbyte() * 1024 * 1024; }
  int32_t CpuDeviceNum() const { return resource_.cpu_device_num(); }
  void SetCpuDeviceNum(int32_t val) { resource_.set_cpu_device_num(val); }
  int32_t GpuDeviceNum() const { return resource_.gpu_device_num(); }
  int32_t MemZoneNum() const { return GpuDeviceNum() + 1; }
  int32_t MaxMdSaveWorkerNum() const { return resource_.max_mdsave_worker_num(); }

 private:
  friend class Global<ResourceDesc>;
  explicit ResourceDesc(const JobConf& job_conf)
      : resource_(job_conf.resource()), config_(job_conf.other()) {}

  Resource resource_;
  Config config_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_RESOURCE_DESC_H_
