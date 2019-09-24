#ifndef ONEFLOW_CORE_JOB_RESOURCE_DESC_H_
#define ONEFLOW_CORE_JOB_RESOURCE_DESC_H_

#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

static const size_t kMB = 1024 * 1024;

class ResourceDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ResourceDesc);
  ~ResourceDesc() = default;

  size_t TotalMachineNum() const { return resource_.machine().size(); }
  const Machine& machine(int32_t idx) const { return resource_.machine(idx); }
  int32_t ctrl_port() const { return resource_.ctrl_port(); }
  int32_t data_port() const { return resource_.data_port(); }
  size_t CommNetWorkerNum() const { return resource_.comm_net_worker_num(); }
  size_t rdma_mem_block_byte() const { return resource_.rdma_mem_block_mbyte() * kMB; }
  size_t rdma_recv_msg_buf_byte() const { return resource_.rdma_recv_msg_buf_mbyte() * kMB; }
  int32_t CpuDeviceNum() const { return resource_.cpu_device_num(); }
  void SetCpuDeviceNum(int32_t val) { resource_.set_cpu_device_num(val); }
  int32_t GpuDeviceNum() const { return resource_.gpu_device_num(); }
  int32_t MemZoneNum() const { return GpuDeviceNum() + 1; }
  int32_t MaxMdSaveWorkerNum() const { return resource_.max_mdsave_worker_num(); }
  size_t reserved_host_mem_byte() const { return resource_.reserved_host_mem_mbyte() * kMB; }
  size_t reserved_device_mem_byte() const { return resource_.reserved_device_mem_mbyte() * kMB; }
  int64_t GetMachineId(const std::string& addr) const;
  bool use_rdma() const { return resource_.use_rdma(); }
  bool enable_numa_aware_cuda_malloc_host() const {
    return resource_.enable_numa_aware_cuda_malloc_host();
  }
  int32_t ComputeThreadPoolSize() const;

 private:
  friend class Global<ResourceDesc>;
  explicit ResourceDesc(const Resource& resource) : resource_(resource) {}

  Resource resource_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_RESOURCE_DESC_H_
