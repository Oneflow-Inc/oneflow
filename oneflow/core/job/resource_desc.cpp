/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <algorithm>
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/common/util.h"
#ifdef WITH_CUDA
#include <nccl.h>
#endif

namespace oneflow {

ResourceDesc::ResourceDesc(const Resource& resource, int64_t num_process_per_node)
    : resource_(resource) {
  CHECK_GT(resource_.machine_num(), 0);
  CHECK_LE(resource_.machine_num(), Global<EnvDesc>::Get()->TotalMachineNum());
  int64_t max_device_num = std::max(resource.gpu_device_num(), resource.cpu_device_num());
  CHECK_GT(max_device_num, 0);
  max_device_num = std::min(max_device_num, num_process_per_node);
  for (int i = 0; i < resource_.machine_num(); ++i) {
    for (int j = 0; j < max_device_num; ++j) {
      CHECK(process_ranks_.emplace(i * num_process_per_node + j).second);
    }
  }
}

Machine ResourceDesc::machine(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK(process_ranks().find(idx) != process_ranks().end());
  if (Global<EnvDesc>::Get()->has_ctrl_bootstrap_conf()) {
    CHECK_NOTNULL(Global<ProcessCtx>::Get());
    CHECK_GE(Global<ProcessCtx>::Get()->ctrl_addr().size(), process_ranks().size());
    Machine machine;
    const Address& addr = Global<ProcessCtx>::Get()->ctrl_addr(idx);
    machine.set_addr(addr.host());
    return machine;
  } else {
    return Global<EnvDesc>::Get()->machine(idx);
  }
}

int32_t ResourceDesc::ComputeThreadPoolSize() const {
  if (resource_.has_compute_thread_pool_size()) {
    CHECK_GT(resource_.compute_thread_pool_size(), 0);
    return resource_.compute_thread_pool_size();
  } else {
    return CpuDeviceNum();
  }
}

bool ResourceDesc::enable_debug_mode() const {
  return std::getenv("ONEFLOW_DEBUG_MODE") != nullptr || std::getenv("ONEFLOW_DEBUG") != nullptr
         || resource_.enable_debug_mode();
}

bool ResourceDesc::enable_dry_run() const { return std::getenv("ONEFLOW_DRY_RUN") != nullptr; }

CollectiveBoxingConf ResourceDesc::collective_boxing_conf() const {
  if (resource_.has_collective_boxing_conf()) {
    return resource_.collective_boxing_conf();
  } else {
    return CollectiveBoxingConf();
  }
}

bool ResourceDesc::nccl_use_compute_stream() const {
#if defined(WITH_CUDA) && NCCL_VERSION_CODE > 2700
  return resource_.nccl_use_compute_stream();
#else
  return false;
#endif
}

void ResourceDesc::DumpCudnnConf(const JobConfigProto& job_conf) {
  resource_.clear_cudnn_conf();
  auto* cudnn_conf = resource_.mutable_cudnn_conf();
  if (job_conf.has_enable_cudnn()) { cudnn_conf->set_enable_cudnn(job_conf.enable_cudnn()); }
  if (job_conf.has_cudnn_buf_limit_mbyte()) {
    cudnn_conf->set_cudnn_buf_limit_mbyte(job_conf.cudnn_buf_limit_mbyte());
  }
  if (job_conf.has_cudnn_conv_force_fwd_algo()) {
    cudnn_conf->set_cudnn_conv_force_fwd_algo(job_conf.cudnn_conv_force_fwd_algo());
  }
  if (job_conf.has_cudnn_conv_force_bwd_data_algo()) {
    cudnn_conf->set_cudnn_conv_force_bwd_data_algo(job_conf.cudnn_conv_force_bwd_data_algo());
  }
  if (job_conf.has_cudnn_conv_force_bwd_filter_algo()) {
    cudnn_conf->set_cudnn_conv_force_bwd_filter_algo(job_conf.cudnn_conv_force_bwd_filter_algo());
  }
  if (job_conf.has_cudnn_conv_heuristic_search_algo()) {
    cudnn_conf->set_cudnn_conv_heuristic_search_algo(job_conf.cudnn_conv_heuristic_search_algo());
  }
  if (job_conf.has_cudnn_conv_use_deterministic_algo_only()) {
    cudnn_conf->set_cudnn_conv_use_deterministic_algo_only(
        job_conf.cudnn_conv_use_deterministic_algo_only());
  }
  if (job_conf.has_enable_cudnn_fused_normalization_add_relu()) {
    cudnn_conf->set_enable_cudnn_fused_normalization_add_relu(
        job_conf.enable_cudnn_fused_normalization_add_relu());
  }
  if (job_conf.has_cudnn_conv_enable_pseudo_half()) {
    cudnn_conf->set_cudnn_conv_enable_pseudo_half(job_conf.cudnn_conv_enable_pseudo_half());
  }
}

}  // namespace oneflow
