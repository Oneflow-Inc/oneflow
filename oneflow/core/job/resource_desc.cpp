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
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/env_var/debug_mode.h"
#include "oneflow/core/control/global_process_ctx.h"
#ifdef WITH_CUDA
#include <nccl.h>
#endif

namespace oneflow {

ResourceDesc::ResourceDesc(const Resource& resource, int64_t num_process_per_node)
    : resource_(resource) {
  CHECK_GT(resource_.machine_num(), 0);
  CHECK_LE(resource_.machine_num(), Singleton<EnvDesc>::Get()->TotalMachineNum());
  for (int i = 0; i < GlobalProcessCtx::WorldSize(); ++i) {
    CHECK(process_ranks_.emplace(i).second);
  }
}

Machine ResourceDesc::machine(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK(process_ranks().find(idx) != process_ranks().end());
  if (Singleton<EnvDesc>::Get()->has_ctrl_bootstrap_conf()) {
    CHECK_NOTNULL(Singleton<ProcessCtx>::Get());
    CHECK_GE(Singleton<ProcessCtx>::Get()->ctrl_addr().size(), process_ranks().size());
    Machine machine;
    const Address& addr = Singleton<ProcessCtx>::Get()->ctrl_addr(idx);
    machine.set_addr(addr.host());
    return machine;
  } else {
    return Singleton<EnvDesc>::Get()->machine(idx);
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
  return IsInDebugMode() || resource_.enable_debug_mode();
}

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

void ResourceDesc::Update(const Resource& reso_conf) { resource_.CopyFrom(reso_conf); }

}  // namespace oneflow
