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
#if defined(WITH_NPU)
#include "oneflow/core/profiler/acl_profiler.h"

namespace oneflow {
namespace profiler {

std::map<std::string, aclprofAicoreMetrics> npu_metrics_map_ = {
    {"ACL_AICORE_PIPE_UTILIZATION", ACL_AICORE_PIPE_UTILIZATION},
    {"ACL_AICORE_ARITHMETIC_UTILIZATION", ACL_AICORE_ARITHMETIC_UTILIZATION},
    {"ACL_AICORE_MEMORY_BANDWIDTH", ACL_AICORE_MEMORY_BANDWIDTH},
    {"ACL_AICORE_L0B_AND_WIDTH", ACL_AICORE_L0B_AND_WIDTH},
    {"ACL_AICORE_RESOURCE_CONFLICT_RATIO", ACL_AICORE_RESOURCE_CONFLICT_RATIO},
    {"ACL_AICORE_MEMORY_UB", ACL_AICORE_MEMORY_UB},
    {"ACL_AICORE_L2_CACHE", ACL_AICORE_L2_CACHE},
    {"ACL_AICORE_NONE", ACL_AICORE_NONE},
};

std::map<std::string, uint64_t> trace_level_map_ = {
    {"Level0", Level0},
    {"Level1", Level1},
    {"Level2", Level2},
    {"Level_none", Level_none},
};

aclError AclProfilingInit(const char* profilerResultPath, size_t length) {
  return aclprofInit(profilerResultPath, length);
}

aclError AclProfilingStart(const aclprofConfig* profilerConfig) {
  return aclprofStart(profilerConfig);
}

aclError AclProfilingStop(const aclprofConfig* profilerConfig) {
  return aclprofStop(profilerConfig);
}

aclError AclProfilingFinalize() { return aclprofFinalize(); }

aclprofConfig* AclProfilingCreateConfig(uint32_t* deviceIdList, uint32_t deviceNums,
                                        aclprofAicoreMetrics aicoreMetrics,
                                        aclprofAicoreEvents* aicoreEvents,
                                        uint64_t dataTypeConfig) {
  return aclprofCreateConfig(deviceIdList, deviceNums, aicoreMetrics, aicoreEvents, dataTypeConfig);
}

aclError AclprofSetConfig(aclprofConfigType configType, const char* config, size_t configLength) {
  return aclprofSetConfig(configType, config, configLength);
}

aclError AclProfilingDestroyConfig(const aclprofConfig* profilerConfig) {
  return aclprofDestroyConfig(profilerConfig);
}

aclprofConfig* AclPrepareTrace() {
  // ref: torch_npu/csrc/profiler/profiler_mgr.cpp
  AclProfilingInit("", 0);

  // torch_npu/profiler/profiler.py
  // torch_npu/profiler/experimental_config.py
  NpuTraceConfig npu_config = {
      /*trace_level*/ "Level2", /*metrics*/ "ACL_AICORE_PIPE_UTILIZATION",
      /*npu_memory*/ true,      /*l2_cache*/ false,
      /*record_op_args*/ false,
      /*msprof_tx*/ false,      /*op_attr*/ false};
  aclprofAicoreMetrics aic_metrics = ACL_AICORE_NONE;
  auto level_iter = trace_level_map_.find(npu_config.trace_level);
  uint64_t datatype_config =
      (level_iter == trace_level_map_.end()) ? Level0 : trace_level_map_[npu_config.trace_level];
  auto metrics_iter = npu_metrics_map_.find(npu_config.metrics);
  if (metrics_iter != npu_metrics_map_.end()
      && npu_config.metrics.compare("ACL_AICORE_NONE") != 0) {
    datatype_config |= ACL_PROF_AICORE_METRICS;
    aic_metrics = npu_metrics_map_[npu_config.metrics];
  }
  if (npu_config.l2_cache) { datatype_config |= ACL_PROF_L2CACHE; }
  if (npu_config.l2_cache) { datatype_config |= ACL_PROF_L2CACHE; }
  if (npu_config.msprof_tx) { datatype_config |= ACL_PROF_MSPROFTX; }
  if (npu_config.npu_memory) {
    datatype_config |= ACL_PROF_TASK_MEMORY;
    const std::string freq = "50";
    auto prof_ret = AclprofSetConfig(ACL_PROF_SYS_HARDWARE_MEM_FREQ, freq.c_str(), freq.size());
    if (prof_ret == ACL_ERROR_PROF_MODULES_UNSUPPORTED) {
      LOG(WARNING) << "ProfileManager npu AclprofSetConfig() failed: "
                   << "not support to set config for sys-hardware-mem.";
    }
  }
  if (npu_config.op_attr) { datatype_config |= ACL_PROF_OP_ATTR; }

  uint32_t deviceId = 0;
  // TODO: get current local device
  // auto ret = c10_npu::GetDevice(&deviceId);
  // if (ret != ACL_ERROR_NONE) {
  //   LOG(WARNING) <<"ProfileManager npu AclprofSetConfig() failed: " << "Get Device ID failed.";
  //   return;
  // }
  const uint32_t deviceNum = 1;
  uint32_t deviceIdList[deviceNum] = {deviceId};
  aclprofConfig* profConfig =
      AclProfilingCreateConfig(deviceIdList, deviceNum, aic_metrics, nullptr, datatype_config);
  return profConfig;
}

aclError AclStartTrace(aclprofConfig* profConfig) { return AclProfilingStart(profConfig); }

void AclReleaseTrace(aclprofConfig* profConfig) {
  aclrtSynchronizeDevice();
  // stop
  AclProfilingStop(profConfig);
  auto ret = AclProfilingDestroyConfig(profConfig);
  if (ret != ACL_SUCCESS) {
    LOG(WARNING) << "ProfileManager npu AclReleaseTrace() failed: "
                 << "AclProfDestoryConfig fail, error code: " << ret;
    return;
  }
  profConfig = nullptr;

  // finalize
  AclProfilingFinalize();
}

}  // namespace profiler
}  // namespace oneflow

#endif  // WITH_NPU
