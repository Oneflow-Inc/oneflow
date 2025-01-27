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
#include <string>
#include <map>
#include <glog/logging.h>
#include "acl/acl.h"
#include "acl/acl_prof.h"

namespace oneflow {
namespace profiler {

// trace_level
constexpr uint64_t Level_none = 0;
constexpr uint64_t Level0 = ACL_PROF_TASK_TIME_L0 | ACL_PROF_ACL_API;
constexpr uint64_t Level1 =
    ACL_PROF_TASK_TIME | ACL_PROF_ACL_API | ACL_PROF_HCCL_TRACE | ACL_PROF_AICORE_METRICS;
constexpr uint64_t Level2 = Level1 | ACL_PROF_RUNTIME_API | ACL_PROF_AICPU;

struct NpuTraceConfig {
  std::string trace_level;
  std::string metrics;
  bool npu_memory;
  bool l2_cache;
  bool record_op_args;
  bool msprof_tx;
  bool op_attr;
};

#define ACL_PROF_OP_ATTR 0x00004000ULL

aclError AclProfilingInit(const char* profilerResultPath, size_t length);
aclError AclProfilingStart(const aclprofConfig* profilerConfig);
aclError AclProfilingStop(const aclprofConfig* profilerConfig);
aclError AclProfilingFinalize();
aclprofConfig* AclProfilingCreateConfig(uint32_t* deviceIdList, uint32_t deviceNums,
                                        aclprofAicoreMetrics aicoreMetrics,
                                        aclprofAicoreEvents* aicoreEvents, uint64_t dataTypeConfig);
aclError AclprofSetConfig(aclprofConfigType configType, const char* config, size_t configLength);
aclError AclProfilingDestroyConfig(const aclprofConfig* profilerConfig);

aclprofConfig* AclPrepareTrace();
aclError AclStartTrace(aclprofConfig* profConfig);
void AclReleaseTrace(aclprofConfig* profConfig);

}  // namespace profiler
}  // namespace oneflow

#endif  // WITH_NPU
