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

#ifdef WITH_NPU
#include <mutex>
#include "oneflow/core/device/npu_util.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/hardware/node_device_descriptor_manager.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/ep/npu/npu_stream.h"
#include "oneflow/core/common/util.h"
#include <acl/acl_op_compiler.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <ge/ge_api.h>

#define GetCurrentDirPath getcwd
#define Mkdir(path, mode) mkdir(path, mode)

namespace oneflow {


NpuCurrentDeviceGuard::NpuCurrentDeviceGuard(int32_t dev_id) {
  CHECK(!pthread_fork::IsForkedSubProcess()) << pthread_fork::kOfNpuNotSupportInForkedSubProcess;
  // if (!is_dev_init) 
  // {
  //   OF_NPU_CHECK(aclrtSetDevice(dev_id));
  //   is_dev_init = true;
  // }
  OF_NPU_CHECK(aclrtSetDevice(dev_id));
  OF_NPU_CHECK(aclrtGetDevice(&saved_dev_id_));
  // OF_NPU_CHECK(aclrtSetDevice(dev_id));
}

NpuCurrentDeviceGuard::NpuCurrentDeviceGuard() {
  OF_NPU_CHECK(aclrtGetDevice(&saved_dev_id_)); 
}

NpuCurrentDeviceGuard::~NpuCurrentDeviceGuard() { 
  OF_NPU_CHECK(aclrtSetDevice(saved_dev_id_));
}

int GetNpuDeviceIndex() { return GlobalProcessCtx::LocalRank(); }

int GetNpuDeviceCount() {
  //std::cout<<"GetNpuDeviceCount"<<std::endl;
  /* static */ uint32_t npu_device_count = 0;
  NpuCurrentDeviceGuard dev_guard(GetNpuDeviceIndex());
  OF_NPU_CHECK(aclrtGetDeviceCount(&npu_device_count));
  return npu_device_count;
}

namespace {

const size_t kMaxPathLen = 4096U;
std::string GetCurDirPath() {
  char buff[kMaxPathLen] = {'\0'};
  GetCurrentDirPath(buff, kMaxPathLen);
  return std::string(buff);
}
void MakeCompileCacheDirAndSetOption() {
  auto compile_cache_dir = GetCurDirPath() + "/cache";
  // mode : 750
  auto ret = Mkdir(compile_cache_dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP);
  if (ret == -1) {
    if (errno != EEXIST) {
      std::cout<<"make compile cache directory error: "<<strerror(errno)<<std::endl;
      return;
    }
  }
  std::string val = "enable";
  std::cout<<val<<" "<<compile_cache_dir<<std::endl;
  OF_NPU_CHECK(aclSetCompileopt(aclCompileOpt::ACL_OP_COMPILER_CACHE_MODE, val.c_str()));
  OF_NPU_CHECK(aclSetCompileopt(aclCompileOpt::ACL_OP_COMPILER_CACHE_DIR, compile_cache_dir.c_str()));
  std::cout<<"MakeCompileCacheDirAndSetOption Over"<<std::endl;
}
void InitNpuOtherOnce(int device_id_)
{
    auto npu_device_id = std::to_string(device_id_);
    std::map<ge::AscendString, ge::AscendString> config = {
        {ge::AscendString(ge::OPTION_EXEC_DEVICE_ID),
         ge::AscendString(npu_device_id.data())},
        {ge::AscendString(ge::OPTION_GRAPH_RUN_MODE), "0"},
        {ge::AscendString(ge::PRECISION_MODE.data()), "allow_fp32_to_fp16"},
        {ge::AscendString(ge::VARIABLE_MEMORY_MAX_SIZE), "1048576"},
        {ge::AscendString(ge::OP_SELECT_IMPL_MODE.data()), "high_precision"}
    };

    config["ge.session_device_id"] = ge::AscendString(npu_device_id.data());
    config["ge.exec.reuseZeroCopyMemory"] = ge::AscendString("1");
  auto ge_ret = ge::GEInitialize(config);
  if (ge_ret != ge::SUCCESS) {
      std::cout<<("GE init failed!")<<std::endl;
  }
  MakeCompileCacheDirAndSetOption();
}

} // namespace

void InitNpuContextOnce(int device_id ) {
  static std::once_flag aclcontext;
  static aclrtContext context_;
  std::call_once(aclcontext,[&](){
    //std::cout<<"Init && Create Context Once"<<std::endl;
    static std::string json_path = "/data/acl_test/acl.json";
    OF_NPU_CHECK(aclInit(json_path.c_str()));
    OF_NPU_CHECK(aclrtCreateContext(&context_, device_id));
    // InitNpuOtherOnce(device_id);
  });
  static int device_count = GetNpuDeviceCount();
  static std::vector<std::once_flag> init_flags = std::vector<std::once_flag>(device_count);
  if (LazyMode::is_enabled()) { return; }
  if (device_id == -1) { device_id = GetNpuDeviceIndex(); }
  std::call_once(init_flags[device_id], [&]() {
    OF_NPU_CHECK(aclrtSetDevice(device_id));
    OF_NPU_CHECK(aclrtSynchronizeDevice());
  });
}


void NpuSynchronize(int device_id) {
  NpuCurrentDeviceGuard dev_guard(device_id);
  OF_NPU_CHECK(aclrtSynchronizeDevice());
}
} // namespace oneflow

#endif  // WITH_NPU
