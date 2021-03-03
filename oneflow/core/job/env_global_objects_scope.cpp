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
#ifdef WITH_CUDA
#include <cuda.h>
#endif  // WITH_CUDA
#include <thread>
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/control/ctrl_bootstrap.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/vm/virtual_machine_scope.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/device/cudnn_conv_util.h"

namespace oneflow {

namespace {

std::string LogDir(const std::string& log_dir) {
  char hostname[255];
  CHECK_EQ(gethostname(hostname, sizeof(hostname)), 0);
  std::string v = log_dir + "/" + std::string(hostname);
  return v;
}

void InitLogging(const CppLoggingConf& logging_conf) {
  FLAGS_log_dir = LogDir(logging_conf.log_dir());
  FLAGS_logtostderr = logging_conf.logtostderr();
  FLAGS_logbuflevel = logging_conf.logbuflevel();
  FLAGS_stderrthreshold = 1;  // 1=WARNING
  google::InitGoogleLogging("oneflow");
  LocalFS()->RecursivelyCreateDirIfNotExist(FLAGS_log_dir);
}

int32_t GetDefaultCpuDeviceNum() { return std::thread::hardware_concurrency(); }

int32_t GetDefaultGpuDeviceNum() {
#ifndef WITH_CUDA
  return 0;
#else
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  return device_count;
#endif
}

Resource GetDefaultResource(const EnvProto& env_proto) {
  Resource resource;
  if (env_proto.has_ctrl_bootstrap_conf()) {
    resource.set_machine_num(env_proto.ctrl_bootstrap_conf().world_size());
  } else {
    resource.set_machine_num(env_proto.machine_size());
  }
  resource.set_cpu_device_num(GetDefaultCpuDeviceNum());
  resource.set_gpu_device_num(GetDefaultGpuDeviceNum());
  return resource;
}

Maybe<CtrlBootstrap> MakeCtrlBootstrap(const EnvDesc& env_desc) {
  std::shared_ptr<CtrlBootstrap> ctrl_bootstrap;
  if (env_desc.has_ctrl_bootstrap_conf()) {
    ctrl_bootstrap.reset(new RankInfoCtrlBootstrap(env_desc.bootstrap_conf()));
  } else {
    ctrl_bootstrap.reset(new HostListCtrlBootstrap(env_desc));
  }
  return ctrl_bootstrap;
}

Maybe<int> GetCtrlPort(const EnvDesc& env_desc) {
  int port = 0;
  if (env_desc.has_bootstrap_conf_ctrl_port()) { port = env_desc.bootstrap_conf_ctrl_port(); }
  return port;
}

}  // namespace

Maybe<void> EnvGlobalObjectsScope::Init(const EnvProto& env_proto) {
  InitLogging(env_proto.cpp_logging_conf());
#ifdef WITH_CUDA
  InitGlobalCudaDeviceProp();
#endif
  Global<EnvDesc>::New(env_proto);
  Global<CtrlServer>::New(JUST(GetCtrlPort(*Global<EnvDesc>::Get())));
  Global<ProcessCtx>::New();
  // Avoid dead lock by using CHECK_JUST instead of JUST. because it maybe be blocked in
  // ~CtrlBootstrap.
  CHECK_JUST(JUST(MakeCtrlBootstrap(*Global<EnvDesc>::Get()))
                 ->InitProcessCtx(Global<CtrlServer>::Get()->port(), Global<ProcessCtx>::Get()));
  Global<CtrlClient>::New(*Global<ProcessCtx>::Get());
  Global<ResourceDesc, ForEnv>::New(GetDefaultResource(env_proto));
  Global<ResourceDesc, ForSession>::New(GetDefaultResource(env_proto));
  Global<ThreadPool>::New(Global<ResourceDesc, ForSession>::Get()->ComputeThreadPoolSize());
  Global<vm::VirtualMachineScope>::New(Global<ResourceDesc, ForSession>::Get()->resource());
  Global<EagerJobBuildAndInferCtxMgr>::New();
#ifdef WITH_CUDA
  Global<EagerNcclCommMgr>::New();
  Global<CudnnConvAlgoCache>::New();
#endif
  return Maybe<void>::Ok();
}

EnvGlobalObjectsScope::~EnvGlobalObjectsScope() {
#ifdef WITH_CUDA
  Global<CudnnConvAlgoCache>::Delete();
  Global<EagerNcclCommMgr>::Delete();
#endif
  Global<EagerJobBuildAndInferCtxMgr>::Delete();
  Global<vm::VirtualMachineScope>::Delete();
  Global<ThreadPool>::Delete();
  if (Global<ResourceDesc, ForSession>::Get() != nullptr) {
    Global<ResourceDesc, ForSession>::Delete();
  }
  Global<ResourceDesc, ForEnv>::Delete();
  CHECK_NOTNULL(Global<CtrlClient>::Get());
  CHECK_NOTNULL(Global<CtrlServer>::Get());
  CHECK_NOTNULL(Global<EnvDesc>::Get());
  Global<CtrlClient>::Delete();
  Global<ProcessCtx>::Delete();
  Global<CtrlServer>::Delete();
  Global<EnvDesc>::Delete();
#ifdef WITH_CUDA
  Global<cudaDeviceProp>::Delete();
#endif
}

}  // namespace oneflow
