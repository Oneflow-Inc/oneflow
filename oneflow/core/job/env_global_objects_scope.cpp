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
#include "oneflow/core/rpc/include/manager.h"
#include "oneflow/core/transport/transport.h"
#include "oneflow/core/device/node_device_descriptor_manager.h"

namespace oneflow {

namespace {

std::string LogDir(const std::string& log_dir) {
  char hostname[255];
  CHECK_EQ(gethostname(hostname, sizeof(hostname)), 0);
  std::string v = JoinPath(log_dir, std::string(hostname));
  return v;
}

void InitLogging(const CppLoggingConf& logging_conf, bool default_physical_env) {
  if (!default_physical_env) {
    FLAGS_log_dir = LogDir(logging_conf.log_dir());
  } else {
    std::string default_env_log_path = JoinPath(logging_conf.log_dir(), "default_physical_env_log");
    FLAGS_log_dir = LogDir(default_env_log_path);
  }
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
    resource.set_machine_num(GlobalProcessCtx::NodeSize());
  } else {
    resource.set_machine_num(env_proto.machine_size());
  }
  resource.set_cpu_device_num(GetDefaultCpuDeviceNum());
  resource.set_gpu_device_num(GetDefaultGpuDeviceNum());
  return resource;
}

}  // namespace

Maybe<void> EnvGlobalObjectsScope::Init(const EnvProto& env_proto) {
  thread_id_ = std::this_thread::get_id();
  is_default_physical_env_ = env_proto.is_default_physical_env();
  InitLogging(env_proto.cpp_logging_conf(), JUST(is_default_physical_env_));
#ifdef WITH_CUDA
  InitGlobalCudaDeviceProp();
#endif
  Global<EnvDesc>::New(env_proto);
  Global<ProcessCtx>::New();
  // Avoid dead lock by using CHECK_JUST instead of JUST. because it maybe be blocked in
  // ~CtrlBootstrap.
  if (Global<ResourceDesc, ForSession>::Get()->enable_dry_run()) {
#ifdef RPC_BACKEND_LOCAL
    LOG(INFO) << "using rpc backend: dry-run";
    Global<RpcManager>::SetAllocated(new DryRunRpcManager());
#else
    static_assert(false, "requires rpc backend dry-run to dry run oneflow");
#endif  // RPC_BACKEND_LOCAL
  } else if ((env_proto.machine_size() == 1 && env_proto.has_ctrl_bootstrap_conf() == false)
             || (env_proto.has_ctrl_bootstrap_conf()
                 && env_proto.ctrl_bootstrap_conf().world_size() == 1)) /*single process*/ {
#ifdef RPC_BACKEND_LOCAL
    LOG(INFO) << "using rpc backend: local";
    Global<RpcManager>::SetAllocated(new LocalRpcManager());
#else
    static_assert(false, "requires rpc backend local to run oneflow in single processs");
#endif  // RPC_BACKEND_LOCAL
  } else /*multi process, multi machine*/ {
#ifdef RPC_BACKEND_GRPC
    LOG(INFO) << "using rpc backend: gRPC";
    Global<RpcManager>::SetAllocated(new GrpcRpcManager());
#else
    UNIMPLEMENTED() << "to run distributed oneflow, you must enable at least one multi-node rpc "
                       "backend by adding cmake argument, for instance: -DRPC_BACKEND=GRPC";
#endif  // RPC_BACKEND_GRPC
  }
  CHECK_JUST(Global<RpcManager>::Get()->CreateServer());
  CHECK_JUST(Global<RpcManager>::Get()->Bootstrap());
  CHECK_JUST(Global<RpcManager>::Get()->CreateClient());
  Global<ResourceDesc, ForEnv>::New(GetDefaultResource(env_proto),
                                    GlobalProcessCtx::NumOfProcessPerNode());
  Global<ResourceDesc, ForSession>::New(GetDefaultResource(env_proto),
                                        GlobalProcessCtx::NumOfProcessPerNode());
  Global<device::NodeDeviceDescriptorManager>::SetAllocated(
      new device::NodeDeviceDescriptorManager());
  if (Global<ResourceDesc, ForEnv>::Get()->enable_debug_mode()) {
    Global<device::NodeDeviceDescriptorManager>::Get()->DumpSummary("devices");
  }
  Global<ThreadPool>::New(Global<ResourceDesc, ForSession>::Get()->ComputeThreadPoolSize());
#ifdef WITH_CUDA
  Global<EagerNcclCommMgr>::New();
  Global<CudnnConvAlgoCache>::New();
#endif
  Global<vm::VirtualMachineScope>::New(Global<ResourceDesc, ForSession>::Get()->resource());
  Global<EagerJobBuildAndInferCtxMgr>::New();
  if (!Global<ResourceDesc, ForSession>::Get()->enable_dry_run()) {
    Global<EpollCommNet>::New();
    Global<Transport>::New();
  }
  return Maybe<void>::Ok();
}

EnvGlobalObjectsScope::~EnvGlobalObjectsScope() {
  if (!Global<ResourceDesc, ForSession>::Get()->enable_dry_run()) {
    Global<Transport>::Delete();
    Global<EpollCommNet>::Delete();
  }
  Global<EagerJobBuildAndInferCtxMgr>::Delete();
  Global<vm::VirtualMachineScope>::Delete();
#ifdef WITH_CUDA
  Global<CudnnConvAlgoCache>::Delete();
  Global<EagerNcclCommMgr>::Delete();
#endif
  Global<ThreadPool>::Delete();
  if (Global<ResourceDesc, ForSession>::Get() != nullptr) {
    Global<ResourceDesc, ForSession>::Delete();
  }
  Global<ResourceDesc, ForEnv>::Delete();
  Global<device::NodeDeviceDescriptorManager>::Delete();
  CHECK_NOTNULL(Global<CtrlClient>::Get());
  CHECK_NOTNULL(Global<EnvDesc>::Get());
  Global<RpcManager>::Delete();
  Global<ProcessCtx>::Delete();
  Global<EnvDesc>::Delete();
#ifdef WITH_CUDA
  Global<cudaDeviceProp>::Delete();
#endif
  google::ShutdownGoogleLogging();
}

const std::shared_ptr<const ParallelDesc>& EnvGlobalObjectsScope::MutParallelDesc4Device(
    const Device& device) {
  CHECK(thread_id_ == std::this_thread::get_id());
  {
    const auto& iter = device2parallel_desc_.find(device);
    if (iter != device2parallel_desc_.end()) { return iter->second; }
  }
  std::string machine_device_id =
      "@" + std::to_string(GlobalProcessCtx::Rank()) + ":" + std::to_string(device.device_id());
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag(CHECK_JUST(device.of_type()));
  parallel_conf.add_device_name(machine_device_id);
  std::shared_ptr<const ParallelDesc> parallel_desc =
      std::make_shared<const ParallelDesc>(parallel_conf);
  return device2parallel_desc_.emplace(device, parallel_desc).first->second;
}

}  // namespace oneflow
