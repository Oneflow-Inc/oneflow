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
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/vm/virtual_machine_scope.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/device/cudnn_conv_util.h"
#include "oneflow/core/rpc/include/manager.h"
#include "oneflow/core/transport/transport.h"
#include "oneflow/core/hardware/node_device_descriptor_manager.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/framework/symbol_id_cache.h"
#include "oneflow/core/operator/op_node_signature.pb.h"
#include "oneflow/core/operator/op_conf.cfg.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/comm_network/epoll/epoll_comm_network.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_comm_network.h"
#include "oneflow/core/kernel/chain_kernel_observer.h"
#include "oneflow/core/kernel/sync_check_kernel_observer.h"
#include "oneflow/core/kernel/blob_access_checker_kernel_observer.h"
#include "oneflow/core/kernel/profiler_kernel_observer.h"
#ifdef WITH_RDMA
#include "oneflow/core/platform/include/ibv.h"
#endif  // WITH_RDMA
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/ep/cpu/cpu_device_manager.h"

namespace oneflow {

namespace {

std::string LogDir(const std::string& log_dir) {
  char hostname[255];
  CHECK_EQ(gethostname(hostname, sizeof(hostname)), 0);
  std::string v = JoinPath(log_dir, std::string(hostname));
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
    resource.set_machine_num(GlobalProcessCtx::NodeSize());
  } else {
    resource.set_machine_num(env_proto.machine_size());
  }
  resource.set_cpu_device_num(GetDefaultCpuDeviceNum());
  resource.set_gpu_device_num(GetDefaultGpuDeviceNum());
  return resource;
}

void SetCpuDeviceManagerNumThreads() {
  ep::CpuDeviceManager* cpu_device_manager = dynamic_cast<ep::CpuDeviceManager*>(
      Global<ep::DeviceManagerRegistry>::Get()->GetDeviceManager(DeviceType::kCPU));
  constexpr size_t kDefaultUsedNumThreads = 2;
  int64_t cpu_logic_core = std::thread::hardware_concurrency();
  int64_t default_num_threads =
      (cpu_logic_core / GlobalProcessCtx::NumOfProcessPerNode()) - kDefaultUsedNumThreads;
  int64_t num_threads = ParseIntegerFromEnv("ONEFLOW_EP_CPU_NUM_THREADS", default_num_threads);
  cpu_device_manager->SetDeviceNumThreads(num_threads);
}

void ClearAllSymbolAndIdCache() {
  Global<symbol::Storage<Scope>>::Get()->ClearAll();
  Global<symbol::IdCache<cfg::ScopeProto>>::Get()->ClearAll();

  Global<symbol::Storage<JobDesc>>::Get()->ClearAll();
  Global<symbol::IdCache<cfg::JobConfigProto>>::Get()->ClearAll();

  Global<symbol::Storage<ParallelDesc>>::Get()->ClearAll();
  Global<symbol::IdCache<cfg::ParallelConf>>::Get()->ClearAll();

  Global<symbol::Storage<OperatorConfSymbol>>::Get()->ClearAll();
  Global<symbol::IdCache<cfg::OperatorConf>>::Get()->ClearAll();
}

#if defined(__linux__) && defined(WITH_RDMA)

bool CommNetIBEnabled() {
  bool user_enabled = ParseBooleanFromEnv("ONEFLOW_COMM_NET_IB_ENABLE", false);
  if (user_enabled) {
    return ibv::IsAvailable();
  } else {
    return false;
  }
}

#endif

}  // namespace

Maybe<void> EnvGlobalObjectsScope::Init(const EnvProto& env_proto) {
  InitLogging(env_proto.cpp_logging_conf());
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
  Global<hardware::NodeDeviceDescriptorManager>::SetAllocated(
      new hardware::NodeDeviceDescriptorManager());
  if (Global<ResourceDesc, ForEnv>::Get()->enable_debug_mode()) {
    Global<hardware::NodeDeviceDescriptorManager>::Get()->DumpSummary("devices");
  }
  Global<ep::DeviceManagerRegistry>::New();
  Global<ThreadPool>::New(Global<ResourceDesc, ForSession>::Get()->ComputeThreadPoolSize());
  SetCpuDeviceManagerNumThreads();
#ifdef WITH_CUDA
  Global<EagerNcclCommMgr>::New();
  Global<CudnnConvAlgoCache>::New();
#endif
  Global<vm::VirtualMachineScope>::New(Global<ResourceDesc, ForSession>::Get()->resource());
  Global<EagerJobBuildAndInferCtxMgr>::New();
  if (!Global<ResourceDesc, ForSession>::Get()->enable_dry_run()) {
#ifdef __linux__
    Global<EpollCommNet>::New();
    Global<Transport>::New();
    if (Global<ResourceDesc, ForSession>::Get()->process_ranks().size() > 1) {
#ifdef WITH_RDMA
      if (CommNetIBEnabled()) {
        Global<IBVerbsCommNet>::New();
        Global<CommNet>::SetAllocated(Global<IBVerbsCommNet>::Get());
      } else {
        Global<CommNet>::SetAllocated(Global<EpollCommNet>::Get());
      }
#else
      Global<CommNet>::SetAllocated(Global<EpollCommNet>::Get());
#endif  // WITH_RDMA
    }
#endif  // __linux__
  }
  {
    std::vector<std::shared_ptr<KernelObserver>> kernel_observers;
    if (ParseBooleanFromEnv("ONEFLOW_DEBUG_KERNEL_SYNC_CHECK", false)) {
      LOG(WARNING)
          << "Environment variable ONEFLOW_DEBUG_KERNEL_SYNC_CHECK has been set to a truthy "
             "value, it will impact performance";
      kernel_observers.emplace_back(new SyncCheckKernelObserver());
    }
    if (!ParseBooleanFromEnv("ONEFLOW_KERNEL_DISABLE_BLOB_ACCESS_CHECKER", false)) {
      kernel_observers.emplace_back(new BlobAccessCheckerKernelObserver());
    }
    kernel_observers.emplace_back(new ProfilerKernelObserver());
    Global<KernelObserver>::SetAllocated(new ChainKernelObserver(kernel_observers));
  }
  TensorBufferPool::New();
  return Maybe<void>::Ok();
}

EnvGlobalObjectsScope::~EnvGlobalObjectsScope() {
  auto session_ctx = Global<MultiClientSessionContext>::Get();
  if (session_ctx != nullptr) {
    VLOG(2) << "Multi client session has not closed , env close it at env scope destruction.";
    CHECK_JUST(session_ctx->TryClose());
  }
  TensorBufferPool::Delete();
  Global<KernelObserver>::Delete();
  if (!Global<ResourceDesc, ForSession>::Get()->enable_dry_run()) {
#ifdef __linux__
    if (Global<ResourceDesc, ForSession>::Get()->process_ranks().size() > 1) {
      if (Global<EpollCommNet>::Get() != static_cast<EpollCommNet*>(Global<CommNet>::Get())) {
        Global<CommNet>::Delete();
      }
    }
    Global<Transport>::Delete();
    Global<EpollCommNet>::Delete();
#endif  // __linux__
  }
  Global<EagerJobBuildAndInferCtxMgr>::Delete();
  Global<vm::VirtualMachineScope>::Delete();
#ifdef WITH_CUDA
  Global<CudnnConvAlgoCache>::Delete();
  Global<EagerNcclCommMgr>::Delete();
#endif
  Global<ThreadPool>::Delete();
  Global<ep::DeviceManagerRegistry>::Delete();
  if (Global<ResourceDesc, ForSession>::Get() != nullptr) {
    Global<ResourceDesc, ForSession>::Delete();
  }
  Global<ResourceDesc, ForEnv>::Delete();
  Global<hardware::NodeDeviceDescriptorManager>::Delete();
  CHECK_NOTNULL(Global<CtrlClient>::Get());
  CHECK_NOTNULL(Global<EnvDesc>::Get());
  Global<RpcManager>::Delete();
  Global<ProcessCtx>::Delete();
  Global<EnvDesc>::Delete();
  ClearAllSymbolAndIdCache();
  google::ShutdownGoogleLogging();
}

}  // namespace oneflow
