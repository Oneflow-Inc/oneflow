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
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/operator/op_node_signature.pb.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/comm_network/epoll/epoll_comm_network.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_comm_network.h"
#include "oneflow/core/kernel/chain_kernel_observer.h"
#include "oneflow/core/kernel/sync_check_kernel_observer.h"
#include "oneflow/core/kernel/blob_access_checker_kernel_observer.h"
#include "oneflow/core/kernel/profiler_kernel_observer.h"
#include "oneflow/core/embedding/embedding_manager.h"
#ifdef WITH_RDMA
#include "oneflow/core/platform/include/ibv.h"
#include "oneflow/core/comm_network/ibverbs/ibverbs_comm_network.h"
#endif  // WITH_RDMA
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/ep/cpu/cpu_device_manager.h"
#include "oneflow/core/common/env_var/debug_mode.h"

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
  FLAGS_minloglevel = logging_conf.minloglevel();
  FLAGS_stderrthreshold = 1;  // 1=WARNING
  google::InitGoogleLogging("oneflow");
  if (IsInDebugMode()) {
    // record all level logs to file in debug mode
    FLAGS_logtostderr = 0;
    FLAGS_minloglevel = 0;  // 0=INFO
  }
  if (!FLAGS_logtostderr) { LocalFS()->RecursivelyCreateDirIfNotExist(FLAGS_log_dir); }
}

int32_t GetDefaultCpuDeviceNum() { return std::thread::hardware_concurrency(); }

Resource GetDefaultResource(const EnvProto& env_proto) {
  Resource resource;
  if (env_proto.has_ctrl_bootstrap_conf()) {
    resource.set_machine_num(GlobalProcessCtx::NodeSize());
  } else {
    resource.set_machine_num(env_proto.machine_size());
  }
  resource.set_cpu_device_num(GetDefaultCpuDeviceNum());
  return resource;
}

void SetCpuDeviceManagerNumThreads() {
  ep::CpuDeviceManager* cpu_device_manager = dynamic_cast<ep::CpuDeviceManager*>(
      Singleton<ep::DeviceManagerRegistry>::Get()->GetDeviceManager(DeviceType::kCPU));
  constexpr size_t kDefaultUsedNumThreads = 2;
  int64_t cpu_logic_core = std::thread::hardware_concurrency();
  int64_t default_num_threads =
      (cpu_logic_core / GlobalProcessCtx::NumOfProcessPerNode()) - kDefaultUsedNumThreads;
  int64_t num_threads = ParseIntegerFromEnv("OMP_NUM_THREADS", default_num_threads);
  cpu_device_manager->SetDeviceNumThreads(num_threads);
}

void ClearAllSymbol() {
  Singleton<symbol::Storage<Scope>>::Get()->ClearAll();
  Singleton<symbol::Storage<JobDesc>>::Get()->ClearAll();
  Singleton<symbol::Storage<ParallelDesc>>::Get()->ClearAll();
  Singleton<symbol::Storage<OperatorConfSymbol>>::Get()->ClearAll();
}

#if defined(WITH_RDMA) && defined(OF_PLATFORM_POSIX)

bool CommNetIBEnabled() {
  if (!ibv::IsAvailable()) { return false; }
  const auto* node_manager = Singleton<hardware::NodeDeviceDescriptorManager>::Get();
  if (node_manager == nullptr) { return false; }
  for (int64_t rank = 0; rank < GlobalProcessCtx::WorldSize(); ++rank) {
    const auto& node = node_manager->GetNodeDeviceDescriptor(rank);
    if (!node) { return false; }
    const auto& list = node->GetDeviceDescriptorList("net_ib");
    if (!list) { return false; }
    if (list->DeviceCount() == 0) { return false; }
  }
  return true;
}

#endif  // WITH_RDMA && OF_PLATFORM_POSIX

}  // namespace

EnvGlobalObjectsScope::EnvGlobalObjectsScope(const std::string& env_proto_str) {
  EnvProto env_proto;
  CHECK(TxtString2PbMessage(env_proto_str, &env_proto))
      << "failed to parse env_proto" << env_proto_str;
  CHECK_JUST(Init(env_proto));
}

EnvGlobalObjectsScope::EnvGlobalObjectsScope(const EnvProto& env_proto) {
  CHECK_JUST(Init(env_proto));
}

Maybe<void> EnvGlobalObjectsScope::Init(const EnvProto& env_proto) {
  CHECK(Singleton<EnvGlobalObjectsScope>::Get() == nullptr);
  Singleton<EnvGlobalObjectsScope>::SetAllocated(this);

  InitLogging(env_proto.cpp_logging_conf());
  Singleton<EnvDesc>::New(env_proto);
  Singleton<ProcessCtx>::New();
  // Avoid dead lock by using CHECK_JUST instead of JUST. because it maybe be blocked in
  // ~CtrlBootstrap.

  if ((env_proto.machine_size() == 1 && env_proto.has_ctrl_bootstrap_conf() == false)
      || (env_proto.has_ctrl_bootstrap_conf()
          && env_proto.ctrl_bootstrap_conf().world_size() == 1)) /*single process*/ {
#ifdef RPC_BACKEND_LOCAL
    LOG(INFO) << "Using rpc backend: local";
    Singleton<RpcManager>::SetAllocated(new LocalRpcManager());
#else
    static_assert(false, "Requires rpc backend local to run oneflow in single processs");
#endif  // RPC_BACKEND_LOCAL
  } else /*multi process, multi machine*/ {
#ifdef RPC_BACKEND_GRPC
    LOG(INFO) << "Using rpc backend: gRPC";
    Singleton<RpcManager>::SetAllocated(new GrpcRpcManager());
#else
    UNIMPLEMENTED() << "To run distributed oneflow, you must enable at least one multi-node rpc "
                       "backend by adding cmake argument, for instance: -DRPC_BACKEND=GRPC";
#endif  // RPC_BACKEND_GRPC
  }
  CHECK_JUST(Singleton<RpcManager>::Get()->CreateServer());
  CHECK_JUST(Singleton<RpcManager>::Get()->Bootstrap());
  CHECK_JUST(Singleton<RpcManager>::Get()->CreateClient());
  Singleton<ResourceDesc, ForEnv>::New(GetDefaultResource(env_proto),
                                       GlobalProcessCtx::NumOfProcessPerNode());
  Singleton<ResourceDesc, ForSession>::New(GetDefaultResource(env_proto),
                                           GlobalProcessCtx::NumOfProcessPerNode());
  Singleton<hardware::NodeDeviceDescriptorManager>::SetAllocated(
      new hardware::NodeDeviceDescriptorManager());
  if (Singleton<ResourceDesc, ForEnv>::Get()->enable_debug_mode()) {
    Singleton<hardware::NodeDeviceDescriptorManager>::Get()->DumpSummary("devices");
  }
  Singleton<ep::DeviceManagerRegistry>::New();
  Singleton<ThreadPool>::New(Singleton<ResourceDesc, ForSession>::Get()->ComputeThreadPoolSize());
  SetCpuDeviceManagerNumThreads();
#ifdef WITH_CUDA
  Singleton<EagerNcclCommMgr>::New();
  Singleton<CudnnConvAlgoCache>::New();
  Singleton<CudnnHandlePool>::New();
  Singleton<embedding::EmbeddingManager>::New();
#endif
  Singleton<vm::VirtualMachineScope>::New(Singleton<ResourceDesc, ForSession>::Get()->resource());
#ifdef __linux__
  Singleton<EpollCommNet>::New();
  Singleton<Transport>::New();
  if (Singleton<ResourceDesc, ForSession>::Get()->process_ranks().size() > 1) {
    Singleton<CommNet>::SetAllocated(Singleton<EpollCommNet>::Get());
  }
#endif  // __linux__
  {
    std::vector<std::shared_ptr<KernelObserver>> kernel_observers;
    if (ParseBooleanFromEnv("ONEFLOW_DEBUG_KERNEL_SYNC_CHECK", false)) {
      LOG(WARNING)
          << "Environment variable ONEFLOW_DEBUG_KERNEL_SYNC_CHECK has been set to a truthy "
             "value, it will impact performance";
      kernel_observers.emplace_back(new SyncCheckKernelObserver());
    }
    if (!ParseBooleanFromEnv("ONEFLOW_KERNEL_DISABLE_BLOB_ACCESS_CHECKER", true)) {
      kernel_observers.emplace_back(new BlobAccessCheckerKernelObserver());
    }
    kernel_observers.emplace_back(new ProfilerKernelObserver());
    Singleton<KernelObserver>::SetAllocated(new ChainKernelObserver(kernel_observers));
  }
  TensorBufferPool::New();
  return Maybe<void>::Ok();
}

EnvGlobalObjectsScope::~EnvGlobalObjectsScope() {
  VLOG(2) << "Try to close env global objects scope." << std::endl;
  OF_ENV_BARRIER();
  if (is_normal_exit_.has_value() && !CHECK_JUST(is_normal_exit_)) { return; }
  TensorBufferPool::Delete();
  Singleton<KernelObserver>::Delete();
#ifdef __linux__
  if (Singleton<ResourceDesc, ForSession>::Get()->process_ranks().size() > 1) {
    if (Singleton<EpollCommNet>::Get() != dynamic_cast<EpollCommNet*>(Singleton<CommNet>::Get())) {
      Singleton<CommNet>::Delete();
    }
  }
  Singleton<Transport>::Delete();
  Singleton<EpollCommNet>::Delete();
#endif  // __linux__
  Singleton<vm::VirtualMachineScope>::Delete();
#ifdef WITH_CUDA
  Singleton<embedding::EmbeddingManager>::Delete();
  Singleton<CudnnConvAlgoCache>::Delete();
  Singleton<CudnnHandlePool>::Delete();
  Singleton<EagerNcclCommMgr>::Delete();
#endif
  Singleton<ThreadPool>::Delete();
  Singleton<ep::DeviceManagerRegistry>::Delete();
  if (Singleton<ResourceDesc, ForSession>::Get() != nullptr) {
    Singleton<ResourceDesc, ForSession>::Delete();
  }
  Singleton<ResourceDesc, ForEnv>::Delete();
  Singleton<hardware::NodeDeviceDescriptorManager>::Delete();
  CHECK_NOTNULL(Singleton<CtrlClient>::Get());
  CHECK_NOTNULL(Singleton<EnvDesc>::Get());
  Singleton<RpcManager>::Delete();
  Singleton<ProcessCtx>::Delete();
  Singleton<EnvDesc>::Delete();
  ClearAllSymbol();
  ClearAllBackwardPassScope();
  if (Singleton<EnvGlobalObjectsScope>::Get() != nullptr) {
    Singleton<EnvGlobalObjectsScope>::SetAllocated(nullptr);
  }
  VLOG(2) << "Finish closing env global objects scope." << std::endl;
  google::ShutdownGoogleLogging();
}

Maybe<void> InitRDMA() {
#ifdef __linux__
  if (Singleton<ResourceDesc, ForSession>::Get()->process_ranks().size() > 1) {
#if defined(WITH_RDMA) && defined(OF_PLATFORM_POSIX)
    if (CommNetIBEnabled()) {
      if (Singleton<IBVerbsCommNet>::Get() == nullptr) {
        Singleton<IBVerbsCommNet>::New();
        Singleton<CommNet>::SetAllocated(Singleton<IBVerbsCommNet>::Get());
      } else {
        LOG(INFO) << "Skip init RDMA because RDMA is already initialized!";
      }
    } else {
      LOG(WARNING) << "Skip init RDMA because RDMA is unavailable!";
    }
#else
    LOG(WARNING) << "Skip init RDMA because RDMA is not compiled!";
#endif  // WITH_RDMA && OF_PLATFORM_POSIX
  } else {
    LOG(INFO) << "Skip init RDMA because only one process in this group!";
  }
#endif  // __linux__
  return Maybe<void>::Ok();
}

Maybe<bool> RDMAIsInitialized() {
#if defined(WITH_RDMA) && defined(OF_PLATFORM_POSIX)
  return Singleton<IBVerbsCommNet>::Get() != nullptr;
#else
  return false;
#endif  // WITH_RDMA && OF_PLATFORM_POSIX
}

Maybe<void> DestoryRDMA() {
#if defined(WITH_RDMA) && defined(OF_PLATFORM_POSIX)
  if (JUST(RDMAIsInitialized())) {
    CHECK_NOTNULL(Singleton<IBVerbsCommNet>::Get());
    CHECK_NOTNULL(Singleton<CommNet>::Get());
    Singleton<IBVerbsCommNet>::Delete();
    if (Singleton<EpollCommNet>::Get()) {
      Singleton<CommNet>::SetAllocated(Singleton<EpollCommNet>::Get());
    }
  }
#endif  // WITH_RDMA && OF_PLATFORM_POSIX
  return Maybe<void>::Ok();
}

}  // namespace oneflow
