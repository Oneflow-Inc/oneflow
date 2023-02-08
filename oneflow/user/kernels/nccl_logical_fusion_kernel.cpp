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

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/ep/include/primitive/permute.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/user/ops/nccl_logical_util.h"

#if defined(WITH_CUDA) && NCCL_VERSION_CODE > 2700

namespace oneflow {

namespace {

class NcclLogicalFusionKernelState : public user_op::OpKernelState {
 public:
  explicit NcclLogicalFusionKernelState(user_op::KernelInitContext* ctx)
      : is_init_(false),
        stream_name_(EagerNcclCommMgr::kDefaultStreamName),
        parallel_desc_(ctx->parallel_desc()) {
    if (ctx->op_conf().has_stream_name_hint()) { stream_name_ = ctx->op_conf().stream_name_hint(); }
  }
  ~NcclLogicalFusionKernelState() override = default;

  ncclComm_t comm() {
    if (!is_init_) {
      std::set<std::pair<int64_t, int64_t>> device_set;
      FOR_RANGE(int64_t, parallel_id, 0, parallel_desc_.parallel_num()) {
        int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
        int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
        device_set.emplace(std::make_pair(machine_id, device_id));
      }
      EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerNcclCommMgr>::Get());
      comm_ = comm_mgr->GetCommForDeviceAndStreamName(device_set, stream_name_);
      is_init_ = true;
    }
    return comm_;
  }

  const std::string& stream_name() const { return stream_name_; }

 private:
  bool is_init_;
  std::string stream_name_;
  ParallelDesc parallel_desc_;
  ncclComm_t comm_{};
};

class NcclLogicalFusion final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclLogicalFusion);
  NcclLogicalFusion() = default;
  ~NcclLogicalFusion() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogicalFusionKernelState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  bool IsKernelLaunchSynchronized() const override {
    EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerNcclCommMgr>::Get());
    return comm_mgr->IsAsyncLaunchNcclLogicalKernel();
  }
};

void NcclLogicalFusion::Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
                                const user_op::OpKernelCache*) const {
  auto* kernel_state = dynamic_cast<NcclLogicalFusionKernelState*>(state);
  CHECK_NOTNULL(kernel_state);
}

size_t InferTmpBufferSize(user_op::InferContext* ctx) { return 0; }

REGISTER_USER_KERNEL("_nccl_logical_fusion")
    .SetCreateFn<NcclLogicalFusion>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCUDA)
    .SetInferTmpSizeFn(InferTmpBufferSize);
}  // namespace

}  // namespace oneflow

#endif  // WITH_CUDA && NCCL_VERSION_CODE > 2700
