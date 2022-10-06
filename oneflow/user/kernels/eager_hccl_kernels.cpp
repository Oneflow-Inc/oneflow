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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/hccl_util.h"
#include "oneflow/core/job/eager_hccl_comm_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/ep/include/primitive/permute.h"
#include "oneflow/core/ep/npu/npu_stream.h"
#include "hccl/hccl.h"

namespace oneflow {
namespace {

class EagerHcclOpKernelCache final : public user_op::OpKernelCache {
 public:
  explicit EagerHcclOpKernelCache(user_op::KernelCacheContext* ctx) { Init(ctx); }
  ~EagerHcclOpKernelCache() override = default;

  Symbol<ParallelDesc> parallel_desc() const { return parallel_desc_; }
  HcclComm comm() const { return comm_; }

 private:
  void Init(user_op::KernelCacheContext* ctx) {
    const std::string& parallel_conf_txt = ctx->Attr<std::string>("parallel_conf");
    ParallelConf parallel_conf;
    std::set<std::pair<int64_t, int64_t>> device_set;
    CHECK(TxtString2PbMessage(parallel_conf_txt, &parallel_conf));
    parallel_desc_ = SymbolOf(ParallelDesc(parallel_conf));
    FOR_RANGE(int64_t, parallel_id, 0, parallel_desc_->parallel_num()) {
      int64_t machine_id = CHECK_JUST(parallel_desc_->MachineId4ParallelId(parallel_id));
      int64_t device_id = CHECK_JUST(parallel_desc_->DeviceId4ParallelId(parallel_id));
      device_set.emplace(std::make_pair(machine_id, device_id));
    }
    comm_ = CHECK_NOTNULL(Singleton<EagerHcclCommMgr>::Get())->GetCommForDevice(device_set);
  }

  Symbol<ParallelDesc> parallel_desc_;
  HcclComm comm_{};
};

void InitEagerHcclOpKernelCache(user_op::KernelCacheContext* ctx,
                                std::shared_ptr<user_op::OpKernelCache>* cache_ptr) {
  // NOTE(jianhao): the cache only depends on parallel_conf, and the kernel is singleton
  // once parallel_conf is determined, so only init the cache at the first time.
  if (*cache_ptr == nullptr) { *cache_ptr = std::make_shared<EagerHcclOpKernelCache>(ctx); }
}

}

class EagerHcclBroadcastKernel final : public user_op::OpKernel {
 public:
  EagerHcclBroadcastKernel() = default;
  ~EagerHcclBroadcastKernel() override = default;

  void InitOpKernelCacheWithFlags(
      user_op::KernelCacheContext* ctx, int8_t flag,
      std::shared_ptr<user_op::OpKernelCache>* cache_ptr) const override {
    InitEagerHcclOpKernelCache(ctx, cache_ptr);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {

    auto* kernel_cache = dynamic_cast<const EagerHcclOpKernelCache*>(cache);
    CHECK(kernel_cache != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t root = ctx->Attr<int64_t>("root");
    int64_t dev_id = GlobalProcessCtx::LocalRank(root);
    int64_t hccl_root =
        CHECK_JUST(kernel_cache->parallel_desc()->ParallelId4MachineDeviceId(root, dev_id));

    // const void* in_ptr = nullptr;
    // if (GlobalProcessCtx::Rank() == root) {
    //   CHECK_EQ(in->shape_view(), out->shape_view());
    //   CHECK_EQ(in->data_type(), out->data_type());
    //   in_ptr = in->dptr();
    // }
    
    // 如果当前是root, 那么out与in的数据地址是一个
    OF_HCCL_CHECK(HcclBroadcast(out->mut_dptr(), out->shape_view().elem_cnt(),
                                GetHcclDataType(out->data_type()), hccl_root, kernel_cache->comm(),
                                ctx->stream()->As<ep::NpuStream>()->npu_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_nccl_broadcast")
    .SetCreateFn<EagerHcclBroadcastKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU);

class EagerHcclAllReduceKernel final : public user_op::OpKernel {
 public:
  EagerHcclAllReduceKernel() = default;
  ~EagerHcclAllReduceKernel() override = default;

  void InitOpKernelCacheWithFlags(
      user_op::KernelCacheContext* ctx, int8_t flag,
      std::shared_ptr<user_op::OpKernelCache>* cache_ptr) const override {
    InitEagerHcclOpKernelCache(ctx, cache_ptr);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {

    auto* kernel_cache = dynamic_cast<const EagerHcclOpKernelCache*>(cache);
    CHECK(kernel_cache != nullptr);
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->shape_view(), out->shape_view());
    CHECK_EQ(in->data_type(), out->data_type());
    HcclReduceOp reduce_type = HCCL_REDUCE_SUM;
    if (in->data_type() == kBool) { reduce_type = HCCL_REDUCE_MAX; }
    OF_HCCL_CHECK(HcclAllReduce(in->mut_dptr(), out->mut_dptr(), in->shape_view().elem_cnt(),
                                GetHcclDataType(in->data_type()), reduce_type, kernel_cache->comm(),
                                ctx->stream()->As<ep::NpuStream>()->npu_stream()));

  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_nccl_all_reduce")
    .SetCreateFn<EagerHcclAllReduceKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU);


} // namespace oneflow