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

namespace oneflow {

class EagerNcclAllReduceKernel final : public user_op::OpKernel {
 public:
  EagerNcclAllReduceKernel() = default;
  ~EagerNcclAllReduceKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->shape(), out->shape());
    CHECK_EQ(in->data_type(), out->data_type());
    std::set<std::pair<int64_t, int64_t>> device_set;
    const std::string& parallel_conf_txt = ctx->Attr<std::string>("parallel_conf");
    ParallelConf parallel_conf{};
    CHECK(TxtString2PbMessage(parallel_conf_txt, &parallel_conf));
    const ParallelDesc parallel_desc(parallel_conf);
    FOR_RANGE(int64_t, parallel_id, 0, parallel_desc.parallel_num()) {
      int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(parallel_id));
      int64_t device_id = CHECK_JUST(parallel_desc.DeviceId4ParallelId(parallel_id));
      device_set.emplace(std::make_pair(machine_id, device_id));
    }
    ncclComm_t comm = CHECK_NOTNULL(Global<EagerNcclCommMgr>::Get())->GetCommForDevice(device_set);
    OF_NCCL_CHECK(ncclAllReduce(in->dptr(), out->mut_dptr(), in->shape().elem_cnt(),
                                GetNcclDataType(in->data_type()), ncclSum, comm,
                                ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_nccl_all_reduce")
    .SetCreateFn<EagerNcclAllReduceKernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");

}  // namespace oneflow
