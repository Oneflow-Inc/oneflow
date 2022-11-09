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
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/include/primitive/permute.h"

namespace oneflow {

class CollectiveBoxingPackKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingPackKernel);
  CollectiveBoxingPackKernel() = default;
  ~CollectiveBoxingPackKernel() override = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(KernelContext* ctx) const override;
};

void CollectiveBoxingPackKernel::ForwardDataContent(KernelContext* ctx) const {
  const Blob* in = ctx->BnInOp2Blob("in");
  Blob* out = ctx->BnInOp2Blob("out");
  const CollectiveBoxingPackOpConf& pack_conf = this->op_conf().collective_boxing_pack_conf();
  const int64_t num_ranks = pack_conf.num_ranks();
  const Shape logical_shape(pack_conf.logical_shape());
  const bool need_transpose = !((pack_conf.dst_sbp_parallel().has_split_parallel()
                                 && pack_conf.dst_sbp_parallel().split_parallel().axis() == 0)
                                || pack_conf.dst_sbp_parallel().has_broadcast_parallel()
                                || pack_conf.dst_sbp_parallel().has_partial_sum_parallel());
  if (need_transpose) {
    const int64_t dst_split_axis = pack_conf.dst_sbp_parallel().split_parallel().axis();
    DimVector transpose_in_dim_vec = logical_shape.dim_vec();
    if (pack_conf.src_sbp_parallel().has_split_parallel()) {
      const int64_t src_split_axis = pack_conf.src_sbp_parallel().split_parallel().axis();
      transpose_in_dim_vec[src_split_axis] = transpose_in_dim_vec.at(src_split_axis) / num_ranks;
    }
    CHECK_EQ(transpose_in_dim_vec.at(dst_split_axis) % num_ranks, 0);
    transpose_in_dim_vec[dst_split_axis] = transpose_in_dim_vec.at(dst_split_axis) / num_ranks;
    transpose_in_dim_vec.insert(transpose_in_dim_vec.begin() + dst_split_axis, num_ranks);
    std::vector<int32_t> perm;
    perm.emplace_back(dst_split_axis);
    FOR_RANGE(int64_t, i, 0, transpose_in_dim_vec.size()) {
      if (i != dst_split_axis) { perm.emplace_back(i); }
    }
    auto transpose = ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(
        ctx->stream()->device_type(), transpose_in_dim_vec.size());
    CHECK(transpose);
    transpose->Launch(ctx->stream(), in->data_type(), transpose_in_dim_vec.size(),
                      transpose_in_dim_vec.data(), in->dptr(), perm.data(), out->mut_dptr());
  } else {
    AutoMemcpy(ctx->stream(), out, in);
  }
}

REGISTER_KERNEL(OperatorConf::kCollectiveBoxingPackConf, CollectiveBoxingPackKernel);

}  // namespace oneflow
