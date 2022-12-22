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

class CollectiveBoxingUnpackKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingUnpackKernel);
  CollectiveBoxingUnpackKernel() = default;
  ~CollectiveBoxingUnpackKernel() override = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(KernelContext* ctx) const override;
};

void CollectiveBoxingUnpackKernel::ForwardDataContent(KernelContext* ctx) const {
  const Blob* in = ctx->BnInOp2Blob("in");
  Blob* out = ctx->BnInOp2Blob("out");
  const CollectiveBoxingUnpackOpConf& unpack_conf = this->op_conf().collective_boxing_unpack_conf();
  const int64_t num_ranks = unpack_conf.num_ranks();
  const Shape logical_shape(unpack_conf.logical_shape());
  // skip 0size tensor boxing
  if (logical_shape.elem_cnt() == 0) { return; }
  const bool need_transpose = !((unpack_conf.src_sbp_parallel().has_split_parallel()
                                 && unpack_conf.src_sbp_parallel().split_parallel().axis() == 0)
                                || unpack_conf.src_sbp_parallel().has_broadcast_parallel()
                                || unpack_conf.src_sbp_parallel().has_partial_sum_parallel());
  if (need_transpose) {
    const int64_t src_split_axis = unpack_conf.src_sbp_parallel().split_parallel().axis();
    DimVector transpose_in_dim_vec = logical_shape.dim_vec();
    CHECK_EQ(transpose_in_dim_vec.at(src_split_axis) % num_ranks, 0);
    transpose_in_dim_vec[src_split_axis] = transpose_in_dim_vec.at(src_split_axis) / num_ranks;
    if (unpack_conf.dst_sbp_parallel().has_split_parallel()) {
      const int64_t dst_split_axis = unpack_conf.dst_sbp_parallel().split_parallel().axis();
      CHECK_EQ(transpose_in_dim_vec.at(dst_split_axis) % num_ranks, 0);
      transpose_in_dim_vec[dst_split_axis] = transpose_in_dim_vec.at(dst_split_axis) / num_ranks;
    }
    transpose_in_dim_vec.insert(transpose_in_dim_vec.begin(), num_ranks);
    std::vector<int32_t> perm;
    FOR_RANGE(int64_t, i, 1, transpose_in_dim_vec.size()) { perm.emplace_back(i); }
    perm.insert(perm.begin() + src_split_axis, 0);
    auto transpose = ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(
        ctx->stream()->device_type(), transpose_in_dim_vec.size());
    CHECK(transpose);
    transpose->Launch(ctx->stream(), in->data_type(), transpose_in_dim_vec.size(),
                      transpose_in_dim_vec.data(), in->dptr(), perm.data(), out->mut_dptr());
  } else {
    AutoMemcpy(ctx->stream(), out, in);
  }
}

REGISTER_KERNEL(OperatorConf::kCollectiveBoxingUnpackConf, CollectiveBoxingUnpackKernel);

}  // namespace oneflow
