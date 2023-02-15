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
        parallel_desc_(ctx->parallel_desc()),
        this_parallel_id_(ctx->parallel_ctx().parallel_id()),
        num_ranks_(-1),
        comm_key_("InvalidKey") {
    if (ctx->op_conf().has_stream_name_hint()) { stream_name_ = ctx->op_conf().stream_name_hint(); }
    nccl_num_ = ctx->input_size("in");
    CHECK_EQ(nccl_num_, ctx->output_size("out"));
    src_split_axis_list_.resize(nccl_num_, -1);
    dst_split_axis_list_.resize(nccl_num_, -1);

    ParseSrcDstSplitAxisFromNdSbpStrAndNcclType(
        ctx->Attr<std::vector<std::string>>("src_nd_sbp_str_list"),
        ctx->Attr<std::vector<std::string>>("dst_nd_sbp_str_list"),
        ctx->Attr<std::vector<std::string>>("nccl_type_list"));
  }
  ~NcclLogicalFusionKernelState() override = default;

  ncclComm_t comm() {
    if (!is_init_) { Init(); }
    return comm_;
  }

  int64_t num_ranks() {
    if (!is_init_) { Init(); }
    return num_ranks_;
  }

  const std::string& stream_name() const { return stream_name_; }
  int64_t src_split_axis(int64_t i) const {
    CHECK_LT(i, src_split_axis_list_.size());
    return src_split_axis_list_.at(i);
  }
  int64_t dst_split_axis(int64_t i) const {
    CHECK_LT(i, dst_split_axis_list_.size());
    return dst_split_axis_list_.at(i);
  }

  int32_t nccl_num() const { return nccl_num_; }

 private:
  void Init() {
    CHECK(!is_init_);
    std::set<std::pair<int64_t, int64_t>> device_set;
    const Shape& hierarchy = *parallel_desc_.hierarchy();

    if (hierarchy.NumAxes() == 1) {
      FOR_RANGE(int64_t, parallel_id, 0, parallel_desc_.parallel_num()) {
        int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
        int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
        device_set.emplace(std::make_pair(machine_id, device_id));
      }
    } else if (hierarchy.NumAxes() == 2) {
      CHECK(comm_key_ == "SameDim0" || comm_key_ == "SameDim1");
      if (comm_key_ == "SameDim0") {
        const int64_t num_groups = hierarchy.At(0);
        const int64_t group_size = hierarchy.At(1);
        CHECK_EQ(num_groups * group_size, parallel_desc_.parallel_num());
        const int64_t this_group_begin_parallel_id = this_parallel_id_ / group_size * group_size;
        CHECK_EQ(this_group_begin_parallel_id % group_size, 0);
        CHECK_LE(this_group_begin_parallel_id + group_size, parallel_desc_.parallel_num());
        for (int64_t id_in_group = 0; id_in_group < group_size; ++id_in_group) {
          const int64_t parallel_id = this_group_begin_parallel_id + id_in_group;
          const int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
          const int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
          device_set.emplace(std::make_pair(machine_id, device_id));
        }
        num_ranks_ = group_size;
      } else if (comm_key_ == "SameDim1") {
        const int64_t group_size = hierarchy.At(0);
        const int64_t num_groups = hierarchy.At(1);
        CHECK_EQ(num_groups * group_size, parallel_desc_.parallel_num());
        const int64_t this_group_begin_parallel_id = this_parallel_id_ % num_groups;
        CHECK_LT(this_group_begin_parallel_id + (group_size - 1) * num_groups,
                 parallel_desc_.parallel_num());
        for (int64_t id_in_group = 0; id_in_group < group_size; ++id_in_group) {
          const int64_t parallel_id = this_group_begin_parallel_id + (id_in_group * num_groups);
          const int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
          const int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
          device_set.emplace(std::make_pair(machine_id, device_id));
        }
      } else {
        UNIMPLEMENTED();
      }
    } else {
      UNIMPLEMENTED();
    }

    EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerNcclCommMgr>::Get());
    comm_ = comm_mgr->GetCommForDeviceAndStreamName(device_set, stream_name_);
    is_init_ = true;
  }

  void UpdateOrCheckEqCommKey(const std::string& val) {
    if (comm_key_ == "InvalidKey") {
      comm_key_ = val;
    } else {
      CHECK_EQ(comm_key_, val);
    }
  }

  void ParseSrcDstSplitAxisFromNdSbpStrAndNcclType(
      const std::vector<std::string>& src_nd_sbp_str_list,
      const std::vector<std::string>& dst_nd_sbp_str_list,
      const std::vector<std::string>& nccl_type_list) {
    CHECK_EQ(src_nd_sbp_str_list.size(), nccl_num_);
    CHECK_EQ(dst_nd_sbp_str_list.size(), nccl_num_);
    CHECK_EQ(nccl_type_list.size(), nccl_num_);

    for (int32_t i = 0; i < nccl_num_; ++i) {
      NdSbp src_nd_sbp;
      NdSbp dst_nd_sbp;
      CHECK(ParseNdSbpFromLongString(src_nd_sbp_str_list.at(i), &src_nd_sbp));
      CHECK(ParseNdSbpFromLongString(dst_nd_sbp_str_list.at(i), &dst_nd_sbp));
      const std::string& nccl_type = nccl_type_list.at(i);
      UpdateOrCheckEqCommKey(GetCommKeyFromNcclType(nccl_type));
      if (nccl_type == "_nccl_logical_all_gather_noncontinuous") {
        CHECK_EQ(src_nd_sbp.sbp_parallel_size(), 1);
        src_split_axis_list_.at(i) = src_nd_sbp.sbp_parallel(0).split_parallel().axis();
        CHECK(src_nd_sbp.sbp_parallel(0).has_split_parallel());
      } else if (nccl_type == "_nccl_logical_reduce_scatter_noncontinuous") {
        CHECK_EQ(dst_nd_sbp.sbp_parallel_size(), 1);
        CHECK(dst_nd_sbp.sbp_parallel(0).has_split_parallel());
        dst_split_axis_list_.at(i) = dst_nd_sbp.sbp_parallel(0).split_parallel().axis();
      } else if (nccl_type == "_nccl_logical_s2s") {
        CHECK_EQ(src_nd_sbp.sbp_parallel_size(), 1);
        CHECK_EQ(dst_nd_sbp.sbp_parallel_size(), 1);
        CHECK(src_nd_sbp.sbp_parallel(0).has_split_parallel());
        CHECK(dst_nd_sbp.sbp_parallel(0).has_split_parallel());
        src_split_axis_list_.at(i) = src_nd_sbp.sbp_parallel(0).split_parallel().axis();
        dst_split_axis_list_.at(i) = dst_nd_sbp.sbp_parallel(0).split_parallel().axis();
        CHECK_NE(src_split_axis_list_.at(i), dst_split_axis_list_.at(i));
      } else if (nccl_type == "_nccl_logical_2D_same_dim0_all_gather_noncontinuous") {
        CHECK_EQ(src_nd_sbp.sbp_parallel_size(), 2);
        CHECK(src_nd_sbp.sbp_parallel(1).has_split_parallel());
        src_split_axis_list_.at(i) = src_nd_sbp.sbp_parallel(1).split_parallel().axis();
      } else if (nccl_type == "_nccl_logical_2D_same_dim0_all2all") {
        CHECK_EQ(src_nd_sbp.sbp_parallel_size(), 2);
        CHECK_EQ(dst_nd_sbp.sbp_parallel_size(), 2);
        CHECK(src_nd_sbp.sbp_parallel(1).has_split_parallel());
        CHECK(dst_nd_sbp.sbp_parallel(1).has_split_parallel());
        src_split_axis_list_.at(i) = src_nd_sbp.sbp_parallel(1).split_parallel().axis();
        dst_split_axis_list_.at(i) = dst_nd_sbp.sbp_parallel(1).split_parallel().axis();
        CHECK_NE(src_split_axis_list_.at(i), dst_split_axis_list_.at(i));
      }
    }
  }

  bool is_init_;
  std::string stream_name_;
  ParallelDesc parallel_desc_;
  int64_t this_parallel_id_;
  int64_t num_ranks_;
  std::string comm_key_;
  int32_t nccl_num_;
  std::vector<int64_t> src_split_axis_list_;
  std::vector<int64_t> dst_split_axis_list_;
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
  LOG(INFO) << "ccdebuglog: kernel " << ctx->op_name() << " nccl_num: " << kernel_state->nccl_num()
            << " stream_name: " << kernel_state->stream_name();
  for (int32_t i = 0; i < kernel_state->nccl_num(); ++i) {
    LOG(INFO) << " === i = " << i << " , src_split_axis = " << kernel_state->src_split_axis(i)
              << " , dst_split_axis = " << kernel_state->dst_split_axis(i);
  }
}

size_t InferTmpBufferSize(user_op::InferContext* ctx) { return 0; }

REGISTER_USER_KERNEL("_nccl_logical_fusion")
    .SetCreateFn<NcclLogicalFusion>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCUDA)
    .SetInferTmpSizeFn(InferTmpBufferSize);
}  // namespace

}  // namespace oneflow

#endif  // WITH_CUDA && NCCL_VERSION_CODE > 2700
