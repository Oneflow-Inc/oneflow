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
#include "oneflow/user/kernels/collective_communication/include/all_reduce.h"
#include "oneflow/user/kernels/collective_communication/include/all_gather.h"
#include "oneflow/user/kernels/collective_communication/include/all_to_all.h"

#if defined(WITH_CUDA) && NCCL_VERSION_CODE > 2700

namespace oneflow {

namespace {

class NcclLogical2DSameDim0KernelCommState : public user_op::OpKernelState {
 public:
  explicit NcclLogical2DSameDim0KernelCommState(user_op::KernelInitContext* ctx)
      : is_init_(false),
        stream_name_(EagerNcclCommMgr::kDefaultStreamName),
        parallel_desc_(ctx->parallel_desc()),
        this_parallel_id_(ctx->parallel_ctx().parallel_id()) {
    if (ctx->op_conf().has_stream_name_hint()) { stream_name_ = ctx->op_conf().stream_name_hint(); }
  }
  ~NcclLogical2DSameDim0KernelCommState() override = default;

  ccl::CclComm ccl_comm() {
    if (!is_init_) { Init(); }
    return ccl_comm_;
  }

  int64_t num_ranks() {
    if (!is_init_) { Init(); }
    return num_ranks_;
  }

  const std::string& stream_name() const { return stream_name_; }

 private:
  void Init() {
    CHECK(!is_init_);
    std::set<std::pair<int64_t, int64_t>> device_set;
    const Shape& hierarchy = *parallel_desc_.hierarchy();
    CHECK_EQ(hierarchy.NumAxes(), 2);
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
    EagerCclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerCclCommMgr>::Get());
    ccl_comm_ = comm_mgr->GetCclCommForDeviceAndStreamName(device_set, stream_name_);
    num_ranks_ = group_size;
    is_init_ = true;
  }

  bool is_init_;
  std::string stream_name_;
  ParallelDesc parallel_desc_;
  int64_t this_parallel_id_;
  int64_t num_ranks_{};
  ccl::CclComm ccl_comm_{};
};

class NcclLogical2DSameDim0AllGatherNoncontinuousKernelState
    : public NcclLogical2DSameDim0KernelCommState {
 public:
  explicit NcclLogical2DSameDim0AllGatherNoncontinuousKernelState(user_op::KernelInitContext* ctx)
      : NcclLogical2DSameDim0KernelCommState(ctx), src_split_axis_(-1) {}
  ~NcclLogical2DSameDim0AllGatherNoncontinuousKernelState() override = default;

  int64_t src_split_axis() const { return src_split_axis_; }
  void set_src_split_axis(int64_t split_axis) { src_split_axis_ = split_axis; }

 private:
  int64_t src_split_axis_;
};

class NcclLogical2DSameDim0All2AllKernelState : public NcclLogical2DSameDim0KernelCommState {
 public:
  explicit NcclLogical2DSameDim0All2AllKernelState(user_op::KernelInitContext* ctx)
      : NcclLogical2DSameDim0KernelCommState(ctx), src_split_axis_(-1), dst_split_axis_(-1) {}
  ~NcclLogical2DSameDim0All2AllKernelState() override = default;

  int64_t src_split_axis() const { return src_split_axis_; }
  void set_src_split_axis(int64_t split_axis) { src_split_axis_ = split_axis; }
  int64_t dst_split_axis() const { return dst_split_axis_; }
  void set_dst_split_axis(int64_t split_axis) { dst_split_axis_ = split_axis; }

 private:
  int64_t src_split_axis_;
  int64_t dst_split_axis_;
};

class NcclLogical2DSameDim0AllReduce final : public user_op::OpKernel {
 public:
  NcclLogical2DSameDim0AllReduce() = default;
  ~NcclLogical2DSameDim0AllReduce() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical2DSameDim0KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* comm_state = dynamic_cast<NcclLogical2DSameDim0KernelCommState*>(state);
    CHECK(comm_state != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->shape_view(), out->shape_view());
    CHECK_EQ(in->data_type(), out->data_type());
    VLOG(3) << "[NcclLogical2D][SameDim0AllReduce] " << comm_state->stream_name() << " "
            << ctx->op_name() << std::endl;
    ccl::ReduceType ccl_reduce_type = ccl::ReduceType::kSum;
    if (in->data_type() == DataType::kBool) { ccl_reduce_type = ccl::ReduceType::kMax; }
    ccl::CclComm ccl_comm = comm_state->ccl_comm();
    std::unique_ptr<ccl::AllReduce> ccl_all_reduce =
        ccl::NewCollectiveCommunication<ccl::AllReduce>(ctx->stream()->device_type(),
                                                        in->data_type(), ccl_reduce_type);
    ccl_all_reduce->Launch(ctx->stream(), in->dptr(), out->mut_dptr(), in->shape_view().elem_cnt(),
                           ccl_comm);
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  bool IsKernelLaunchSynchronized() const override {
    const EagerCclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerCclCommMgr>::Get());
    return comm_mgr->IsAsyncLaunchCclLogicalKernel();
  }
};

class NcclLogical2DSameDim0AllGather final : public user_op::OpKernel {
 public:
  NcclLogical2DSameDim0AllGather() = default;
  ~NcclLogical2DSameDim0AllGather() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical2DSameDim0KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* comm_state = dynamic_cast<NcclLogical2DSameDim0KernelCommState*>(state);
    CHECK(comm_state != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->data_type(), out->data_type());
    const int64_t num_ranks = comm_state->num_ranks();
    CHECK_EQ(in->shape_view().elem_cnt() * num_ranks, out->shape_view().elem_cnt());
    VLOG(3) << "[NcclLogical2D][SameDim0AllGather] " << comm_state->stream_name() << " "
            << ctx->op_name() << std::endl;

    std::unique_ptr<ccl::AllGather> ccl_all_gather =
        ccl::NewCollectiveCommunication<ccl::AllGather>(ctx->stream()->device_type(),
                                                        in->data_type());
    ccl::CclComm ccl_comm = comm_state->ccl_comm();
    ccl_all_gather->Launch(ctx->stream(), in->dptr(), out->mut_dptr(), in->shape_view().elem_cnt(),
                           ccl_comm);
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  bool IsKernelLaunchSynchronized() const override {
    const EagerCclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerCclCommMgr>::Get());
    return comm_mgr->IsAsyncLaunchCclLogicalKernel();
  }
};

template<typename T>
class NcclLogical2DSameDim0AllGatherNoncontinuous final : public user_op::OpKernel {
 public:
  NcclLogical2DSameDim0AllGatherNoncontinuous() = default;
  ~NcclLogical2DSameDim0AllGatherNoncontinuous() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    auto state = std::make_shared<NcclLogical2DSameDim0AllGatherNoncontinuousKernelState>(ctx);
    NdSbp src_nd_sbp;
    CHECK_JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_reduced_nd_sbp", &src_nd_sbp));
    CHECK_EQ(src_nd_sbp.sbp_parallel_size(), 2);
    CHECK(src_nd_sbp.sbp_parallel(1).has_split_parallel());
    state->set_src_split_axis(src_nd_sbp.sbp_parallel(1).split_parallel().axis());
    return state;
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state =
        dynamic_cast<NcclLogical2DSameDim0AllGatherNoncontinuousKernelState*>(state);
    CHECK_NOTNULL(kernel_state);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t dtype_size = GetSizeOfDataType(in->data_type());
    int64_t data_size = GetCudaAlignedSize(out->shape_view().elem_cnt() * dtype_size);
    void* unpack_from_ptr = tmp_buffer->mut_dptr();
    CHECK_EQ(tmp_buffer->shape_view().elem_cnt(), data_size);

    CHECK_EQ(in->data_type(), out->data_type());
    const int64_t num_ranks = kernel_state->num_ranks();
    const int64_t in_split_axis = kernel_state->src_split_axis();

    DimVector logical_shape_dim_vec;
    in->shape_view().ToDimVector(&logical_shape_dim_vec);
    logical_shape_dim_vec[in_split_axis] = logical_shape_dim_vec.at(in_split_axis) * num_ranks;

    VLOG(3) << "[NcclLogical2D][SameDim0AllGatherNoncontinuous] " << kernel_state->stream_name()
            << " " << ctx->op_name() << std::endl;

    // NOTE(chengcheng): Do AllGather
    CHECK_EQ(in->shape_view().elem_cnt() * num_ranks, out->shape_view().elem_cnt());

    std::unique_ptr<ccl::AllGather> ccl_all_gather =
        ccl::NewCollectiveCommunication<ccl::AllGather>(ctx->stream()->device_type(),
                                                        in->data_type());
    ccl::CclComm ccl_comm = kernel_state->ccl_comm();
    ccl_all_gather->Launch(ctx->stream(), in->dptr(), unpack_from_ptr, in->shape_view().elem_cnt(),
                           ccl_comm);

    CHECK_GT(in_split_axis, 0);
    // NOTE(chengcheng): Do unpack.
    DimVector unpack_from_dim_vec = logical_shape_dim_vec;
    CHECK_EQ(unpack_from_dim_vec.at(in_split_axis) % num_ranks, 0);
    unpack_from_dim_vec[in_split_axis] = unpack_from_dim_vec.at(in_split_axis) / num_ranks;
    unpack_from_dim_vec.insert(unpack_from_dim_vec.begin(), num_ranks);
    std::vector<int32_t> perm;
    FOR_RANGE(int64_t, i, 1, unpack_from_dim_vec.size()) { perm.emplace_back(i); }
    perm.insert(perm.begin() + in_split_axis, 0);

    auto transpose = ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(
        ctx->stream()->device_type(), unpack_from_dim_vec.size());
    CHECK(transpose);
    transpose->Launch(ctx->stream(), in->data_type(), unpack_from_dim_vec.size(),
                      unpack_from_dim_vec.data(), unpack_from_ptr, perm.data(), out->mut_dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  bool IsKernelLaunchSynchronized() const override {
    const EagerCclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerCclCommMgr>::Get());
    return comm_mgr->IsAsyncLaunchCclLogicalKernel();
  }
};

size_t Infer2DSameDim0AllGatherNoncontinuousKernelTmpBufferSize(user_op::InferContext* ctx) {
  const user_op::TensorDesc& out_tensor = ctx->OutputTensorDesc("out", 0);
  return GetCudaAlignedSize(out_tensor.shape().elem_cnt()
                            * GetSizeOfDataType(out_tensor.data_type()));
}

template<typename T>
class NcclLogical2DSameDim0All2All final : public user_op::OpKernel {
 public:
  NcclLogical2DSameDim0All2All() = default;
  ~NcclLogical2DSameDim0All2All() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    auto state = std::make_shared<NcclLogical2DSameDim0All2AllKernelState>(ctx);
    NdSbp src_nd_sbp;
    NdSbp dst_nd_sbp;
    CHECK_JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_reduced_nd_sbp", &src_nd_sbp));
    CHECK_JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_reduced_nd_sbp", &dst_nd_sbp));
    CHECK_EQ(src_nd_sbp.sbp_parallel_size(), 2);
    CHECK_EQ(dst_nd_sbp.sbp_parallel_size(), 2);
    CHECK(src_nd_sbp.sbp_parallel(1).has_split_parallel());
    CHECK(dst_nd_sbp.sbp_parallel(1).has_split_parallel());
    state->set_src_split_axis(src_nd_sbp.sbp_parallel(1).split_parallel().axis());
    state->set_dst_split_axis(dst_nd_sbp.sbp_parallel(1).split_parallel().axis());
    return state;
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* kernel_state = dynamic_cast<NcclLogical2DSameDim0All2AllKernelState*>(state);
    CHECK_NOTNULL(kernel_state);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    int64_t tmp_size = 0;
    const int64_t dtype_size = GetSizeOfDataType(in->data_type());
    int64_t data_size = GetCudaAlignedSize(in->shape_view().elem_cnt() * dtype_size);
    // NOTE(chengcheng): in (transpose)-> pack_to_ptr (all2all)-> unpack_from_ptr (transpose)-> out
    const char* pack_to_ptr = in->dptr<char>();
    char* unpack_from_ptr = out->mut_dptr<char>();
    if (tmp_buffer) { tmp_size = tmp_buffer->shape_view().elem_cnt(); }
    CHECK(tmp_size == 0 || tmp_size == data_size || tmp_size == data_size * 2);

    CHECK_EQ(in->data_type(), out->data_type());
    const int64_t num_ranks = kernel_state->num_ranks();
    CHECK_EQ(in->shape_view().elem_cnt(), out->shape_view().elem_cnt());
    const int64_t elem_cnt = in->shape_view().elem_cnt();
    const int64_t in_split_axis = kernel_state->src_split_axis();
    const int64_t out_split_axis = kernel_state->dst_split_axis();

    DimVector logical_shape_dim_vec;
    in->shape_view().ToDimVector(&logical_shape_dim_vec);
    logical_shape_dim_vec[in_split_axis] = logical_shape_dim_vec.at(in_split_axis) * num_ranks;

    VLOG(3) << "[NcclLogical2D][SameDim0All2All] " << kernel_state->stream_name() << " "
            << ctx->op_name() << std::endl;

    if (out_split_axis != 0) {
      // NOTE(chengcheng): Do pack. Need transpose in -> pack_to
      // pack use temp buffer offset: [0, data_size]
      pack_to_ptr = CHECK_NOTNULL(tmp_buffer)->dptr<char>();
      DimVector transpose_in_dim_vec = logical_shape_dim_vec;
      CHECK_EQ(transpose_in_dim_vec.at(in_split_axis) % num_ranks, 0);
      transpose_in_dim_vec[in_split_axis] = transpose_in_dim_vec.at(in_split_axis) / num_ranks;
      CHECK_EQ(transpose_in_dim_vec.at(out_split_axis) % num_ranks, 0);
      transpose_in_dim_vec[out_split_axis] = transpose_in_dim_vec.at(out_split_axis) / num_ranks;
      transpose_in_dim_vec.insert(transpose_in_dim_vec.begin() + out_split_axis, num_ranks);
      std::vector<int32_t> perm;
      perm.emplace_back(out_split_axis);
      FOR_RANGE(int64_t, i, 0, transpose_in_dim_vec.size()) {
        if (i != out_split_axis) { perm.emplace_back(i); }
      }
      auto transpose = ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(
          ctx->stream()->device_type(), transpose_in_dim_vec.size());
      CHECK(transpose);
      transpose->Launch(ctx->stream(), in->data_type(), transpose_in_dim_vec.size(),
                        transpose_in_dim_vec.data(), in->dptr(), perm.data(),
                        tmp_buffer->mut_dptr());
    }

    if (in_split_axis != 0) {
      // NOTE(chengcheng): Do unpack. Need transpose unpack_from -> out
      // unpack use temp buffer offset: [tmp_size - data_size, tmp_size]
      unpack_from_ptr = CHECK_NOTNULL(tmp_buffer)->mut_dptr<char>() + (tmp_size - data_size);
    }

    {
      // NOTE(chengcheng): Do S2S
      const int64_t elem_per_chunk = elem_cnt / num_ranks;
      std::unique_ptr<ccl::AllToAll> all_to_all = ccl::NewCollectiveCommunication<ccl::AllToAll>(
          ctx->stream()->device_type(), in->data_type(), in->data_type(), num_ranks);
      ccl::CclComm ccl_comm = kernel_state->ccl_comm();
      all_to_all->Launch(ctx->stream(), pack_to_ptr, elem_per_chunk, unpack_from_ptr,
                         elem_per_chunk, ccl_comm);
    }

    if (in_split_axis != 0) {
      // Do unpack.
      CHECK(unpack_from_ptr != out->mut_dptr<char>());
      DimVector unpack_from_dim_vec = logical_shape_dim_vec;
      CHECK_EQ(unpack_from_dim_vec.at(in_split_axis) % num_ranks, 0);
      unpack_from_dim_vec[in_split_axis] = unpack_from_dim_vec.at(in_split_axis) / num_ranks;
      CHECK_EQ(unpack_from_dim_vec.at(out_split_axis) % num_ranks, 0);
      unpack_from_dim_vec[out_split_axis] = unpack_from_dim_vec.at(out_split_axis) / num_ranks;
      unpack_from_dim_vec.insert(unpack_from_dim_vec.begin(), num_ranks);
      std::vector<int32_t> perm;
      FOR_RANGE(int64_t, i, 1, unpack_from_dim_vec.size()) { perm.emplace_back(i); }
      perm.insert(perm.begin() + in_split_axis, 0);
      auto transpose = ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(
          ctx->stream()->device_type(), unpack_from_dim_vec.size());
      CHECK(transpose);
      transpose->Launch(ctx->stream(), in->data_type(), unpack_from_dim_vec.size(),
                        unpack_from_dim_vec.data(), unpack_from_ptr, perm.data(), out->mut_dptr());
    }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  bool IsKernelLaunchSynchronized() const override {
    const EagerCclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerCclCommMgr>::Get());
    return comm_mgr->IsAsyncLaunchCclLogicalKernel();
  }
};

size_t Infer2DSameDim0All2AllKernelTmpBufferSize(user_op::InferContext* ctx) {
  size_t ret = 0;
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  size_t tensor_byte_size =
      GetCudaAlignedSize(in_tensor.shape().elem_cnt() * GetSizeOfDataType(in_tensor.data_type()));
  NdSbp src_nd_sbp;
  NdSbp dst_nd_sbp;
  CHECK_JUST(GetNcclLogicalNdSbpFromAttr(ctx, "src_reduced_nd_sbp", &src_nd_sbp));
  CHECK_JUST(GetNcclLogicalNdSbpFromAttr(ctx, "dst_reduced_nd_sbp", &dst_nd_sbp));
  CHECK_EQ(src_nd_sbp.sbp_parallel_size(), 2);
  CHECK_EQ(dst_nd_sbp.sbp_parallel_size(), 2);
  CHECK(src_nd_sbp.sbp_parallel(1).has_split_parallel());
  CHECK(dst_nd_sbp.sbp_parallel(1).has_split_parallel());
  if (src_nd_sbp.sbp_parallel(1).split_parallel().axis() != 0) { ret += tensor_byte_size; }
  if (dst_nd_sbp.sbp_parallel(1).split_parallel().axis() != 0) { ret += tensor_byte_size; }
  return ret;
}

class NcclLogical2DSameDim1KernelCommState final : public user_op::OpKernelState {
 public:
  explicit NcclLogical2DSameDim1KernelCommState(user_op::KernelInitContext* ctx)
      : is_init_(false),
        stream_name_(EagerNcclCommMgr::kDefaultStreamName),
        parallel_desc_(ctx->parallel_desc()),
        this_parallel_id_(ctx->parallel_ctx().parallel_id()) {
    if (ctx->op_conf().has_stream_name_hint()) { stream_name_ = ctx->op_conf().stream_name_hint(); }
  }
  ~NcclLogical2DSameDim1KernelCommState() = default;

  ccl::CclComm ccl_comm() {
    if (!is_init_) {
      std::set<std::pair<int64_t, int64_t>> device_set;
      const Shape& hierarchy = *parallel_desc_.hierarchy();
      CHECK_EQ(hierarchy.NumAxes(), 2);
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
      EagerCclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerCclCommMgr>::Get());
      ccl_comm_ = comm_mgr->GetCclCommForDeviceAndStreamName(device_set, stream_name_);
      is_init_ = true;
    }
    return ccl_comm_;
  }

  const std::string& stream_name() const { return stream_name_; }

 private:
  bool is_init_;
  std::string stream_name_;
  ParallelDesc parallel_desc_;
  int64_t this_parallel_id_;
  ccl::CclComm ccl_comm_{};
};

class NcclLogical2DSameDim1AllReduce final : public user_op::OpKernel {
 public:
  NcclLogical2DSameDim1AllReduce() = default;
  ~NcclLogical2DSameDim1AllReduce() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical2DSameDim1KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* comm_state = dynamic_cast<NcclLogical2DSameDim1KernelCommState*>(state);
    CHECK(comm_state != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->shape_view(), out->shape_view());
    CHECK_EQ(in->data_type(), out->data_type());
    VLOG(3) << "[NcclLogical2D][SameDim1AllReduce] " << comm_state->stream_name() << " "
            << ctx->op_name() << std::endl;
    ccl::ReduceType ccl_reduce_type = ccl::ReduceType::kSum;
    if (in->data_type() == DataType::kBool) { ccl_reduce_type = ccl::ReduceType::kMax; }

    ccl::CclComm ccl_comm = comm_state->ccl_comm();
    std::unique_ptr<ccl::AllReduce> ccl_all_reduce =
        ccl::NewCollectiveCommunication<ccl::AllReduce>(ctx->stream()->device_type(),
                                                        in->data_type(), ccl_reduce_type);
    ccl_all_reduce->Launch(ctx->stream(), in->dptr(), out->mut_dptr(), in->shape_view().elem_cnt(),
                           ccl_comm);
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  bool IsKernelLaunchSynchronized() const override {
    const EagerCclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerCclCommMgr>::Get());
    return comm_mgr->IsAsyncLaunchCclLogicalKernel();
  }
};

}  // namespace

REGISTER_USER_KERNEL("_nccl_logical_2D_same_dim0_all_reduce")
    .SetCreateFn<NcclLogical2DSameDim0AllReduce>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCUDA);

REGISTER_USER_KERNEL("_nccl_logical_2D_same_dim0_all_gather")
    .SetCreateFn<NcclLogical2DSameDim0AllGather>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCUDA);

#define REGISTER_2D_SAME_DIM0_ALLGATHER_NONCONTINUOUS_KERNEL(dtype)                      \
  REGISTER_USER_KERNEL("_nccl_logical_2D_same_dim0_all_gather_noncontinuous")            \
      .SetCreateFn<NcclLogical2DSameDim0AllGatherNoncontinuous<dtype>>()                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                   \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)   \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(Infer2DSameDim0AllGatherNoncontinuousKernelTmpBufferSize);

REGISTER_2D_SAME_DIM0_ALLGATHER_NONCONTINUOUS_KERNEL(bool)
REGISTER_2D_SAME_DIM0_ALLGATHER_NONCONTINUOUS_KERNEL(int8_t)
REGISTER_2D_SAME_DIM0_ALLGATHER_NONCONTINUOUS_KERNEL(int32_t)
REGISTER_2D_SAME_DIM0_ALLGATHER_NONCONTINUOUS_KERNEL(int64_t)
REGISTER_2D_SAME_DIM0_ALLGATHER_NONCONTINUOUS_KERNEL(float)
REGISTER_2D_SAME_DIM0_ALLGATHER_NONCONTINUOUS_KERNEL(double)
REGISTER_2D_SAME_DIM0_ALLGATHER_NONCONTINUOUS_KERNEL(float16)
#if defined(__CUDA_BF16_TYPES_EXIST__)
REGISTER_2D_SAME_DIM0_ALLGATHER_NONCONTINUOUS_KERNEL(nv_bfloat16)
#endif

#define REGISTER_2D_SAME_DIM0_ALL2ALL_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("_nccl_logical_2D_same_dim0_all2all")                             \
      .SetCreateFn<NcclLogical2DSameDim0All2All<dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                   \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)   \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(Infer2DSameDim0All2AllKernelTmpBufferSize);

REGISTER_2D_SAME_DIM0_ALL2ALL_KERNEL(bool)
REGISTER_2D_SAME_DIM0_ALL2ALL_KERNEL(int8_t)
REGISTER_2D_SAME_DIM0_ALL2ALL_KERNEL(int32_t)
REGISTER_2D_SAME_DIM0_ALL2ALL_KERNEL(int64_t)
REGISTER_2D_SAME_DIM0_ALL2ALL_KERNEL(float)
REGISTER_2D_SAME_DIM0_ALL2ALL_KERNEL(double)
REGISTER_2D_SAME_DIM0_ALL2ALL_KERNEL(float16)
#if defined(__CUDA_BF16_TYPES_EXIST__)
REGISTER_2D_SAME_DIM0_ALL2ALL_KERNEL(nv_bfloat16)
#endif

REGISTER_USER_KERNEL("_nccl_logical_2D_same_dim1_all_reduce")
    .SetCreateFn<NcclLogical2DSameDim1AllReduce>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCUDA);

REGISTER_USER_KERNEL_UNIFIED_NCCL_COMM_INIT("_nccl_logical_2D_same_dim0_all_reduce");
REGISTER_USER_KERNEL_UNIFIED_NCCL_COMM_INIT("_nccl_logical_2D_same_dim0_all_gather");
REGISTER_USER_KERNEL_UNIFIED_NCCL_COMM_INIT("_nccl_logical_2D_same_dim0_all_gather_noncontinuous");
REGISTER_USER_KERNEL_UNIFIED_NCCL_COMM_INIT("_nccl_logical_2D_same_dim0_all2all");
REGISTER_USER_KERNEL_UNIFIED_NCCL_COMM_INIT("_nccl_logical_2D_same_dim1_all_reduce");

}  // namespace oneflow

#endif  // WITH_CUDA && NCCL_VERSION_CODE > 2700
