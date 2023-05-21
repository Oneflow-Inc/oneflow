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

size_t GetTmpBufferSizeByNcclType(const std::string& nccl_type, size_t in_tensor_byte_size,
                                  size_t out_tensor_byte_size, const NdSbp& src_nd_sbp,
                                  const NdSbp& dst_nd_sbp) {
  if (nccl_type == "_nccl_logical_all_gather_noncontinuous") {
    return out_tensor_byte_size;
  } else if (nccl_type == "_nccl_logical_reduce_scatter_noncontinuous") {
    return in_tensor_byte_size;
  } else if (nccl_type == "_nccl_logical_s2s") {
    size_t ret = 0;
    CHECK_EQ(src_nd_sbp.sbp_parallel_size(), 1);
    CHECK_EQ(dst_nd_sbp.sbp_parallel_size(), 1);
    CHECK(src_nd_sbp.sbp_parallel(0).has_split_parallel());
    CHECK(dst_nd_sbp.sbp_parallel(0).has_split_parallel());
    if (src_nd_sbp.sbp_parallel(0).split_parallel().axis() != 0) { ret += in_tensor_byte_size; }
    if (dst_nd_sbp.sbp_parallel(0).split_parallel().axis() != 0) { ret += in_tensor_byte_size; }
    return ret;
  } else if (nccl_type == "_nccl_logical_2D_same_dim0_all_gather_noncontinuous") {
    return out_tensor_byte_size;
  } else if (nccl_type == "_nccl_logical_2D_same_dim0_all2all") {
    size_t ret = 0;
    CHECK_EQ(src_nd_sbp.sbp_parallel_size(), 2);
    CHECK_EQ(dst_nd_sbp.sbp_parallel_size(), 2);
    CHECK(src_nd_sbp.sbp_parallel(1).has_split_parallel());
    CHECK(dst_nd_sbp.sbp_parallel(1).has_split_parallel());
    if (src_nd_sbp.sbp_parallel(1).split_parallel().axis() != 0) { ret += in_tensor_byte_size; }
    if (dst_nd_sbp.sbp_parallel(1).split_parallel().axis() != 0) { ret += in_tensor_byte_size; }
    return ret;
  }
  return 0;
}

size_t GetTensorByteSize(const user_op::TensorDesc& tensor_desc) {
  return GetCudaAlignedSize(tensor_desc.shape().elem_cnt()
                            * GetSizeOfDataType(tensor_desc.data_type()));
}

size_t GetTensorByteSize(const user_op::Tensor& tensor) {
  return GetCudaAlignedSize(tensor.shape_view().elem_cnt() * GetSizeOfDataType(tensor.data_type()));
}

class NcclLogicalFusionKernelState : public user_op::OpKernelState {
 public:
  explicit NcclLogicalFusionKernelState(user_op::KernelInitContext* ctx)
      : is_init_(false),
        stream_name_(EagerNcclCommMgr::kDefaultStreamName),
        parallel_desc_(ctx->parallel_desc()),
        this_parallel_id_(ctx->parallel_ctx().parallel_id()),
        num_ranks_(-1),
        comm_key_("InvalidKey"),
        nccl_num_(-1) {
    if (ctx->op_conf().has_stream_name_hint()) { stream_name_ = ctx->op_conf().stream_name_hint(); }
    InitSplitAxisAndTmpBufferOffset(ctx);
  }
  ~NcclLogicalFusionKernelState() override = default;

  ncclComm_t comm() {
    if (!is_init_) { InitComm(); }
    return comm_;
  }

  int64_t num_ranks() {
    if (!is_init_) { InitComm(); }
    return num_ranks_;
  }

  const std::string& stream_name() const { return stream_name_; }
  int64_t src_split_axis(int32_t i) const {
    CHECK_GE(i, 0);
    CHECK_LT(i, src_split_axis_list_.size());
    return src_split_axis_list_.at(i);
  }
  int64_t dst_split_axis(int32_t i) const {
    CHECK_GE(i, 0);
    CHECK_LT(i, dst_split_axis_list_.size());
    return dst_split_axis_list_.at(i);
  }

  int32_t nccl_num() const { return nccl_num_; }
  size_t tmp_buffer_offset(int32_t i) {
    CHECK_GE(i, 0);
    CHECK_LT(i, tmp_buffer_offset_.size());
    return tmp_buffer_offset_.at(i);
  }

  size_t tmp_buffer_size(int32_t i) {
    CHECK_GE(i, 0);
    CHECK_LT(i, tmp_buffer_size_.size());
    return tmp_buffer_size_.at(i);
  }

 private:
  void InitComm() {
    CHECK(!is_init_);
    std::set<std::pair<int64_t, int64_t>> device_set;
    const Shape& hierarchy = *parallel_desc_.hierarchy();

    if (hierarchy.NumAxes() == 1) {
      num_ranks_ = parallel_desc_.parallel_num();
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
        num_ranks_ = group_size;
      } else {
        UNIMPLEMENTED();
      }
    } else {
      UNIMPLEMENTED();
    }

    EagerCclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerCclCommMgr>::Get());
    comm_ =
        comm_mgr->As<EagerNcclCommMgr>()->GetCommForDeviceAndStreamName(device_set, stream_name_);
    is_init_ = true;
  }

  void UpdateOrCheckEqCommKey(const std::string& val) {
    if (comm_key_ == "InvalidKey") {
      comm_key_ = val;
    } else {
      CHECK_EQ(comm_key_, val);
    }
  }

  void UpdateSplitAxisByNcclType(const std::string& nccl_type, const int32_t i,
                                 const NdSbp& src_nd_sbp, const NdSbp& dst_nd_sbp) {
    if (nccl_type == "_nccl_logical_all_gather_noncontinuous") {
      CHECK_EQ(src_nd_sbp.sbp_parallel_size(), 1);
      CHECK(src_nd_sbp.sbp_parallel(0).has_split_parallel());
      src_split_axis_list_.at(i) = src_nd_sbp.sbp_parallel(0).split_parallel().axis();
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

  void InitSplitAxisAndTmpBufferOffset(user_op::KernelInitContext* ctx) {
    nccl_num_ = ctx->input_size("in");
    const std::vector<std::string>& src_nd_sbp_str_list =
        ctx->Attr<std::vector<std::string>>("src_nd_sbp_str_list");
    const std::vector<std::string>& dst_nd_sbp_str_list =
        ctx->Attr<std::vector<std::string>>("dst_nd_sbp_str_list");
    const std::vector<std::string>& nccl_type_list =
        ctx->Attr<std::vector<std::string>>("nccl_type_list");

    CHECK_EQ(nccl_num_, ctx->output_size("out"));
    src_split_axis_list_.resize(nccl_num_, -1);
    dst_split_axis_list_.resize(nccl_num_, -1);

    CHECK_EQ(src_nd_sbp_str_list.size(), nccl_num_);
    CHECK_EQ(dst_nd_sbp_str_list.size(), nccl_num_);
    CHECK_EQ(nccl_type_list.size(), nccl_num_);
    CHECK_EQ(src_split_axis_list_.size(), nccl_num_);
    CHECK_EQ(dst_split_axis_list_.size(), nccl_num_);

    size_t total_buffer_size = 0;

    for (int32_t i = 0; i < nccl_num_; ++i) {
      NdSbp src_nd_sbp;
      NdSbp dst_nd_sbp;
      CHECK(ParseNdSbpFromLongString(src_nd_sbp_str_list.at(i), &src_nd_sbp));
      CHECK(ParseNdSbpFromLongString(dst_nd_sbp_str_list.at(i), &dst_nd_sbp));
      const std::string& nccl_type = nccl_type_list.at(i);
      UpdateOrCheckEqCommKey(GetCommKeyFromNcclType(nccl_type));
      UpdateSplitAxisByNcclType(nccl_type, i, src_nd_sbp, dst_nd_sbp);
      size_t in_tensor_byte_size = GetTensorByteSize(*ctx->TensorDesc4ArgNameAndIndex("in", i));
      size_t out_tensor_byte_size = GetTensorByteSize(*ctx->TensorDesc4ArgNameAndIndex("out", i));

      tmp_buffer_offset_.push_back(total_buffer_size);
      size_t tmp_buffer_size = GetTmpBufferSizeByNcclType(
          nccl_type, in_tensor_byte_size, out_tensor_byte_size, src_nd_sbp, dst_nd_sbp);
      tmp_buffer_size_.push_back(tmp_buffer_size);
      total_buffer_size += tmp_buffer_size;
    }
    // NOTE(chengcheng): last element of vector is total_buffer_size
    tmp_buffer_offset_.push_back(total_buffer_size);
    CHECK_EQ(tmp_buffer_offset_.size(), nccl_num_ + 1);
    CHECK_EQ(tmp_buffer_size_.size(), nccl_num_);
    const user_op::TensorDesc* tmp_buffer_tensor_desc =
        ctx->TensorDesc4ArgNameAndIndex("tmp_buffer", 0);
    if (tmp_buffer_tensor_desc == nullptr) {
      CHECK_EQ(total_buffer_size, 0);
    } else {
      CHECK_EQ(total_buffer_size, GetTensorByteSize(*tmp_buffer_tensor_desc));
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
  std::vector<size_t> tmp_buffer_offset_;
  std::vector<size_t> tmp_buffer_size_;
  ncclComm_t comm_{};
};

class NcclLogicalFusionKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclLogicalFusionKernel);
  NcclLogicalFusionKernel() = default;
  ~NcclLogicalFusionKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogicalFusionKernelState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  bool IsKernelLaunchSynchronized() const override {
    EagerCclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerCclCommMgr>::Get());
    return comm_mgr->IsAsyncLaunchCclLogicalKernel();
  }
};

const void* UpdatePackToPtrByNcclType(const void* pack_to_ptr, const std::string& nccl_type,
                                      user_op::Tensor* tmp_buffer,
                                      NcclLogicalFusionKernelState* kernel_state, const int32_t i) {
  CHECK_NOTNULL(tmp_buffer);
  const void* tmp_dptr =
      static_cast<const void*>(tmp_buffer->dptr<char>() + kernel_state->tmp_buffer_offset(i));
  if (nccl_type == "_nccl_logical_reduce_scatter_noncontinuous") {
    return tmp_dptr;
  } else if (nccl_type == "_nccl_logical_s2s") {
    if (kernel_state->dst_split_axis(i) != 0) {
      return tmp_dptr;  // need do pack;
    }
  } else if (nccl_type == "_nccl_logical_2D_same_dim0_all2all") {
    if (kernel_state->dst_split_axis(i) != 0) {
      return tmp_dptr;  // need do pack;
    }
  }
  return pack_to_ptr;
}

void* UpdateUnpackFromPtrByNcclType(void* unpack_from_ptr, const std::string& nccl_type,
                                    user_op::Tensor* tmp_buffer, const user_op::Tensor* in,
                                    NcclLogicalFusionKernelState* kernel_state, const int32_t i) {
  CHECK_NOTNULL(tmp_buffer);
  void* tmp_dptr =
      static_cast<void*>(tmp_buffer->mut_dptr<char>() + kernel_state->tmp_buffer_offset(i));
  int64_t data_size = GetTensorByteSize(*in);
  int64_t tmp_buffer_size = kernel_state->tmp_buffer_size(i);
  if (nccl_type == "_nccl_logical_all_gather_noncontinuous") {
    return tmp_dptr;
  } else if (nccl_type == "_nccl_logical_s2s") {
    if (kernel_state->src_split_axis(i) != 0) {
      CHECK(tmp_buffer_size == data_size || tmp_buffer_size == 2 * data_size);
      return static_cast<void*>(static_cast<char*>(tmp_dptr) + (tmp_buffer_size - data_size));
    }
  } else if (nccl_type == "_nccl_logical_2D_same_dim0_all_gather_noncontinuous") {
    return tmp_dptr;
  } else if (nccl_type == "_nccl_logical_2D_same_dim0_all2all") {
    if (kernel_state->src_split_axis(i) != 0) {
      CHECK(tmp_buffer_size == data_size || tmp_buffer_size == 2 * data_size);
      return static_cast<void*>(static_cast<char*>(tmp_dptr) + (tmp_buffer_size - data_size));
    }
  }
  return unpack_from_ptr;
}

void DoPackBeforeNcclGroup(void* pack_to_ptr, const std::string& nccl_type,
                           const user_op::Tensor* in, user_op::KernelComputeContext* ctx,
                           NcclLogicalFusionKernelState* kernel_state, const int32_t i) {
  if (nccl_type == "_nccl_logical_reduce_scatter_noncontinuous") {
    // Do pack before reduce scatter
    const int64_t num_ranks = kernel_state->num_ranks();
    const int64_t out_split_axis = kernel_state->dst_split_axis(i);
    DimVector transpose_in_dim_vec;
    in->shape_view().ToDimVector(&transpose_in_dim_vec);

    transpose_in_dim_vec[out_split_axis] = transpose_in_dim_vec.at(out_split_axis) / num_ranks;
    transpose_in_dim_vec.insert(transpose_in_dim_vec.begin() + out_split_axis, num_ranks);
    const Shape transpose_in_shape(transpose_in_dim_vec);
    std::vector<int32_t> perm;
    perm.emplace_back(out_split_axis);
    FOR_RANGE(int64_t, i, 0, transpose_in_dim_vec.size()) {
      if (i != out_split_axis) { perm.emplace_back(i); }
    }
    auto transpose = ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(
        ctx->stream()->device_type(), transpose_in_dim_vec.size());
    CHECK(transpose);
    transpose->Launch(ctx->stream(), in->data_type(), transpose_in_dim_vec.size(),
                      transpose_in_dim_vec.data(), in->dptr(), perm.data(), pack_to_ptr);
    VLOG(3) << "[NcclLogicalFusion] op: " << ctx->op_name() << " , i= " << i
            << ", stream: " << kernel_state->stream_name() << " Do pack before [ReduceScatter]";
  } else if (nccl_type == "_nccl_logical_s2s") {
    const int64_t out_split_axis = kernel_state->dst_split_axis(i);
    if (out_split_axis != 0) {
      // Do pack before all2all
      const int64_t num_ranks = kernel_state->num_ranks();
      DimVector transpose_in_dim_vec;
      in->shape_view().ToDimVector(&transpose_in_dim_vec);
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
                        transpose_in_dim_vec.data(), in->dptr(), perm.data(), pack_to_ptr);
      VLOG(3) << "[NcclLogicalFusion] op: " << ctx->op_name() << " , i= " << i
              << ", stream: " << kernel_state->stream_name() << " Do pack before [All2All]";
    }
  } else if (nccl_type == "_nccl_logical_2D_same_dim0_all2all") {
    const int64_t out_split_axis = kernel_state->dst_split_axis(i);
    if (out_split_axis != 0) {
      const int64_t num_ranks = kernel_state->num_ranks();
      DimVector transpose_in_dim_vec;
      in->shape_view().ToDimVector(&transpose_in_dim_vec);
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
                        transpose_in_dim_vec.data(), in->dptr(), perm.data(), pack_to_ptr);
      VLOG(3) << "[NcclLogicalFusion] op: " << ctx->op_name() << " , i= " << i
              << ", stream: " << kernel_state->stream_name()
              << " Do pack before [2DSameDim0All2All]";
    }
  }
}

void DoNcclComputeByNcclTypeInGroup(const void* pack_to_ptr, void* unpack_from_ptr,
                                    const std::string& nccl_type, const user_op::Tensor* in,
                                    user_op::Tensor* out, user_op::KernelComputeContext* ctx,
                                    NcclLogicalFusionKernelState* kernel_state, const int32_t i,
                                    const ncclComm_t& comm) {
  const int64_t num_ranks = kernel_state->num_ranks();
  VLOG(3) << "[NcclLogicalFusion] op: " << ctx->op_name() << " , i= " << i
          << ", stream: " << kernel_state->stream_name() << " Try launch nccl_type: " << nccl_type;
  if (nccl_type == "_nccl_logical_all_reduce") {
    CHECK(in->dptr() == pack_to_ptr);
    CHECK(out->mut_dptr() == unpack_from_ptr);
    ncclRedOp_t reduce_type = ncclRedOp_t::ncclSum;
    if (in->data_type() == DataType::kBool) { reduce_type = ncclRedOp_t::ncclMax; }
    OF_NCCL_CHECK(ncclAllReduce(in->dptr(), out->mut_dptr(), in->shape_view().elem_cnt(),
                                GetNcclDataType(in->data_type()), reduce_type, comm,
                                ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  } else if (nccl_type == "_nccl_logical_reduce_scatter") {
    CHECK(in->dptr() == pack_to_ptr);
    CHECK(out->mut_dptr() == unpack_from_ptr);
    CHECK_EQ(in->shape_view().elem_cnt(), out->shape_view().elem_cnt() * num_ranks);
    ncclRedOp_t reduce_type = ncclRedOp_t::ncclSum;
    if (in->data_type() == DataType::kBool) { reduce_type = ncclRedOp_t::ncclMax; }
    OF_NCCL_CHECK(ncclReduceScatter(in->dptr(), out->mut_dptr(), out->shape_view().elem_cnt(),
                                    GetNcclDataType(in->data_type()), reduce_type, comm,
                                    ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  } else if (nccl_type == "_nccl_logical_all_gather") {
    CHECK(in->dptr() == pack_to_ptr);
    CHECK(out->mut_dptr() == unpack_from_ptr);
    CHECK_EQ(in->shape_view().elem_cnt() * num_ranks, out->shape_view().elem_cnt());
    OF_NCCL_CHECK(ncclAllGather(in->dptr(), out->mut_dptr(), in->shape_view().elem_cnt(),
                                GetNcclDataType(in->data_type()), comm,
                                ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  } else if (nccl_type == "_nccl_logical_all_gather_noncontinuous") {
    CHECK(in->dptr() == pack_to_ptr);
    CHECK(out->mut_dptr() != unpack_from_ptr);  // do unpack from ptr -> out
    CHECK_EQ(in->shape_view().elem_cnt() * num_ranks, out->shape_view().elem_cnt());
    OF_NCCL_CHECK(ncclAllGather(in->dptr(), unpack_from_ptr, in->shape_view().elem_cnt(),
                                GetNcclDataType(in->data_type()), comm,
                                ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  } else if (nccl_type == "_nccl_logical_reduce_scatter_noncontinuous") {
    CHECK(in->dptr() != pack_to_ptr);  // do in -> pack to ptr
    CHECK(out->mut_dptr() == unpack_from_ptr);
    ncclRedOp_t reduce_type = ncclRedOp_t::ncclSum;
    if (in->data_type() == DataType::kBool) { reduce_type = ncclRedOp_t::ncclMax; }
    OF_NCCL_CHECK(ncclReduceScatter(pack_to_ptr, out->mut_dptr(), out->shape_view().elem_cnt(),
                                    GetNcclDataType(in->data_type()), reduce_type, comm,
                                    ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  } else if (nccl_type == "_nccl_logical_s2s") {
    const int64_t elem_cnt = in->shape_view().elem_cnt();
    const int64_t dtype_size = GetSizeOfDataType(in->data_type());
    const int64_t elem_per_chunk = elem_cnt / num_ranks;
    const int64_t chunk_size = elem_per_chunk * dtype_size;
    for (int64_t j = 0; j < num_ranks; ++j) {
      OF_NCCL_CHECK(ncclSend(reinterpret_cast<const void*>(
                                 reinterpret_cast<const char*>(pack_to_ptr) + j * chunk_size),
                             elem_per_chunk, GetNcclDataType(in->data_type()), j, comm,
                             ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
      OF_NCCL_CHECK(ncclRecv(
          reinterpret_cast<void*>(reinterpret_cast<char*>(unpack_from_ptr) + j * chunk_size),
          elem_per_chunk, GetNcclDataType(in->data_type()), j, comm,
          ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
    }
  } else if (nccl_type == "_nccl_logical_2D_same_dim0_all_reduce") {
    CHECK(in->dptr() == pack_to_ptr);
    CHECK(out->mut_dptr() == unpack_from_ptr);
    ncclRedOp_t reduce_type = ncclRedOp_t::ncclSum;
    if (in->data_type() == DataType::kBool) { reduce_type = ncclRedOp_t::ncclMax; }
    OF_NCCL_CHECK(ncclAllReduce(in->dptr(), out->mut_dptr(), in->shape_view().elem_cnt(),
                                GetNcclDataType(in->data_type()), reduce_type, comm,
                                ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  } else if (nccl_type == "_nccl_logical_2D_same_dim0_all_gather") {
    CHECK(in->dptr() == pack_to_ptr);
    CHECK(out->mut_dptr() == unpack_from_ptr);
    CHECK_EQ(in->shape_view().elem_cnt() * num_ranks, out->shape_view().elem_cnt());
    OF_NCCL_CHECK(ncclAllGather(in->dptr(), out->mut_dptr(), in->shape_view().elem_cnt(),
                                GetNcclDataType(in->data_type()), comm,
                                ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  } else if (nccl_type == "_nccl_logical_2D_same_dim0_all_gather_noncontinuous") {
    CHECK(in->dptr() == pack_to_ptr);
    CHECK(out->mut_dptr() != unpack_from_ptr);  // do unpack from ptr -> out
    CHECK_EQ(in->shape_view().elem_cnt() * num_ranks, out->shape_view().elem_cnt());
    OF_NCCL_CHECK(ncclAllGather(in->dptr(), unpack_from_ptr, in->shape_view().elem_cnt(),
                                GetNcclDataType(in->data_type()), comm,
                                ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  } else if (nccl_type == "_nccl_logical_2D_same_dim0_all2all") {
    const int64_t elem_cnt = in->shape_view().elem_cnt();
    const int64_t dtype_size = GetSizeOfDataType(in->data_type());
    const int64_t elem_per_chunk = elem_cnt / num_ranks;
    const int64_t chunk_size = elem_per_chunk * dtype_size;
    for (int64_t j = 0; j < num_ranks; ++j) {
      OF_NCCL_CHECK(ncclSend(reinterpret_cast<const void*>(
                                 reinterpret_cast<const char*>(pack_to_ptr) + j * chunk_size),
                             elem_per_chunk, GetNcclDataType(in->data_type()), j, comm,
                             ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
      OF_NCCL_CHECK(ncclRecv(
          reinterpret_cast<void*>(reinterpret_cast<char*>(unpack_from_ptr) + j * chunk_size),
          elem_per_chunk, GetNcclDataType(in->data_type()), j, comm,
          ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
    }
  } else if (nccl_type == "_nccl_logical_2D_same_dim1_all_reduce") {
    CHECK(in->dptr() == pack_to_ptr);
    CHECK(out->mut_dptr() == unpack_from_ptr);
    ncclRedOp_t reduce_type = ncclRedOp_t::ncclSum;
    if (in->data_type() == DataType::kBool) { reduce_type = ncclRedOp_t::ncclMax; }
    OF_NCCL_CHECK(ncclAllReduce(in->dptr(), out->mut_dptr(), in->shape_view().elem_cnt(),
                                GetNcclDataType(in->data_type()), reduce_type, comm,
                                ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
  } else {
    UNIMPLEMENTED();
  }
  VLOG(3) << "[NcclLogicalFusion] op: " << ctx->op_name() << " , i= " << i
          << ", stream: " << kernel_state->stream_name() << " launched nccl_type: " << nccl_type;
}

void DoUnpackAfterNcclGroup(void* unpack_from_ptr, const std::string& nccl_type,
                            const user_op::Tensor* in, user_op::Tensor* out,
                            user_op::KernelComputeContext* ctx,
                            NcclLogicalFusionKernelState* kernel_state, const int32_t i) {
  const int64_t num_ranks = kernel_state->num_ranks();
  const int64_t in_split_axis = kernel_state->src_split_axis(i);
  const int64_t out_split_axis = kernel_state->dst_split_axis(i);
  if (nccl_type == "_nccl_logical_all_gather_noncontinuous") {
    CHECK_GT(in_split_axis, 0);
    DimVector unpack_from_dim_vec;
    in->shape_view().ToDimVector(&unpack_from_dim_vec);
    unpack_from_dim_vec.insert(unpack_from_dim_vec.begin(), num_ranks);
    std::vector<int32_t> perm;
    FOR_RANGE(int64_t, i, 1, unpack_from_dim_vec.size()) { perm.emplace_back(i); }
    perm.insert(perm.begin() + in_split_axis, 0);
    auto transpose = ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(
        ctx->stream()->device_type(), unpack_from_dim_vec.size());
    CHECK(transpose);
    transpose->Launch(ctx->stream(), in->data_type(), unpack_from_dim_vec.size(),
                      unpack_from_dim_vec.data(), unpack_from_ptr, perm.data(), out->mut_dptr());
    VLOG(3) << "[NcclLogicalFusion] op: " << ctx->op_name() << " , i= " << i
            << ", stream: " << kernel_state->stream_name()
            << " Do unpack after [AllGatherNoncontinuous]";
  } else if (nccl_type == "_nccl_logical_s2s") {
    CHECK_GE(in_split_axis, 0);
    CHECK_GE(out_split_axis, 0);
    if (in_split_axis != 0) {
      // Do unpack.
      CHECK(unpack_from_ptr != out->mut_dptr());

      DimVector unpack_from_dim_vec;
      in->shape_view().ToDimVector(&unpack_from_dim_vec);
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
      VLOG(3) << "[NcclLogicalFusion] op: " << ctx->op_name() << " , i= " << i
              << ", stream: " << kernel_state->stream_name() << " Do unpack after [All2All]";
    }
  } else if (nccl_type == "_nccl_logical_2D_same_dim0_all_gather_noncontinuous") {
    DimVector unpack_from_dim_vec;
    in->shape_view().ToDimVector(&unpack_from_dim_vec);
    CHECK_GT(in_split_axis, 0);
    // NOTE(chengcheng): Do unpack.
    unpack_from_dim_vec.insert(unpack_from_dim_vec.begin(), num_ranks);
    std::vector<int32_t> perm;
    FOR_RANGE(int64_t, i, 1, unpack_from_dim_vec.size()) { perm.emplace_back(i); }
    perm.insert(perm.begin() + in_split_axis, 0);

    auto transpose = ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(
        ctx->stream()->device_type(), unpack_from_dim_vec.size());
    CHECK(transpose);
    transpose->Launch(ctx->stream(), in->data_type(), unpack_from_dim_vec.size(),
                      unpack_from_dim_vec.data(), unpack_from_ptr, perm.data(), out->mut_dptr());
    VLOG(3) << "[NcclLogicalFusion] op: " << ctx->op_name() << " , i= " << i
            << ", stream: " << kernel_state->stream_name()
            << " Do unpack after [SameDim0AllGatherNoncontinuous]";
  } else if (nccl_type == "_nccl_logical_2D_same_dim0_all2all") {
    CHECK_GE(in_split_axis, 0);
    CHECK_GE(out_split_axis, 0);
    if (in_split_axis != 0) {
      DimVector unpack_from_dim_vec;
      in->shape_view().ToDimVector(&unpack_from_dim_vec);
      // Do unpack.
      CHECK(unpack_from_ptr != out->mut_dptr());
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
  }
}

void NcclLogicalFusionKernel::Compute(user_op::KernelComputeContext* ctx,
                                      user_op::OpKernelState* state,
                                      const user_op::OpKernelCache*) const {
  auto* kernel_state = dynamic_cast<NcclLogicalFusionKernelState*>(state);
  CHECK_NOTNULL(kernel_state);
  const int32_t nccl_num = kernel_state->nccl_num();
  const std::vector<std::string>& nccl_type_list =
      ctx->Attr<std::vector<std::string>>("nccl_type_list");

  user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

  // NOTE(chengcheng):
  //    pack:   in.dptr -> pack_to_ptr              // if not pack : pack_to_ptr = in.dptr
  //    nccl:   pack_to_ptr -> unpack_from_ptr
  //    unpack: unpack_from_ptr ->out.dptr          // if not unpack: unpack_from_ptr = out.dptr
  std::vector<const void*> pack_to_ptr_list(nccl_num, nullptr);
  std::vector<void*> unpack_from_ptr_list(nccl_num, nullptr);
  std::vector<DataType> dtype_list(nccl_num, DataType::kInvalidDataType);
  for (int32_t i = 0; i < nccl_num; ++i) {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", i);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", i);
    pack_to_ptr_list.at(i) = in->dptr();
    unpack_from_ptr_list.at(i) = out->mut_dptr();
    dtype_list.at(i) = in->data_type();
    CHECK_EQ(dtype_list.at(i), out->data_type());
  }

  // try to do pack before all nccl
  for (int32_t i = 0; i < nccl_num; ++i) {
    if (kernel_state->tmp_buffer_size(i) == 0) { continue; }
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", i);
    // TODO(chengcheng): refactor code by template.
    pack_to_ptr_list.at(i) = UpdatePackToPtrByNcclType(pack_to_ptr_list.at(i), nccl_type_list.at(i),
                                                       tmp_buffer, kernel_state, i);
    unpack_from_ptr_list.at(i) = UpdateUnpackFromPtrByNcclType(
        unpack_from_ptr_list.at(i), nccl_type_list.at(i), tmp_buffer, in, kernel_state, i);
    DoPackBeforeNcclGroup(const_cast<void*>(pack_to_ptr_list.at(i)) /* mut dptr */,
                          nccl_type_list.at(i), in, ctx, kernel_state, i);
  }

  // NOTE(chengcheng): init nccl comm need before ncclGroupStart.
  ncclComm_t comm = kernel_state->comm();

  // do nccl compute in group
  OF_NCCL_CHECK(ncclGroupStart());
  for (int32_t i = 0; i < nccl_num; ++i) {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", i);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", i);
    DoNcclComputeByNcclTypeInGroup(pack_to_ptr_list.at(i), unpack_from_ptr_list.at(i),
                                   nccl_type_list.at(i), in, out, ctx, kernel_state, i, comm);
  }
  OF_NCCL_CHECK(ncclGroupEnd());

  // try to do unpack after all nccl
  for (int32_t i = 0; i < nccl_num; ++i) {
    if (kernel_state->tmp_buffer_size(i) == 0) { continue; }
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", i);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", i);
    DoUnpackAfterNcclGroup(unpack_from_ptr_list.at(i), nccl_type_list.at(i), in, out, ctx,
                           kernel_state, i);
  }
}

size_t InferNcclLogicalFusionKernelTmpBufferSize(user_op::InferContext* ctx) {
  size_t total_buffer_size = 0;
  const auto& src_nd_sbp_str_list = ctx->Attr<std::vector<std::string>>("src_nd_sbp_str_list");
  const auto& dst_nd_sbp_str_list = ctx->Attr<std::vector<std::string>>("dst_nd_sbp_str_list");
  const auto& nccl_type_list = ctx->Attr<std::vector<std::string>>("nccl_type_list");
  int32_t nccl_num = nccl_type_list.size();
  CHECK_EQ(nccl_num, ctx->input_size("in"));
  CHECK_EQ(nccl_num, ctx->output_size("out"));
  CHECK_EQ(nccl_num, src_nd_sbp_str_list.size());
  CHECK_EQ(nccl_num, dst_nd_sbp_str_list.size());
  for (int32_t i = 0; i < nccl_num; ++i) {
    const std::string& nccl_type = nccl_type_list.at(i);
    size_t in_tensor_byte_size = GetTensorByteSize(ctx->InputTensorDesc("in", i));
    size_t out_tensor_byte_size = GetTensorByteSize(ctx->OutputTensorDesc("out", i));
    NdSbp src_nd_sbp;
    NdSbp dst_nd_sbp;
    CHECK(ParseNdSbpFromLongString(src_nd_sbp_str_list.at(i), &src_nd_sbp));
    CHECK(ParseNdSbpFromLongString(dst_nd_sbp_str_list.at(i), &dst_nd_sbp));
    total_buffer_size += GetTmpBufferSizeByNcclType(nccl_type, in_tensor_byte_size,
                                                    out_tensor_byte_size, src_nd_sbp, dst_nd_sbp);
  }
  return total_buffer_size;
}

REGISTER_USER_KERNEL("_nccl_logical_fusion")
    .SetCreateFn<NcclLogicalFusionKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCUDA)
    .SetInferTmpSizeFn(InferNcclLogicalFusionKernelTmpBufferSize);
}  // namespace

}  // namespace oneflow

#endif  // WITH_CUDA && NCCL_VERSION_CODE > 2700
