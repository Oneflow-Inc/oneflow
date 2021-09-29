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
#include "oneflow/core/kernel/new_kernel_util.h"

#if defined(WITH_CUDA) && NCCL_VERSION_CODE > 2700

namespace oneflow {

namespace {

class NcclLogical3DChangeDim2KernelCommState final : public user_op::OpKernelState {
 public:
  NcclLogical3DChangeDim2KernelCommState(user_op::KernelInitContext* ctx)
      : is_init_(false),
        has_independent_stream_(ctx->op_conf().has_stream_index_hint()),
        stream_index_(ctx->op_conf().stream_index_hint()),
        parallel_desc_(ctx->parallel_desc()),
        this_parallel_id_(ctx->parallel_ctx().parallel_id()) {
    CHECK_JUST(GetIdMap(parallel_desc_, &parallel_id2map_id_, &map_id2parallel_id_));
  }
  ~NcclLogical3DChangeDim2KernelCommState() = default;

  ncclComm_t comm() {
    if (!is_init_) { Init(); }
    return comm_;
  }

  int64_t num_ranks() {
    if (!is_init_) { Init(); }
    return num_ranks_;
  }

 private:
  void Init() {
    CHECK(!is_init_);
    std::set<std::pair<int64_t, int64_t>> device_set;
    const Shape& hierarchy = *parallel_desc_.hierarchy();
    CHECK_EQ(hierarchy.NumAxes(), 3);
    const int64_t num_groups = hierarchy.At(0) * hierarchy.At(1);
    const int64_t group_size = hierarchy.At(2);
    CHECK_EQ(num_groups * group_size, parallel_desc_.parallel_num());
    const int64_t this_group_begin_data_id =
        parallel_id2map_id_.at(this_parallel_id_) / group_size * group_size;
    CHECK_EQ(this_group_begin_data_id % group_size, 0);
    CHECK_LE(this_group_begin_data_id + group_size, parallel_desc_.parallel_num());
    for (int64_t id_in_group = 0; id_in_group < group_size; ++id_in_group) {
      const int64_t parallel_id = map_id2parallel_id_.at(this_group_begin_data_id + id_in_group);
      const int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
      const int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
      device_set.emplace(std::make_pair(machine_id, device_id));
    }
    EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Global<EagerNcclCommMgr>::Get());
    if (has_independent_stream_) {
      comm_ = comm_mgr->GetCommForDeviceAndStreamId(device_set, stream_index_);
    } else {
      comm_ = comm_mgr->GetCommForDevice(device_set);
    }
    num_ranks_ = group_size;
    is_init_ = true;
  }

  bool is_init_;
  bool has_independent_stream_;
  int32_t stream_index_;
  ParallelDesc parallel_desc_;
  int64_t this_parallel_id_;
  int64_t num_ranks_;
  ncclComm_t comm_;
  HashMap<int64_t, int64_t> map_id2parallel_id_;
  HashMap<int64_t, int64_t> parallel_id2map_id_;
};

class NcclLogical3DChangeDim1KernelCommState final : public user_op::OpKernelState {
 public:
  NcclLogical3DChangeDim1KernelCommState(user_op::KernelInitContext* ctx)
      : is_init_(false),
        parallel_desc_(ctx->parallel_desc()),
        this_parallel_id_(ctx->parallel_ctx().parallel_id()) {}
  ~NcclLogical3DChangeDim1KernelCommState() = default;

  ncclComm_t comm() {
    if (!is_init_) { Init(); }
    return comm_;
  }

  int64_t num_ranks() {
    if (!is_init_) { Init(); }
    return num_ranks_;
  }

 private:
  void Init() {
    CHECK(!is_init_);
    std::set<std::pair<int64_t, int64_t>> device_set;
    const Shape& hierarchy = *parallel_desc_.hierarchy();
    CHECK_EQ(hierarchy.NumAxes(), 3);
    const int64_t group_size = hierarchy.At(1);
    const int64_t num_groups = hierarchy.At(2);
    const int64_t this_group_begin_parallel_id =
        this_parallel_id_ / (group_size * num_groups) * (group_size * num_groups)
        + this_parallel_id_ % num_groups;
    CHECK_LT(this_group_begin_parallel_id + (group_size - 1) * num_groups,
             parallel_desc_.parallel_num());
    for (int64_t id_in_group = 0; id_in_group < group_size; ++id_in_group) {
      const int64_t parallel_id = this_group_begin_parallel_id + (id_in_group * num_groups);
      const int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
      const int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
      device_set.emplace(std::make_pair(machine_id, device_id));
    }
    comm_ = CHECK_NOTNULL(Global<EagerNcclCommMgr>::Get())->GetCommForDevice(device_set);
    num_ranks_ = group_size;
    is_init_ = true;
  }

  bool is_init_;
  ParallelDesc parallel_desc_;
  int64_t this_parallel_id_;
  int64_t num_ranks_;
  ncclComm_t comm_;
};

class NcclLogical3DChangeDim0KernelCommState final : public user_op::OpKernelState {
 public:
  NcclLogical3DChangeDim0KernelCommState(user_op::KernelInitContext* ctx)
      : is_init_(false),
        parallel_desc_(ctx->parallel_desc()),
        this_parallel_id_(ctx->parallel_ctx().parallel_id()) {}
  ~NcclLogical3DChangeDim0KernelCommState() = default;

  ncclComm_t comm() {
    if (!is_init_) {
      std::set<std::pair<int64_t, int64_t>> device_set;
      const Shape& hierarchy = *parallel_desc_.hierarchy();
      CHECK_EQ(hierarchy.NumAxes(), 3);
      const int64_t group_size = hierarchy.At(0);
      const int64_t num_groups = hierarchy.At(1) * hierarchy.At(2);
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
      comm_ = CHECK_NOTNULL(Global<EagerNcclCommMgr>::Get())->GetCommForDevice(device_set);
      is_init_ = true;
    }
    return comm_;
  }

 private:
  bool is_init_;
  ParallelDesc parallel_desc_;
  int64_t this_parallel_id_;
  ncclComm_t comm_;
};

class NcclLogical3DChangeDim0AllReduce final : public user_op::OpKernel {
 public:
  NcclLogical3DChangeDim0AllReduce() = default;
  ~NcclLogical3DChangeDim0AllReduce() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical3DChangeDim0KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* nccl_comm = dynamic_cast<NcclLogical3DChangeDim0KernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->shape(), out->shape());
    CHECK_EQ(in->data_type(), out->data_type());
    OF_NCCL_CHECK(ncclAllReduce(in->dptr(), out->mut_dptr(), in->shape().elem_cnt(),
                                GetNcclDataType(in->data_type()), ncclRedOp_t::ncclSum,
                                nccl_comm->comm(), ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class NcclLogical3DChangeDim1AllReduce final : public user_op::OpKernel {
 public:
  NcclLogical3DChangeDim1AllReduce() = default;
  ~NcclLogical3DChangeDim1AllReduce() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical3DChangeDim1KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* nccl_comm = dynamic_cast<NcclLogical3DChangeDim1KernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->shape(), out->shape());
    CHECK_EQ(in->data_type(), out->data_type());
    OF_NCCL_CHECK(ncclAllReduce(in->dptr(), out->mut_dptr(), in->shape().elem_cnt(),
                                GetNcclDataType(in->data_type()), ncclRedOp_t::ncclSum,
                                nccl_comm->comm(), ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class NcclLogical3DChangeDim2AllReduce final : public user_op::OpKernel {
 public:
  NcclLogical3DChangeDim2AllReduce() = default;
  ~NcclLogical3DChangeDim2AllReduce() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical3DChangeDim2KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* nccl_comm = dynamic_cast<NcclLogical3DChangeDim2KernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->shape(), out->shape());
    CHECK_EQ(in->data_type(), out->data_type());
    OF_NCCL_CHECK(ncclAllReduce(in->dptr(), out->mut_dptr(), in->shape().elem_cnt(),
                                GetNcclDataType(in->data_type()), ncclRedOp_t::ncclSum,
                                nccl_comm->comm(), ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

REGISTER_USER_KERNEL("_nccl_logical_3D_change_dim0_all_reduce")
    .SetCreateFn<NcclLogical3DChangeDim0AllReduce>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");

REGISTER_USER_KERNEL("_nccl_logical_3D_change_dim1_all_reduce")
    .SetCreateFn<NcclLogical3DChangeDim1AllReduce>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");

REGISTER_USER_KERNEL("_nccl_logical_3D_change_dim2_all_reduce")
    .SetCreateFn<NcclLogical3DChangeDim2AllReduce>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");

class NcclLogical3DChangeDim2AllGather final : public user_op::OpKernel {
 public:
  NcclLogical3DChangeDim2AllGather() = default;
  ~NcclLogical3DChangeDim2AllGather() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical3DChangeDim2KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* nccl_comm = dynamic_cast<NcclLogical3DChangeDim2KernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->data_type(), out->data_type());
    const int64_t num_ranks = nccl_comm->num_ranks();
    CHECK_EQ(in->shape().elem_cnt() * num_ranks, out->shape().elem_cnt());
    OF_NCCL_CHECK(ncclAllGather(in->dptr(), out->mut_dptr(), in->shape().elem_cnt(),
                                GetNcclDataType(in->data_type()), nccl_comm->comm(),
                                ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class NcclLogical3DChangeDim1AllGather final : public user_op::OpKernel {
 public:
  NcclLogical3DChangeDim1AllGather() = default;
  ~NcclLogical3DChangeDim1AllGather() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical3DChangeDim1KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* nccl_comm = dynamic_cast<NcclLogical3DChangeDim1KernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->data_type(), out->data_type());
    // const int64_t num_ranks = nccl_comm->num_ranks();
    // CHECK_EQ(in->shape().elem_cnt() * num_ranks, out->shape().elem_cnt());
    OF_NCCL_CHECK(ncclAllGather(in->dptr(), out->mut_dptr(), in->shape().elem_cnt(),
                                GetNcclDataType(in->data_type()), nccl_comm->comm(),
                                ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class NcclLogical3DChangeDim1AllGatherNoncontinuous final : public user_op::OpKernel {
 public:
  NcclLogical3DChangeDim1AllGatherNoncontinuous() = default;
  ~NcclLogical3DChangeDim1AllGatherNoncontinuous() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical3DChangeDim1KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* nccl_comm = dynamic_cast<NcclLogical3DChangeDim1KernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t dtype_size = GetSizeOfDataType(in->data_type());
    int64_t data_size = GetCudaAlignedSize(out->shape().elem_cnt() * dtype_size);
    void* unpack_from_ptr = tmp_buffer->mut_dptr();
    CHECK_EQ(tmp_buffer->shape().elem_cnt(), data_size);

    CHECK_EQ(in->data_type(), out->data_type());
    const int64_t num_ranks = nccl_comm->num_ranks();
    const int64_t in_split_axis = ctx->Attr<int64_t>("in_dim1_split_axis");

    DimVector logical_shape_dim_vec;
    in->shape().ToDimVector(&logical_shape_dim_vec);
    logical_shape_dim_vec[in_split_axis] = logical_shape_dim_vec.at(in_split_axis) * num_ranks;

    // NOTE(chengcheng): Do AllGather
    CHECK_EQ(in->shape().elem_cnt() * num_ranks, out->shape().elem_cnt());
    OF_NCCL_CHECK(ncclAllGather(in->dptr(), unpack_from_ptr, in->shape().elem_cnt(),
                                GetNcclDataType(in->data_type()), nccl_comm->comm(),
                                ctx->device_ctx()->cuda_stream()));

    CHECK_GT(in_split_axis, 0);
    // NOTE(chengcheng): Do unpack.
    DimVector unpack_from_dim_vec = logical_shape_dim_vec;
    CHECK_EQ(unpack_from_dim_vec.at(in_split_axis) % num_ranks, 0);
    unpack_from_dim_vec[in_split_axis] = unpack_from_dim_vec.at(in_split_axis) / num_ranks;
    unpack_from_dim_vec.insert(unpack_from_dim_vec.begin(), num_ranks);
    const Shape unpack_from_shape(unpack_from_dim_vec);
    DimVector transpose_out_dim_vec;
    std::vector<int32_t> perm;
    FOR_RANGE(int64_t, i, 1, unpack_from_shape.NumAxes()) {
      perm.push_back(i);
      transpose_out_dim_vec.push_back(unpack_from_shape.At(i));
    }
    perm.insert(perm.begin() + in_split_axis, 0);
    transpose_out_dim_vec.insert(transpose_out_dim_vec.begin() + in_split_axis,
                                 unpack_from_shape.At(0));
    const Shape transpose_out_shape(transpose_out_dim_vec);
    NewKernelUtil<DeviceType::kGPU>::Transpose(
        ctx->device_ctx(), unpack_from_shape.NumAxes(), unpack_from_shape, transpose_out_shape,
        perm, unpack_from_shape.elem_cnt(), reinterpret_cast<const T*>(unpack_from_ptr),
        out->mut_dptr<T>());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

size_t Infer3DChangeDim1AllGatherNoncontinuousKernelTmpBufferSize(user_op::InferContext* ctx) {
  const user_op::TensorDesc* out_tensor = ctx->OutputTensorDesc("out", 0);
  return GetCudaAlignedSize(out_tensor->shape().elem_cnt()
                            * GetSizeOfDataType(out_tensor->data_type()));
}

template<typename T>
class NcclLogical3DChangeDim1ReduceScatterNoncontinuous final : public user_op::OpKernel {
 public:
  NcclLogical3DChangeDim1ReduceScatterNoncontinuous() = default;
  ~NcclLogical3DChangeDim1ReduceScatterNoncontinuous() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical3DChangeDim1KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* nccl_comm = dynamic_cast<NcclLogical3DChangeDim1KernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t dtype_size = GetSizeOfDataType(in->data_type());
    int64_t data_size = GetCudaAlignedSize(in->shape().elem_cnt() * dtype_size);
    CHECK_EQ(tmp_buffer->shape().elem_cnt(), data_size);

    CHECK_EQ(in->data_type(), out->data_type());
    const int64_t num_ranks = nccl_comm->num_ranks();
    const int64_t elem_cnt = in->shape().elem_cnt();
    const int64_t out_split_axis = ctx->Attr<int64_t>("out_dim1_split_axis");

    DimVector logical_shape_dim_vec;
    in->shape().ToDimVector(&logical_shape_dim_vec);

    DimVector transpose_in_dim_vec = logical_shape_dim_vec;
    CHECK_EQ(transpose_in_dim_vec.at(out_split_axis) % num_ranks, 0);
    transpose_in_dim_vec[out_split_axis] = transpose_in_dim_vec.at(out_split_axis) / num_ranks;
    transpose_in_dim_vec.insert(transpose_in_dim_vec.begin() + out_split_axis, num_ranks);
    const Shape transpose_in_shape(transpose_in_dim_vec);
    DimVector pack_to_dim_vec;
    std::vector<int32_t> perm;
    perm.push_back(out_split_axis);
    pack_to_dim_vec.push_back(transpose_in_shape.At(out_split_axis));
    FOR_RANGE(int64_t, i, 0, transpose_in_shape.NumAxes()) {
      if (i != out_split_axis) {
        perm.push_back(i);
        pack_to_dim_vec.push_back(transpose_in_shape.At(i));
      }
    }
    CHECK_EQ(elem_cnt, transpose_in_shape.elem_cnt());
    const Shape pack_to_shape(pack_to_dim_vec);
    CHECK_EQ(elem_cnt, pack_to_shape.elem_cnt());
    NewKernelUtil<DeviceType::kGPU>::Transpose(
        ctx->device_ctx(), transpose_in_shape.NumAxes(), transpose_in_shape, pack_to_shape, perm,
        elem_cnt, in->dptr<T>(), reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()));

    OF_NCCL_CHECK(ncclReduceScatter(tmp_buffer->dptr(), out->mut_dptr(), out->shape().elem_cnt(),
                                    GetNcclDataType(in->data_type()), ncclRedOp_t::ncclSum,
                                    nccl_comm->comm(), ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

size_t Infer3DChangeDim1ReduceScatterNoncontinuousKernelTmpBufferSize(user_op::InferContext* ctx) {
  const user_op::TensorDesc* in_tensor = ctx->OutputTensorDesc("in", 0);
  return GetCudaAlignedSize(in_tensor->shape().elem_cnt()
                            * GetSizeOfDataType(in_tensor->data_type()));
}

template<typename T>
class NcclLogical3DChangeDim2AllGatherNoncontinuous final : public user_op::OpKernel {
 public:
  NcclLogical3DChangeDim2AllGatherNoncontinuous() = default;
  ~NcclLogical3DChangeDim2AllGatherNoncontinuous() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical3DChangeDim2KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* nccl_comm = dynamic_cast<NcclLogical3DChangeDim2KernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t dtype_size = GetSizeOfDataType(in->data_type());
    int64_t data_size = GetCudaAlignedSize(out->shape().elem_cnt() * dtype_size);
    void* unpack_from_ptr = tmp_buffer->mut_dptr();
    CHECK_EQ(tmp_buffer->shape().elem_cnt(), data_size);

    CHECK_EQ(in->data_type(), out->data_type());
    const int64_t num_ranks = nccl_comm->num_ranks();
    const int64_t in_split_axis = ctx->Attr<int64_t>("in_dim2_split_axis");

    DimVector logical_shape_dim_vec;
    in->shape().ToDimVector(&logical_shape_dim_vec);
    logical_shape_dim_vec[in_split_axis] = logical_shape_dim_vec.at(in_split_axis) * num_ranks;

    // NOTE(chengcheng): Do AllGather
    CHECK_EQ(in->shape().elem_cnt() * num_ranks, out->shape().elem_cnt());
    OF_NCCL_CHECK(ncclAllGather(in->dptr(), unpack_from_ptr, in->shape().elem_cnt(),
                                GetNcclDataType(in->data_type()), nccl_comm->comm(),
                                ctx->device_ctx()->cuda_stream()));

    CHECK_GT(in_split_axis, 0);
    // NOTE(chengcheng): Do unpack.
    DimVector unpack_from_dim_vec = logical_shape_dim_vec;
    CHECK_EQ(unpack_from_dim_vec.at(in_split_axis) % num_ranks, 0);
    unpack_from_dim_vec[in_split_axis] = unpack_from_dim_vec.at(in_split_axis) / num_ranks;
    unpack_from_dim_vec.insert(unpack_from_dim_vec.begin(), num_ranks);
    const Shape unpack_from_shape(unpack_from_dim_vec);
    DimVector transpose_out_dim_vec;
    std::vector<int32_t> perm;
    FOR_RANGE(int64_t, i, 1, unpack_from_shape.NumAxes()) {
      perm.push_back(i);
      transpose_out_dim_vec.push_back(unpack_from_shape.At(i));
    }
    perm.insert(perm.begin() + in_split_axis, 0);
    transpose_out_dim_vec.insert(transpose_out_dim_vec.begin() + in_split_axis,
                                 unpack_from_shape.At(0));
    const Shape transpose_out_shape(transpose_out_dim_vec);
    NewKernelUtil<DeviceType::kGPU>::Transpose(
        ctx->device_ctx(), unpack_from_shape.NumAxes(), unpack_from_shape, transpose_out_shape,
        perm, unpack_from_shape.elem_cnt(), reinterpret_cast<const T*>(unpack_from_ptr),
        out->mut_dptr<T>());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

size_t Infer3DChangeDim2AllGatherNoncontinuousKernelTmpBufferSize(user_op::InferContext* ctx) {
  const user_op::TensorDesc* out_tensor = ctx->OutputTensorDesc("out", 0);
  return GetCudaAlignedSize(out_tensor->shape().elem_cnt()
                            * GetSizeOfDataType(out_tensor->data_type()));
}

template<typename T>
class NcclLogical3DChangeDim2All2All final : public user_op::OpKernel {
 public:
  NcclLogical3DChangeDim2All2All() = default;
  ~NcclLogical3DChangeDim2All2All() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical3DChangeDim2KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* nccl_comm = dynamic_cast<NcclLogical3DChangeDim2KernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    int64_t tmp_size = 0;
    const int64_t dtype_size = GetSizeOfDataType(in->data_type());
    int64_t data_size = GetCudaAlignedSize(in->shape().elem_cnt() * dtype_size);
    // NOTE(chengcheng): in (transpose)-> pack_to_ptr (all2all)-> unpack_from_ptr (transpose)-> out
    const char* pack_to_ptr = in->dptr<char>();
    char* unpack_from_ptr = out->mut_dptr<char>();
    if (tmp_buffer) { tmp_size = tmp_buffer->shape().elem_cnt(); }
    CHECK(tmp_size == 0 || tmp_size == data_size || tmp_size == data_size * 2);

    CHECK_EQ(in->data_type(), out->data_type());
    const int64_t num_ranks = nccl_comm->num_ranks();
    CHECK_EQ(in->shape().elem_cnt(), out->shape().elem_cnt());
    const int64_t elem_cnt = in->shape().elem_cnt();
    const int64_t in_split_axis = ctx->Attr<int64_t>("in_dim2_split_axis");
    const int64_t out_split_axis = ctx->Attr<int64_t>("out_dim2_split_axis");

    DimVector logical_shape_dim_vec;
    in->shape().ToDimVector(&logical_shape_dim_vec);
    logical_shape_dim_vec[in_split_axis] = logical_shape_dim_vec.at(in_split_axis) * num_ranks;

    if (out_split_axis != 0) {
      // NOTE(chengcheng): Do pack. Need transpose in -> pack_to
      // pack use temp buffer offset: [0, data_size]
      pack_to_ptr = tmp_buffer->dptr<char>();
      DimVector transpose_in_dim_vec = logical_shape_dim_vec;
      CHECK_EQ(transpose_in_dim_vec.at(in_split_axis) % num_ranks, 0);
      transpose_in_dim_vec[in_split_axis] = transpose_in_dim_vec.at(in_split_axis) / num_ranks;
      CHECK_EQ(transpose_in_dim_vec.at(out_split_axis) % num_ranks, 0);
      transpose_in_dim_vec[out_split_axis] = transpose_in_dim_vec.at(out_split_axis) / num_ranks;
      transpose_in_dim_vec.insert(transpose_in_dim_vec.begin() + out_split_axis, num_ranks);
      const Shape transpose_in_shape(transpose_in_dim_vec);
      DimVector pack_to_dim_vec;
      std::vector<int32_t> perm;
      perm.push_back(out_split_axis);
      pack_to_dim_vec.push_back(transpose_in_shape.At(out_split_axis));
      FOR_RANGE(int64_t, i, 0, transpose_in_shape.NumAxes()) {
        if (i != out_split_axis) {
          perm.push_back(i);
          pack_to_dim_vec.push_back(transpose_in_shape.At(i));
        }
      }
      CHECK_EQ(elem_cnt, transpose_in_shape.elem_cnt());
      const Shape pack_to_shape(pack_to_dim_vec);
      CHECK_EQ(elem_cnt, pack_to_shape.elem_cnt());
      NewKernelUtil<DeviceType::kGPU>::Transpose(
          ctx->device_ctx(), transpose_in_shape.NumAxes(), transpose_in_shape, pack_to_shape, perm,
          elem_cnt, in->dptr<T>(), reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()));
    }

    if (in_split_axis != 0) {
      // NOTE(chengcheng): Do unpack. Need transpose unpack_from -> out
      // unpack use temp buffer offset: [tmp_size - data_size, tmp_size]
      unpack_from_ptr = tmp_buffer->mut_dptr<char>() + (tmp_size - data_size);
    }

    {
      // NOTE(chengcheng): Do S2S
      OF_NCCL_CHECK(ncclGroupStart());
      const int64_t elem_per_chunk = elem_cnt / num_ranks;
      const int64_t chunk_size = elem_per_chunk * dtype_size;
      for (int64_t j = 0; j < num_ranks; ++j) {
        OF_NCCL_CHECK(ncclSend(reinterpret_cast<const void*>(
                                   reinterpret_cast<const char*>(pack_to_ptr) + j * chunk_size),
                               elem_per_chunk, GetNcclDataType(in->data_type()), j,
                               nccl_comm->comm(), ctx->device_ctx()->cuda_stream()));
        OF_NCCL_CHECK(ncclRecv(
            reinterpret_cast<void*>(reinterpret_cast<char*>(unpack_from_ptr) + j * chunk_size),
            elem_per_chunk, GetNcclDataType(in->data_type()), j, nccl_comm->comm(),
            ctx->device_ctx()->cuda_stream()));
      }
      OF_NCCL_CHECK(ncclGroupEnd());
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
      const Shape unpack_from_shape(unpack_from_dim_vec);
      DimVector transpose_out_dim_vec;
      std::vector<int32_t> perm;
      FOR_RANGE(int64_t, i, 1, unpack_from_shape.NumAxes()) {
        perm.push_back(i);
        transpose_out_dim_vec.push_back(unpack_from_shape.At(i));
      }
      perm.insert(perm.begin() + in_split_axis, 0);
      transpose_out_dim_vec.insert(transpose_out_dim_vec.begin() + in_split_axis,
                                   unpack_from_shape.At(0));
      const Shape transpose_out_shape(transpose_out_dim_vec);
      NewKernelUtil<DeviceType::kGPU>::Transpose(
          ctx->device_ctx(), unpack_from_shape.NumAxes(), unpack_from_shape, transpose_out_shape,
          perm, unpack_from_shape.elem_cnt(), reinterpret_cast<const T*>(unpack_from_ptr),
          out->mut_dptr<T>());
    }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

size_t Infer3DChangeDim2All2AllKernelTmpBufferSize(user_op::InferContext* ctx) {
  size_t ret = 0;
  const user_op::TensorDesc* in_tensor = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  size_t tensor_byte_size =
      GetCudaAlignedSize(in_tensor->shape().elem_cnt() * GetSizeOfDataType(in_tensor->data_type()));
  const cfg::SbpParallel& in_sbp =
      ctx->ParallelDistribution4ArgNameAndIndex("in", 0).sbp_parallel(2);
  const cfg::SbpParallel& out_sbp =
      ctx->ParallelDistribution4ArgNameAndIndex("out", 0).sbp_parallel(2);
  CHECK(in_sbp.has_split_parallel() && out_sbp.has_split_parallel());
  if (in_sbp.split_parallel().axis() != 0) { ret += tensor_byte_size; }
  if (out_sbp.split_parallel().axis() != 0) { ret += tensor_byte_size; }
  return ret;
}

class NcclLogical3DChangeDim2ReduceScatter final : public user_op::OpKernel {
 public:
  NcclLogical3DChangeDim2ReduceScatter() = default;
  ~NcclLogical3DChangeDim2ReduceScatter() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical3DChangeDim2KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* nccl_comm = dynamic_cast<NcclLogical3DChangeDim2KernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->data_type(), out->data_type());
    const int64_t num_ranks = nccl_comm->num_ranks();
    CHECK_EQ(in->shape().elem_cnt(), out->shape().elem_cnt() * num_ranks);
    OF_NCCL_CHECK(ncclReduceScatter(in->dptr(), out->mut_dptr(), out->shape().elem_cnt(),
                                    GetNcclDataType(in->data_type()), ncclRedOp_t::ncclSum,
                                    nccl_comm->comm(), ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class NcclLogical3DChangeDim1ReduceScatter final : public user_op::OpKernel {
 public:
  NcclLogical3DChangeDim1ReduceScatter() = default;
  ~NcclLogical3DChangeDim1ReduceScatter() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical3DChangeDim1KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* nccl_comm = dynamic_cast<NcclLogical3DChangeDim1KernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->data_type(), out->data_type());
    // const int64_t num_ranks = nccl_comm->num_ranks();
    // CHECK_EQ(in->shape().elem_cnt(), out->shape().elem_cnt() * num_ranks);
    OF_NCCL_CHECK(ncclReduceScatter(in->dptr(), out->mut_dptr(), out->shape().elem_cnt(),
                                    GetNcclDataType(in->data_type()), ncclRedOp_t::ncclSum,
                                    nccl_comm->comm(), ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class NcclLogical3DChangeDim2ReduceScatterNoncontinuous final : public user_op::OpKernel {
 public:
  NcclLogical3DChangeDim2ReduceScatterNoncontinuous() = default;
  ~NcclLogical3DChangeDim2ReduceScatterNoncontinuous() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical3DChangeDim2KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* nccl_comm = dynamic_cast<NcclLogical3DChangeDim2KernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t dtype_size = GetSizeOfDataType(in->data_type());
    int64_t data_size = GetCudaAlignedSize(in->shape().elem_cnt() * dtype_size);
    CHECK_EQ(tmp_buffer->shape().elem_cnt(), data_size);

    CHECK_EQ(in->data_type(), out->data_type());
    const int64_t num_ranks = nccl_comm->num_ranks();
    const int64_t elem_cnt = in->shape().elem_cnt();
    const int64_t out_split_axis = ctx->Attr<int64_t>("out_dim2_split_axis");

    DimVector logical_shape_dim_vec;
    in->shape().ToDimVector(&logical_shape_dim_vec);

    DimVector transpose_in_dim_vec = logical_shape_dim_vec;
    CHECK_EQ(transpose_in_dim_vec.at(out_split_axis) % num_ranks, 0);
    transpose_in_dim_vec[out_split_axis] = transpose_in_dim_vec.at(out_split_axis) / num_ranks;
    transpose_in_dim_vec.insert(transpose_in_dim_vec.begin() + out_split_axis, num_ranks);
    const Shape transpose_in_shape(transpose_in_dim_vec);
    DimVector pack_to_dim_vec;
    std::vector<int32_t> perm;
    perm.push_back(out_split_axis);
    pack_to_dim_vec.push_back(transpose_in_shape.At(out_split_axis));
    FOR_RANGE(int64_t, i, 0, transpose_in_shape.NumAxes()) {
      if (i != out_split_axis) {
        perm.push_back(i);
        pack_to_dim_vec.push_back(transpose_in_shape.At(i));
      }
    }
    CHECK_EQ(elem_cnt, transpose_in_shape.elem_cnt());
    const Shape pack_to_shape(pack_to_dim_vec);
    CHECK_EQ(elem_cnt, pack_to_shape.elem_cnt());
    NewKernelUtil<DeviceType::kGPU>::Transpose(
        ctx->device_ctx(), transpose_in_shape.NumAxes(), transpose_in_shape, pack_to_shape, perm,
        elem_cnt, in->dptr<T>(), reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()));

    OF_NCCL_CHECK(ncclReduceScatter(tmp_buffer->dptr(), out->mut_dptr(), out->shape().elem_cnt(),
                                    GetNcclDataType(in->data_type()), ncclRedOp_t::ncclSum,
                                    nccl_comm->comm(), ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

size_t Infer3DChangeDim2ReduceScatterNoncontinuousKernelTmpBufferSize(user_op::InferContext* ctx) {
  const user_op::TensorDesc* in_tensor = ctx->OutputTensorDesc("in", 0);
  return GetCudaAlignedSize(in_tensor->shape().elem_cnt()
                            * GetSizeOfDataType(in_tensor->data_type()));
}

REGISTER_USER_KERNEL("_nccl_logical_3D_change_dim1_all_gather")
    .SetCreateFn<NcclLogical3DChangeDim1AllGather>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");

REGISTER_USER_KERNEL("_nccl_logical_3D_change_dim1_reduce_scatter")
    .SetCreateFn<NcclLogical3DChangeDim1ReduceScatter>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");

REGISTER_USER_KERNEL("_nccl_logical_3D_change_dim2_all_gather")
    .SetCreateFn<NcclLogical3DChangeDim2AllGather>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");

REGISTER_USER_KERNEL("_nccl_logical_3D_change_dim2_reduce_scatter")
    .SetCreateFn<NcclLogical3DChangeDim2ReduceScatter>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");

#define REGISTER_3D_CHANGE_DIM1_ALLGATHER_NONCONTINUOUS_KERNEL(dtype)                   \
  REGISTER_USER_KERNEL("_nccl_logical_3D_change_dim1_all_gather_noncontinuous")         \
      .SetCreateFn<NcclLogical3DChangeDim1AllGatherNoncontinuous<dtype>>()              \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                               \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)   \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(Infer3DChangeDim1AllGatherNoncontinuousKernelTmpBufferSize);

REGISTER_3D_CHANGE_DIM1_ALLGATHER_NONCONTINUOUS_KERNEL(int8_t)
REGISTER_3D_CHANGE_DIM1_ALLGATHER_NONCONTINUOUS_KERNEL(int32_t)
REGISTER_3D_CHANGE_DIM1_ALLGATHER_NONCONTINUOUS_KERNEL(int64_t)
REGISTER_3D_CHANGE_DIM1_ALLGATHER_NONCONTINUOUS_KERNEL(float)
REGISTER_3D_CHANGE_DIM1_ALLGATHER_NONCONTINUOUS_KERNEL(double)
REGISTER_3D_CHANGE_DIM1_ALLGATHER_NONCONTINUOUS_KERNEL(float16)

#define REGISTER_3D_CHANGE_DIM1_REDUCE_SCATTER_NONCONTINUOUS_KERNEL(dtype)              \
  REGISTER_USER_KERNEL("_nccl_logical_3D_change_dim1_reduce_scatter_noncontinuous")     \
      .SetCreateFn<NcclLogical3DChangeDim1ReduceScatterNoncontinuous<dtype>>()          \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                               \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)   \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(Infer3DChangeDim1ReduceScatterNoncontinuousKernelTmpBufferSize);

REGISTER_3D_CHANGE_DIM1_REDUCE_SCATTER_NONCONTINUOUS_KERNEL(int8_t)
REGISTER_3D_CHANGE_DIM1_REDUCE_SCATTER_NONCONTINUOUS_KERNEL(int32_t)
REGISTER_3D_CHANGE_DIM1_REDUCE_SCATTER_NONCONTINUOUS_KERNEL(int64_t)
REGISTER_3D_CHANGE_DIM1_REDUCE_SCATTER_NONCONTINUOUS_KERNEL(float)
REGISTER_3D_CHANGE_DIM1_REDUCE_SCATTER_NONCONTINUOUS_KERNEL(double)
REGISTER_3D_CHANGE_DIM1_REDUCE_SCATTER_NONCONTINUOUS_KERNEL(float16)

#define REGISTER_3D_CHANGE_DIM2_ALLGATHER_NONCONTINUOUS_KERNEL(dtype)                   \
  REGISTER_USER_KERNEL("_nccl_logical_3D_change_dim2_all_gather_noncontinuous")         \
      .SetCreateFn<NcclLogical3DChangeDim2AllGatherNoncontinuous<dtype>>()              \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                               \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)   \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(Infer3DChangeDim2AllGatherNoncontinuousKernelTmpBufferSize);

REGISTER_3D_CHANGE_DIM2_ALLGATHER_NONCONTINUOUS_KERNEL(int8_t)
REGISTER_3D_CHANGE_DIM2_ALLGATHER_NONCONTINUOUS_KERNEL(int32_t)
REGISTER_3D_CHANGE_DIM2_ALLGATHER_NONCONTINUOUS_KERNEL(int64_t)
REGISTER_3D_CHANGE_DIM2_ALLGATHER_NONCONTINUOUS_KERNEL(float)
REGISTER_3D_CHANGE_DIM2_ALLGATHER_NONCONTINUOUS_KERNEL(double)
REGISTER_3D_CHANGE_DIM2_ALLGATHER_NONCONTINUOUS_KERNEL(float16)

#define REGISTER_3D_CHANGE_DIM2_REDUCE_SCATTER_NONCONTINUOUS_KERNEL(dtype)              \
  REGISTER_USER_KERNEL("_nccl_logical_3D_change_dim2_reduce_scatter_noncontinuous")     \
      .SetCreateFn<NcclLogical3DChangeDim2ReduceScatterNoncontinuous<dtype>>()          \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                               \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)   \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(Infer3DChangeDim2ReduceScatterNoncontinuousKernelTmpBufferSize);

REGISTER_3D_CHANGE_DIM2_REDUCE_SCATTER_NONCONTINUOUS_KERNEL(int8_t)
REGISTER_3D_CHANGE_DIM2_REDUCE_SCATTER_NONCONTINUOUS_KERNEL(int32_t)
REGISTER_3D_CHANGE_DIM2_REDUCE_SCATTER_NONCONTINUOUS_KERNEL(int64_t)
REGISTER_3D_CHANGE_DIM2_REDUCE_SCATTER_NONCONTINUOUS_KERNEL(float)
REGISTER_3D_CHANGE_DIM2_REDUCE_SCATTER_NONCONTINUOUS_KERNEL(double)
REGISTER_3D_CHANGE_DIM2_REDUCE_SCATTER_NONCONTINUOUS_KERNEL(float16)

#define REGISTER_3D_CHANGE_DIM2_ALL2ALL_KERNEL(dtype)                                   \
  REGISTER_USER_KERNEL("_nccl_logical_3D_change_dim2_all2all")                          \
      .SetCreateFn<NcclLogical3DChangeDim2All2All<dtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                               \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)   \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(Infer3DChangeDim2All2AllKernelTmpBufferSize);

REGISTER_3D_CHANGE_DIM2_ALL2ALL_KERNEL(int8_t)
REGISTER_3D_CHANGE_DIM2_ALL2ALL_KERNEL(int32_t)
REGISTER_3D_CHANGE_DIM2_ALL2ALL_KERNEL(int64_t)
REGISTER_3D_CHANGE_DIM2_ALL2ALL_KERNEL(float)
REGISTER_3D_CHANGE_DIM2_ALL2ALL_KERNEL(double)
REGISTER_3D_CHANGE_DIM2_ALL2ALL_KERNEL(float16)

}  // namespace oneflow

#endif  // WITH_CUDA && NCCL_VERSION_CODE > 2700
