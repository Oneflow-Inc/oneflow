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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/slice_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"

namespace oneflow {

namespace {

const int SPLIT_AXIS_FOR_BROADCAST = -1;

// [start, end)
int64_t GetSizeInSlice(const int64_t start, const int64_t end, const int64_t step) {
  if (end <= start) { return 0; }
  return (end - start - 1) / step + 1;
}

struct SliceContext final {
  SliceContext(int64_t split_axis, int64_t lower, int64_t upper, int64_t logical_length)
      : split_axis(split_axis), lower(lower), upper(upper), logical_length(logical_length) {}

  // These fields shows how the logical tensor is splited.
  // The logical tensor is splited on the axis `split_axis`
  // The physical tensor on current device is in the range [lower, upper)
  // The length of the logical tensor on `split_axis` is `logical_length`
  // Example:
  // Variable shape = (8, 7, 6, 5), sbp = S(0), on 4 devices, then on the first card:
  // split_axis = 0
  // lower = 0
  // upper = 2
  // logical_length = 8
  const int64_t split_axis;
  const int64_t lower;
  const int64_t upper;
  const int64_t logical_length;
};

void ConstructSliceParamsLarge(const SliceContext& ctx, const std::vector<int64_t>& start_vec,
                               const std::vector<int64_t>& stop_vec,
                               const std::vector<int64_t>& step_vec, const ShapeView& shape,
                               SliceParams* slice_param) {
  const int64_t ndim = shape.NumAxes();
  CHECK_LE(ndim, kSliceMaxDims);
  CHECK_EQ(start_vec.size(), ndim);
  CHECK_EQ(stop_vec.size(), ndim);
  CHECK_EQ(step_vec.size(), ndim);

  std::memset(slice_param, 0, sizeof(SliceParams));
  slice_param->ndim = ndim;
  FOR_RANGE(int, i, 0, slice_param->ndim) {
    const int64_t dim_size = shape.At(i);
    const int64_t start_in_full_large = start_vec.at(i);
    const int64_t stop_in_full_large = stop_vec.at(i);
    const int64_t step = step_vec.at(i);
    CHECK_GT(step, 0);
    int64_t start_in_splitted_large = start_in_full_large;
    int64_t stop_in_splitted_large = stop_in_full_large;
    // large tensor has split sbp attribute
    if (i == ctx.split_axis) {
      if (start_in_splitted_large < ctx.lower) {
        start_in_splitted_large =
            ctx.lower + (step - (ctx.lower - start_in_splitted_large) % step) % step;
      }
      start_in_splitted_large = std::min(std::max(start_in_splitted_large, ctx.lower), ctx.upper);
      stop_in_splitted_large = std::min(std::max(stop_in_splitted_large, ctx.lower), ctx.upper);
      start_in_splitted_large -= ctx.lower;
      stop_in_splitted_large -= ctx.lower;
    }
    const int64_t slice_size =
        GetSizeInSlice(start_in_splitted_large, stop_in_splitted_large, step);
    slice_param->dims[i] = dim_size;
    slice_param->start[i] = start_in_splitted_large;
    slice_param->step[i] = step;
    slice_param->size[i] = slice_size;
  }
}

void ConstructSliceParamsSmall(const SliceContext& ctx, const std::vector<int64_t>& start_vec,
                               const std::vector<int64_t>& stop_vec,
                               const std::vector<int64_t>& step_vec, const ShapeView& shape,
                               SliceParams* slice_param) {
  const int64_t ndim = shape.NumAxes();
  CHECK_LE(ndim, kSliceMaxDims);
  CHECK_EQ(start_vec.size(), ndim);
  CHECK_EQ(stop_vec.size(), ndim);
  CHECK_EQ(step_vec.size(), ndim);

  std::memset(slice_param, 0, sizeof(SliceParams));
  slice_param->ndim = ndim;
  FOR_RANGE(int, i, 0, slice_param->ndim) {
    const int64_t start_in_full_large = start_vec.at(i);
    const int64_t step = step_vec.at(i);
    CHECK_GT(step, 0);
    // small tensor has broadcast/partialsum sbp attribute
    const int64_t dim_size = shape.At(i);
    int64_t start_in_full_small = 0;
    int64_t stop_in_full_small = dim_size;
    if (i == ctx.split_axis) {
      start_in_full_small = GetSizeInSlice(start_in_full_large, ctx.lower, step);
      stop_in_full_small = GetSizeInSlice(start_in_full_large, ctx.upper, step);
      start_in_full_small = std::min(std::max<int64_t>(start_in_full_small, 0), dim_size);
      stop_in_full_small = std::min(std::max<int64_t>(stop_in_full_small, 0), dim_size);
    }
    const int64_t slice_size = stop_in_full_small - start_in_full_small;
    slice_param->dims[i] = dim_size;
    slice_param->start[i] = start_in_full_small;
    slice_param->step[i] = 1;
    slice_param->size[i] = slice_size;
  }
}

SliceParams ConstructSliceParams(user_op::KernelComputeContext* ctx, const user_op::Tensor* entire,
                                 const user_op::Tensor* sliced) {
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  const int64_t ndim = entire->shape().NumAxes();
  CHECK_LE(ndim, kSliceMaxDims);
  CHECK_EQ(sliced->shape().NumAxes(), ndim);
  CHECK_EQ(start_vec.size(), ndim);
  CHECK_EQ(stop_vec.size(), ndim);
  CHECK_EQ(step_vec.size(), ndim);

  SliceParams params;
  std::memset(&params, 0, sizeof(SliceParams));
  params.ndim = ndim;
  FOR_RANGE(int, i, 0, params.ndim) {
    const int64_t dim_size = entire->shape().At(i);
    const int64_t slice_size = sliced->shape().At(i);
    const int64_t step = step_vec.at(i);
    CHECK_NE(step, 0);
    const int64_t start = RegulateSliceStart(start_vec.at(i), dim_size);
    const int64_t stop = RegulateSliceStop(stop_vec.at(i), dim_size);
    if (step > 0) {
      CHECK_LT(start + step * (slice_size - 1), stop);
    } else {
      CHECK_GT(start + step * (slice_size - 1), stop);
    }
    params.dims[i] = dim_size;
    params.start[i] = start;
    params.step[i] = step;
    params.size[i] = slice_size;
  }
  return params;
}

}  // namespace

template<DeviceType device_type, typename T>
class SliceKernel final : public user_op::OpKernel {
 public:
  SliceKernel() = default;
  ~SliceKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    SliceParams params = ConstructSliceParams(ctx, x_tensor, y_tensor);
    SliceKernelUtil<device_type, T>::Forward(ctx->device_ctx(), params, x_tensor->dptr<T>(),
                                             y_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class SliceGradKernel final : public user_op::OpKernel {
 public:
  SliceGradKernel() = default;
  ~SliceGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    size_t dx_byte_size = dx_tensor->shape().elem_cnt() * sizeof(T);
    Memset<device_type>(ctx->device_ctx(), dx_tensor->mut_dptr<T>(), 0, dx_byte_size);
    SliceParams params = ConstructSliceParams(ctx, dx_tensor, dy_tensor);
    SliceKernelUtil<device_type, T>::Backward(ctx->device_ctx(), params, dy_tensor->dptr<T>(),
                                              dx_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<int NDIM, typename T>
void WriteSlice(user_op::KernelComputeContext* ctx, const user_op::Tensor* src,
                user_op::Tensor* dst, const SliceContext& slice_ctx,
                const bool from_large_to_small) {
  const user_op::Tensor* large = from_large_to_small ? src : dst;
  const user_op::Tensor* small = from_large_to_small ? dst : src;
  if (slice_ctx.split_axis != SPLIT_AXIS_FOR_BROADCAST) {
    CHECK_EQ(large->shape().At(slice_ctx.split_axis), slice_ctx.upper - slice_ctx.lower);
  }

  std::vector<int64_t> positive_start_vec;
  std::vector<int64_t> positive_stop_vec;
  const std::vector<int64_t> start_attr = ctx->Attr<std::vector<int64_t>>("start");
  const std::vector<int64_t> stop_attr = ctx->Attr<std::vector<int64_t>>("stop");
  const std::vector<int64_t> step_attr = ctx->Attr<std::vector<int64_t>>("step");
  const int64_t ndim = start_attr.size();
  for (int i = 0; i < ndim; i++) {
    const int64_t dim_size = large->shape().At(i);
    positive_start_vec.push_back(RegulateSliceStart(
        start_attr.at(i), i == slice_ctx.split_axis ? slice_ctx.logical_length : dim_size));
    positive_stop_vec.push_back(RegulateSliceStop(
        stop_attr.at(i), i == slice_ctx.split_axis ? slice_ctx.logical_length : dim_size));
  }
  SliceParams large_slice_param;
  SliceParams small_slice_param;
  ConstructSliceParamsLarge(slice_ctx, positive_start_vec, positive_stop_vec, step_attr,
                            large->shape(), &large_slice_param);
  ConstructSliceParamsSmall(slice_ctx, positive_start_vec, positive_stop_vec, step_attr,
                            small->shape(), &small_slice_param);
  CHECK_EQ(large_slice_param.elem_cnt(), small_slice_param.elem_cnt());

  const int64_t elem_cnt = large_slice_param.elem_cnt();
  SliceIndexHelper<NDIM> entire_splitted_large_idx_cvtr(large_slice_param.dims);
  SliceIndexHelper<NDIM> sliced_splitted_large_idx_cvtr(large_slice_param.size);
  SliceIndexHelper<NDIM> entire_full_small_idx_cvtr(small_slice_param.dims);
  SliceIndexHelper<NDIM> sliced_full_small_idx_cvtr(small_slice_param.size);
  // Calculate the length of continuous part
  int cnt = 1;
  for (int i = NDIM - 1; i >= 0; i--) {
    if (large_slice_param.step[i] == 1) { cnt *= large_slice_param.size[i]; }
    if (!large_slice_param.IsFullSlice(i) || !small_slice_param.IsFullSlice(i)) { break; }
  }
  const auto* src_ptr = src->dptr<T>();
  auto* dst_ptr = dst->mut_dptr<T>();
  for (int i = 0; i < elem_cnt; i += cnt) {
    const int64_t large_offset = SliceOffsetToEntireOffset<NDIM>(
        i, large_slice_param, entire_splitted_large_idx_cvtr, sliced_splitted_large_idx_cvtr);
    const int64_t small_offset = SliceOffsetToEntireOffset<NDIM>(
        i, small_slice_param, entire_full_small_idx_cvtr, sliced_full_small_idx_cvtr);
    const int64_t src_offset = from_large_to_small ? large_offset : small_offset;
    const int64_t dst_offset = from_large_to_small ? small_offset : large_offset;
    AutoMemcpy(ctx->device_ctx(), dst_ptr + dst_offset, src_ptr + src_offset,
               cnt * GetSizeOfDataType(src->data_type()), src->mem_case(), dst->mem_case());
  }
}

#define MAKE_WRITE_SLICE_SWITCH_ENTRY(func_name, N, T) func_name<N, T>
DEFINE_STATIC_SWITCH_FUNC(void, WriteSlice, MAKE_WRITE_SLICE_SWITCH_ENTRY,
                          MAKE_NDIM_CTRV_SEQ(DIM_SEQ),
                          MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ
#if defined(WITH_CUDA)
                                                      HALF_DATA_TYPE_SEQ
#endif
                                                  ));
#undef MAKE_WRITE_SLICE_SWITCH_ENTRY

std::shared_ptr<user_op::OpKernelState> CreateSliceState(user_op::KernelInitContext* ctx,
                                                         const std::string& large_tensor_name) {
  const SbpParallel& in_sbp = ctx->SbpParallel4ArgNameAndIndex(large_tensor_name, 0);
  if (in_sbp.has_split_parallel() && ctx->parallel_ctx().parallel_num() > 1) {
    const user_op::TensorDesc* in_logical_desc =
        ctx->LogicalTensorDesc4ArgNameAndIndex(large_tensor_name, 0);
    const auto split_axis = in_sbp.split_parallel().axis();
    const int64_t split_dim_size = in_logical_desc->shape().At(split_axis);
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    BalancedSplitter bs(split_dim_size, ctx->parallel_ctx().parallel_num());
    return std::make_shared<OpKernelStateWrapper<SliceContext>>(
        split_axis, bs.At(parallel_id).begin(), bs.At(parallel_id).end(), split_dim_size);
  } else if (in_sbp.has_broadcast_parallel() || ctx->parallel_ctx().parallel_num() == 1) {
    // split_axis == SPLIT_AXIS_FOR_BROADCAST means the sbp attribute is broadcast instead of split
    return std::make_shared<OpKernelStateWrapper<SliceContext>>(SPLIT_AXIS_FOR_BROADCAST, 0, 0, 0);
  } else {
    // TODO(jianhao): support partialsum
    UNIMPLEMENTED();
  }
}

template<typename T>
class LogicalSliceKernel final : public user_op::OpKernel {
 public:
  LogicalSliceKernel() = default;
  ~LogicalSliceKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const SbpParallel& x_sbp = ctx->SbpParallel4ArgNameAndIndex("x", 0);
    const SbpParallel& y_sbp = ctx->SbpParallel4ArgNameAndIndex("y", 0);
    if (ctx->parallel_ctx().parallel_num() > 1) {
      if (x_sbp.has_split_parallel()) {
        CHECK(y_sbp.has_partial_sum_parallel());
      } else {
        CHECK(y_sbp.has_broadcast_parallel());
      }
    }
    return CreateSliceState(ctx, "x");
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const SliceContext& slice_ctx = dynamic_cast<OpKernelStateWrapper<SliceContext>*>(state)->Get();
    if (y_tensor->mem_case().has_host_mem()) {
      memset(y_tensor->mut_dptr(), 0,
             y_tensor->shape().elem_cnt() * GetSizeOfDataType(y_tensor->data_type()));
    } else if (y_tensor->mem_case().has_device_cuda_mem()) {
#if defined(WITH_CUDA)
      cudaMemset(y_tensor->mut_dptr(), 0,
                 y_tensor->shape().elem_cnt() * GetSizeOfDataType(y_tensor->data_type()));
#else
      UNIMPLEMENTED();
#endif
    } else {
      UNIMPLEMENTED();
    }
    SwitchWriteSlice(SwitchCase(y_tensor->shape().NumAxes(), y_tensor->data_type()), ctx, x_tensor,
                     y_tensor, slice_ctx, true);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class LogicalSliceAssignKernel final : public user_op::OpKernel {
 public:
  LogicalSliceAssignKernel() = default;
  ~LogicalSliceAssignKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const SbpParallel& value_sbp = ctx->SbpParallel4ArgNameAndIndex("value", 0);
    if (ctx->parallel_ctx().parallel_num() > 1) { CHECK(value_sbp.has_broadcast_parallel()); }
    return CreateSliceState(ctx, "ref");
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* value_tensor = ctx->Tensor4ArgNameAndIndex("value", 0);
    user_op::Tensor* ref_tensor = ctx->Tensor4ArgNameAndIndex("ref", 0);
    const SliceContext& slice_ctx = dynamic_cast<OpKernelStateWrapper<SliceContext>*>(state)->Get();
    SwitchWriteSlice(SwitchCase(value_tensor->shape().NumAxes(), value_tensor->data_type()), ctx,
                     value_tensor, ref_tensor, slice_ctx, false);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

template<DeviceType device_type, typename T>
class SliceUpdateKernel final : public user_op::OpKernel {
 public:
  SliceUpdateKernel() = default;
  ~SliceUpdateKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* update_tensor = ctx->Tensor4ArgNameAndIndex("update", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    Memcpy<device_type>(ctx->device_ctx(), y_tensor->mut_dptr<T>(), x_tensor->dptr<T>(),
                        y_tensor->shape().elem_cnt() * sizeof(T));
    SliceParams params = ConstructSliceParams(ctx, y_tensor, update_tensor);
    SliceKernelUtil<device_type, T>::Backward(ctx->device_ctx(), params, update_tensor->dptr<T>(),
                                              y_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SLICE_KERNELS(device, dtype)                                                   \
  REGISTER_USER_KERNEL("slice").SetCreateFn<SliceKernel<device, dtype>>().SetIsMatchedHob(      \
      (user_op::HobDeviceTag() == device)                                                       \
      & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));                           \
  REGISTER_USER_KERNEL("slice_grad")                                                            \
      .SetCreateFn<SliceGradKernel<device, dtype>>()                                            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));         \
  REGISTER_USER_KERNEL("slice_update")                                                          \
      .SetCreateFn<SliceUpdateKernel<device, dtype>>()                                          \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                      \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)            \
                       & (user_op::HobDataType("update", 0) == GetDataType<dtype>::value))      \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));                          \
        return Maybe<void>::Ok();                                                               \
      });

#define REGISTER_SLICE_KERNELS_WITH_DEVICE(device) \
  REGISTER_SLICE_KERNELS(device, float)            \
  REGISTER_SLICE_KERNELS(device, double)           \
  REGISTER_SLICE_KERNELS(device, int32_t)          \
  REGISTER_SLICE_KERNELS(device, int64_t)          \
  REGISTER_SLICE_KERNELS(device, int8_t)           \
  REGISTER_SLICE_KERNELS(device, uint8_t)

REGISTER_SLICE_KERNELS_WITH_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_SLICE_KERNELS_WITH_DEVICE(DeviceType::kGPU)
REGISTER_SLICE_KERNELS(DeviceType::kGPU, float16)
#endif

#define REGISTER_LOGICAL_SLICE_ASSIGN_AND_LOGICAL_SLICE_KERNELS(dtype)               \
  REGISTER_USER_KERNEL("logical_slice_assign")                                       \
      .SetCreateFn<LogicalSliceAssignKernel<dtype>>()                                \
      .SetIsMatchedHob(user_op::HobDataType("ref", 0) == GetDataType<dtype>::value); \
  REGISTER_USER_KERNEL("logical_slice")                                              \
      .SetCreateFn<LogicalSliceKernel<dtype>>()                                      \
      .SetIsMatchedHob(user_op::HobDataType("x", 0) == GetDataType<dtype>::value);

REGISTER_LOGICAL_SLICE_ASSIGN_AND_LOGICAL_SLICE_KERNELS(float)
REGISTER_LOGICAL_SLICE_ASSIGN_AND_LOGICAL_SLICE_KERNELS(double)
REGISTER_LOGICAL_SLICE_ASSIGN_AND_LOGICAL_SLICE_KERNELS(int32_t)
REGISTER_LOGICAL_SLICE_ASSIGN_AND_LOGICAL_SLICE_KERNELS(int64_t)
REGISTER_LOGICAL_SLICE_ASSIGN_AND_LOGICAL_SLICE_KERNELS(int8_t)
REGISTER_LOGICAL_SLICE_ASSIGN_AND_LOGICAL_SLICE_KERNELS(uint8_t)
#ifdef WITH_CUDA
REGISTER_LOGICAL_SLICE_ASSIGN_AND_LOGICAL_SLICE_KERNELS(float16)
#endif

}  // namespace oneflow
