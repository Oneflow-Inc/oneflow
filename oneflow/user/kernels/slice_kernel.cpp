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

namespace oneflow {

namespace {

// [start, end)
int64_t get_size_in_slice(const int64_t start, const int64_t end, const int64_t step) {
  if (end <= start) { return 0; }
  return (end - start - 1) / step + 1;
}

class SliceState final : public user_op::OpKernelState {
 public:
  SliceState(int64_t split_axis, int64_t lower, int64_t upper, int64_t length)
      : split_axis_(split_axis), lower_(lower), upper_(upper), length_(length) {}
  ~SliceState() override = default;

  int64_t split_axis() const { return split_axis_; }
  int64_t lower() const { return lower_; }
  int64_t upper() const { return upper_; }
  int64_t length() const { return length_; }

 private:
  const int64_t split_axis_;
  const int64_t lower_;
  const int64_t upper_;
  const int64_t length_;
};

std::pair<SliceParams, SliceParams> ConstructSliceParams(
    user_op::KernelComputeContext* ctx, const user_op::Tensor* large, const user_op::Tensor* small,
    const int64_t split_axis, const int64_t lower, const int64_t upper, const int64_t length) {
  const auto& start_vec = ctx->Attr<std::vector<int64_t>>("start");
  const auto& stop_vec = ctx->Attr<std::vector<int64_t>>("stop");
  const auto& step_vec = ctx->Attr<std::vector<int64_t>>("step");
  const int64_t ndim = large->shape().NumAxes();
  CHECK_LE(ndim, kSliceMaxDims);
  CHECK_EQ(small->shape().NumAxes(), ndim);
  CHECK_EQ(start_vec.size(), ndim);
  CHECK_EQ(stop_vec.size(), ndim);
  CHECK_EQ(step_vec.size(), ndim);

  SliceParams large_slice_param, small_slice_param;
  std::memset(&large_slice_param, 0, sizeof(SliceParams));
  std::memset(&small_slice_param, 0, sizeof(SliceParams));
  large_slice_param.ndim = ndim;
  small_slice_param.ndim = ndim;
  FOR_RANGE(int, i, 0, large_slice_param.ndim) {
    const int64_t dim_size = large->shape().At(i);
    int64_t small_size = small->shape().At(i);
    const int64_t step = step_vec.at(i);
    CHECK_GE(step, 0);
    const int64_t slice_start_in_full_large =
        RegulateSliceStart(start_vec.at(i), i == split_axis ? length : dim_size);
    const int64_t slice_stop_in_full_large =
        RegulateSliceStop(stop_vec.at(i), i == split_axis ? length : dim_size);
    CHECK_LT(slice_start_in_full_large + step * (small_size - 1), slice_stop_in_full_large);
    {
      int64_t slice_start_in_splitted_large = slice_start_in_full_large;
      int64_t slice_stop_in_splitted_large = slice_stop_in_full_large;
      int64_t slice_size = small_size;
      if (i == split_axis) {
        if (slice_start_in_splitted_large < lower) {
          slice_start_in_splitted_large =
              lower + (step - (lower - slice_start_in_splitted_large) % step) % step;
        }
        // assume large tensor has split sbp attribute
        slice_start_in_splitted_large =
            std::min(std::max(slice_start_in_splitted_large, lower), upper);
        slice_stop_in_splitted_large =
            std::min(std::max(slice_stop_in_splitted_large, lower), upper);
        slice_start_in_splitted_large -= lower;
        slice_stop_in_splitted_large -= lower;
        slice_size =
            get_size_in_slice(slice_start_in_splitted_large, slice_stop_in_splitted_large, step);
      }
      large_slice_param.dims[i] = dim_size;
      large_slice_param.start[i] = slice_start_in_splitted_large;
      large_slice_param.step[i] = step;
      large_slice_param.size[i] = slice_size;
    }
    {
      // assume small tensor has broadcast/partialsum sbp attribute
      const int64_t dim_size = small->shape().At(i);
      int64_t slice_start_in_full_small = 0;
      int64_t slice_stop_in_full_small = dim_size;
      if (i == split_axis) {
        slice_start_in_full_small = get_size_in_slice(slice_start_in_full_large, lower, step);
        slice_stop_in_full_small = get_size_in_slice(slice_start_in_full_large, upper, step);
        slice_start_in_full_small =
            std::min(std::max<int64_t>(slice_start_in_full_small, 0), dim_size);
        slice_stop_in_full_small =
            std::min(std::max<int64_t>(slice_stop_in_full_small, 0), dim_size);
      }
      int64_t slice_size = slice_stop_in_full_small - slice_start_in_full_small;
      small_slice_param.dims[i] = dim_size;
      small_slice_param.start[i] = slice_start_in_full_small;
      small_slice_param.step[i] = 1;
      small_slice_param.size[i] = slice_size;
    }
  }
  CHECK_EQ(large_slice_param.elem_cnt(), small_slice_param.elem_cnt());
  return {large_slice_param, small_slice_param};
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
                user_op::Tensor* dst, const SliceState* slice_state,
                const bool from_large_to_small) {
  const user_op::Tensor* large = from_large_to_small ? src : dst;
  const user_op::Tensor* small = from_large_to_small ? dst : src;
  CHECK_NOTNULL(slice_state);
  if (slice_state->split_axis() != -1) {
    CHECK_EQ(large->shape().At(slice_state->split_axis()),
             slice_state->upper() - slice_state->lower());
  }

  const auto params_pair =
      ConstructSliceParams(ctx, large, small, slice_state->split_axis(), slice_state->lower(),
                           slice_state->upper(), slice_state->length());

  const auto large_slice_param = params_pair.first;
  const auto small_slice_param = params_pair.second;

  const int64_t elem_cnt = large_slice_param.elem_cnt();
  SliceIndexHelper<NDIM> entire_splitted_large_idx_cvtr(large_slice_param.dims);
  SliceIndexHelper<NDIM> sliced_splitted_large_idx_cvtr(large_slice_param.size);
  SliceIndexHelper<NDIM> entire_full_small_idx_cvtr(small_slice_param.dims);
  SliceIndexHelper<NDIM> sliced_full_small_idx_cvtr(small_slice_param.size);
  const auto* src_ptr = src->dptr<T>();
  auto* dst_ptr = dst->mut_dptr<T>();
  FOR_RANGE(int, i, 0, elem_cnt) {
    const int64_t large_offset = SliceOffsetToEntireOffset<NDIM>(
        i, large_slice_param, entire_splitted_large_idx_cvtr, sliced_splitted_large_idx_cvtr);
    const int64_t small_offset = SliceOffsetToEntireOffset<NDIM>(
        i, small_slice_param, entire_full_small_idx_cvtr, sliced_full_small_idx_cvtr);
    const int64_t src_offset = from_large_to_small ? large_offset : small_offset;
    const int64_t dst_offset = from_large_to_small ? small_offset : large_offset;
    AutoMemcpy(ctx->device_ctx(), dst_ptr + dst_offset, src_ptr + src_offset,
               GetSizeOfDataType(src->data_type()), src->mem_case(), dst->mem_case());
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
    BalancedSplitter bs(split_dim_size, ctx->parallel_ctx().parallel_num());
    return std::make_shared<SliceState>(
        split_axis, bs.At(ctx->parallel_ctx().parallel_id()).begin(),
        bs.At(ctx->parallel_ctx().parallel_id()).end(), split_dim_size);
  } else {
    return std::make_shared<SliceState>(-1, 0, 0, 0);
  }
}

template<typename T>
class Slice2Kernel final : public user_op::OpKernel {
 public:
  Slice2Kernel() = default;
  ~Slice2Kernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return CreateSliceState(ctx, "x");
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    auto* slice_state = dynamic_cast<SliceState*>(state);
    if (y_tensor->mem_case().has_host_mem()) {
      memset(y_tensor->mut_dptr(), 0,
             y_tensor->shape().elem_cnt() * GetSizeOfDataType(y_tensor->data_type()));
    } else if (y_tensor->mem_case().has_device_cuda_mem()) {
      cudaMemset(y_tensor->mut_dptr(), 0,
                 y_tensor->shape().elem_cnt() * GetSizeOfDataType(y_tensor->data_type()));
    }
    SwitchWriteSlice(SwitchCase(y_tensor->shape().NumAxes(), y_tensor->data_type()), ctx, x_tensor,
                     y_tensor, slice_state, true);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

template<typename T>
class SliceAssignKernel final : public user_op::OpKernel {
 public:
  SliceAssignKernel() = default;
  ~SliceAssignKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return CreateSliceState(ctx, "ref");
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* value_tensor = ctx->Tensor4ArgNameAndIndex("value", 0);
    user_op::Tensor* ref_tensor = ctx->Tensor4ArgNameAndIndex("ref", 0);
    auto* slice_state = dynamic_cast<SliceState*>(state);
    SwitchWriteSlice(SwitchCase(value_tensor->shape().NumAxes(), value_tensor->data_type()), ctx,
                     value_tensor, ref_tensor, slice_state, false);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_SLICE_KERNELS(device, dtype)                                              \
  REGISTER_USER_KERNEL("slice").SetCreateFn<SliceKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device)                                                  \
      & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));                      \
  REGISTER_USER_KERNEL("slice_grad")                                                       \
      .SetCreateFn<SliceGradKernel<device, dtype>>()                                       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                 \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

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

#define REGISTER_SLICE_ASSIGN_KERNELS(dtype)                                         \
  REGISTER_USER_KERNEL("slice_assign")                                               \
      .SetCreateFn<SliceAssignKernel<dtype>>()                                       \
      .SetIsMatchedHob(user_op::HobDataType("ref", 0) == GetDataType<dtype>::value); \
  REGISTER_USER_KERNEL("slice2").SetCreateFn<Slice2Kernel<dtype>>().SetIsMatchedHob( \
      user_op::HobDataType("x", 0) == GetDataType<dtype>::value);

REGISTER_SLICE_ASSIGN_KERNELS(float)
REGISTER_SLICE_ASSIGN_KERNELS(double)
REGISTER_SLICE_ASSIGN_KERNELS(int32_t)
REGISTER_SLICE_ASSIGN_KERNELS(int64_t)
REGISTER_SLICE_ASSIGN_KERNELS(int8_t)
REGISTER_SLICE_ASSIGN_KERNELS(uint8_t)
#ifdef WITH_CUDA
REGISTER_SLICE_ASSIGN_KERNELS(float16)
#endif

}  // namespace oneflow
