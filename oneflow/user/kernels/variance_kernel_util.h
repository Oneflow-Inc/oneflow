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
#ifndef ONEFLOW_USER_KERNELS_VARIANCE_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_VARIANCE_KERNEL_UTIL_H_

#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/device/cuda_util.h"
namespace oneflow {
namespace user_op {

OF_DEVICE_FUNC size_t LinearIndex2Offset(const size_t linear_index,
                                         const int32_t* dim_size_in_axis_ptr,
                                         const int32_t* stride_vec_ptr, const int32_t size) {
  // low dim at begin
  size_t offset = 0;
  size_t num_dim = 0;
  for (int j = 0; j < size; j++) {
    num_dim = (j == 0 ? linear_index : (num_dim / dim_size_in_axis_ptr[j - 1]));
    offset += num_dim % dim_size_in_axis_ptr[j] * stride_vec_ptr[j];
  }
  return offset;
}

namespace {
constexpr size_t MaxDims = 8;
}  // namespace

struct VarParam {
  VarParam() : unbiased(true), parallel_num(1), elem_cnt(1), axis_size(1), caxis_size(1) {}
  bool unbiased;
  size_t parallel_num;
  size_t elem_cnt;
  int32_t axis_size;
  int32_t caxis_size;
  int32_t stride_in_axis[MaxDims];
  int32_t dim_size_in_axis[MaxDims];
  int32_t stride_in_caxis[MaxDims];
  int32_t dim_size_in_caxis[MaxDims];
};

class VarParamHelper final {
 public:
  VarParamHelper() = delete;
  explicit VarParamHelper(const ShapeView& input_shape, const std::vector<int32_t> axis,
                          const bool unbiased)
      : axis_(axis), input_shape_(input_shape) {
    param.unbiased = unbiased;
    ComputeStrideVec(axis_, param.stride_in_axis);
    caxis_ = GetCAxis();
    ComputeStrideVec(caxis_, param.stride_in_caxis);
    GetDimSizeInAxis(axis_, param.dim_size_in_axis);
    GetDimSizeInAxis(caxis_, param.dim_size_in_caxis);
    ComputeElemCntAndParallelNum();
    param.axis_size = axis_.size();
    param.caxis_size = caxis_.size();
  }

  VarParam param;

 private:
  void ComputeElemCntAndParallelNum() {
    for (int i = 0; i < axis_.size(); i++) { param.elem_cnt *= input_shape_.At(axis_[i]); }
    for (int i = 0; i < caxis_.size(); i++) { param.parallel_num *= input_shape_.At(caxis_[i]); }
  }

  void ComputeStrideVec(const std::vector<int32_t> axis, int32_t* stride_vec) {
    // low dim at begin
    const int axis_size = axis.size();
    for (int i = 0; i < axis_size; i++) {
      int stride = 1;
      if (axis.at(i) + 1 == input_shape_.NumAxes()) {
        stride_vec[axis_size - 1 - i] = 1;
      } else {
        for (int j = axis.at(i) + 1; j < input_shape_.NumAxes(); j++) {
          stride *= input_shape_.At(j);
        }
        stride_vec[axis_size - 1 - i] = stride;
      }
    }
  }

  std::vector<int32_t> GetCAxis() {
    std::vector<int32_t> caxis;
    caxis.resize(input_shape_.NumAxes());
    std::iota(caxis.begin(), caxis.end(), 0);
    for (int i = 0; i < axis_.size(); i++) { caxis.erase(caxis.begin() + axis_.at(i) - i); }
    return caxis;
  }

  void GetDimSizeInAxis(const std::vector<int32_t> axis, int32_t* dim_size_in_axis) {
    // low dim at begin
    const int axis_size = axis.size();
    for (int i = 0; i < axis_size; i++) {
      dim_size_in_axis[axis_size - 1 - i] = input_shape_.At(axis.at(i));
    }
  }

  const std::vector<int32_t> axis_;
  const ShapeView input_shape_;
  std::vector<int32_t> caxis_;
};

template<typename T, typename ComputeType>
OF_DEVICE_FUNC void ComputeVarUsingWelford(const T* in_ptr, T* out_ptr, const VarParam& var_param) {
  size_t count = 0;
  // torch use double even for float data, so here float will result in accuracy error.
  ComputeType mean = 0.0;
  ComputeType old_mean = 0.0;
  ComputeType m2 = 0.0;
  for (size_t i = 0; i < var_param.elem_cnt; i++) {
    const size_t offset = LinearIndex2Offset(i, var_param.dim_size_in_axis,
                                             var_param.stride_in_axis, var_param.axis_size);
    count++;
    old_mean = mean;
    mean += (static_cast<ComputeType>(in_ptr[offset]) - mean) / count;
    m2 += (static_cast<ComputeType>(in_ptr[offset]) - mean)
          * (static_cast<ComputeType>(in_ptr[offset]) - old_mean);
  }
  *out_ptr = m2 / (var_param.unbiased ? count - 1 : count);
}

namespace {

OF_DEVICE_FUNC bool IsNanOut(const VarParam var_param) {
  return (var_param.elem_cnt == 0) || (var_param.elem_cnt == 1 && var_param.unbiased == true);
}

#ifdef WITH_CUDA
void SetGridDimAndBlockDim(const size_t total_elem_cnt, int* grid_dim, int* block_dim) {
  // when total_elem_cnt > 2 * kCudaThreadsNumPerBlock, use two cuda kernel
  if (total_elem_cnt > (kCudaThreadsNumPerBlock << 1)) {
    *grid_dim =
        std::min(static_cast<int32_t>(std::ceil(std::sqrt(total_elem_cnt))), kCudaMaxBlocksNum);
    *block_dim = kCudaThreadsNumPerBlock;
  } else {
    *grid_dim = 1;
    int32_t aligned_block_dim =
        (total_elem_cnt >= kCudaThreadsNumPerBlock)
            ? kCudaThreadsNumPerBlock
            // avoid get block_dim = 0 when total_elem_cnt is 0
            : std::max<long unsigned>((total_elem_cnt + kCudaWarpSize - 1) / kCudaWarpSize, 1)
                  * kCudaWarpSize;
    *block_dim = std::min(aligned_block_dim, kCudaThreadsNumPerBlock);
  }
}
#endif  // WITH_CUDA
}  // namespace

template<DeviceType device_type, typename T, typename ComputeType>
struct VarFunctor final {
  void operator()(ep::Stream* stream, const T* in_ptr, T* out_ptr, ComputeType* tmp_buffer_ptr,
                  const VarParam var_param);
};

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_VARIANCE_KERNEL_UTIL_H_
