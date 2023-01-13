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
#ifndef ONEFLOW_USER_KERNELS_UNPOOL_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_UNPOOL_KERNEL_UTIL_H_
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

#define UNPOOL_DATA_TYPE_SEQ                      \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64) \
  OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat)   \
  OF_PP_MAKE_TUPLE_SEQ(double, DataType::kDouble)

#define UNPOOL_IDX_DATA_TYPE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

#define UNPOOL_DATA_TYPE_CPU_SEQ                                         \
  UNPOOL_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(float16, DataType::kFloat16) \
      OF_PP_MAKE_TUPLE_SEQ(bfloat16, DataType::kBFloat16)
#define UNPOOL_DATA_TYPE_CUDA_SEQ \
  UNPOOL_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(half, DataType::kFloat16)
// OF_PP_MAKE_TUPLE_SEQ(nv_bfloat16, DataType::kBFloat16)

typedef small_vector<int64_t, SHAPE_MAX_AXIS_SIZE> FixedDimVector;

class MaxUnpoolParams3D {
 public:
  MaxUnpoolParams3D(const int32_t dim, const ShapeView& x_shape,
                    const std::vector<int32_t>& padding, const std::vector<int32_t>& kernel_size,
                    const std::vector<int32_t>& stride);
  ~MaxUnpoolParams3D() = default;

  const std::vector<int32_t>& padding() const { return padding_; }
  const std::vector<int32_t>& pool_size_3d() const { return pool_size_3d_; }
  const std::vector<int32_t>& stride_3d() const { return stride_3d_; }
  const int32_t& num_batch() const { return batch_num_; }
  const int32_t& num_channel() const { return channel_num_; }

  void Reset(const ShapeView& x_shape);

  Shape GetYShape() const;
  Shape GetXShape5D() const;
  Shape GetYShape5D() const;
  int64_t GetYStride() const;

 private:
  int32_t dim_;
  FixedDimVector x_3d_;
  FixedDimVector y_3d_;
  std::vector<int32_t> padding_;
  std::vector<int32_t> pool_size_3d_;
  std::vector<int32_t> stride_3d_;
  int32_t batch_num_;
  int32_t channel_num_;
};

template<DeviceType device_type, typename T, typename IDX>
struct UnpoolKernelUtil {
  static void MaxUnpoolNdForward(ep::Stream* stream,
                                 const NdIndexOffsetHelper<IDX, 2>& index_helper,
                                 const IDX elem_num, const T* src, T* dest,
                                 const int64_t* indice_ptr, const int64_t y_hwd_size,
                                 const int64_t y_elem_num);
  static void MaxUnpoolNdBackward(ep::Stream* stream,
                                  const NdIndexOffsetHelper<IDX, 2>& index_helper,
                                  const IDX elem_num, const T* src, T* dest,
                                  const int64_t* indice_ptr, const int64_t dy_hwd_size,
                                  const int64_t dy_elem_num);
};

#define INSTANTIATE_UNPOOL_KERNEL_UTIL(device_type_v, dtype_pair, index_dtype_pair) \
  template struct UnpoolKernelUtil<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),     \
                                   OF_PP_PAIR_FIRST(index_dtype_pair)>;

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_UNPOOL_KERNEL_UTIL_H_
