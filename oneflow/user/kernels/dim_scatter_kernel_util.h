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
#ifndef ONEFLOW_USER_KERNELS_DIM_SCATTER_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_DIM_SCATTER_KERNEL_UTIL_H_
#include "oneflow/user/kernels/dim_gather_scatter_util.h"
#include "oneflow/core/common/error.pb.h"

// Steps for adding a binary operation on scatter are as follows:
// 1. implment binop in DeviceBinOp, for example "Mul":
//    OF_DEVICE_FUNC static void Mul(const T* x, T* y) { *y *= *x; }
//
// 2. Implement kernels in dim_scatter_kernels.cpp:
//    IMPLEMENT_DIMSCATTER_KERNEL_CLASS(Mul);
//
// 3. Register kernels
//    REGISTER_SCATTER_OUTPLACE_KERNEL("dim_scatter_mul_like", Mul);
//
// 4. Declare Functor in dim_scatter_kernel_util.h:
//    DECLARE_DIMSCATTER_FUNCTOR(Mul);
//
// 5. Implement functors in dim_scatter_kernel_util.cu and cpp file:
//    in .cu file:
//      IMPLEMENT_DIMSCATTER_GPUFUNCTOR(Mul);
//      INSTANTIATE_DIM_SCATTER_GPUFUNCTORS(Mul);
//    in .cpp file:
//      IMPLEMENT_DIMSCATTER_CPUFUNCTOR(Mul);
//      INSTANTIATE_DIM_SCATTER_CPUFUNCTORS(Mul);
//

namespace oneflow {

namespace user_op {

// DECLARE_DIMSCATTER_FUNCTOR(Update);

template<typename IN_T, typename IDX_T, template<typename T> class Opt>
OF_DEVICE_FUNC void DoDimScatter(const DimOpIndexNdHelper<IDX_T>& src_nd_helper,
                                 const DimOpIndexNdHelper<IDX_T>& idx_nd_helper,
                                 const DimOpIndexNdHelper<IDX_T>& output_nd_helper, const int ndim,
                                 const int64_t elem_cnt, const int32_t dim, int64_t upper_bound,
                                 const IDX_T* index, const IN_T* src, IN_T* output) {
  XPU_1D_KERNEL_LOOP(idx_offset, elem_cnt) {
    IDX_T coordinate[kDimGatherMaxDimCount] = {0};
    idx_nd_helper.OffsetToNdIndex(idx_offset, coordinate, ndim);  // idx_offset -> ijk
    IDX_T idx_elem = index[idx_offset];
    if (idx_elem >= upper_bound) {
#if __CUDA_ARCH__
      __trap();
#else
      std::cout << "The index element " << idx_elem << " is out of bounds for dimension " << dim
                << " with size " << upper_bound << std::endl;
      throw Error::CheckFailedError();
#endif
    }
    IDX_T src_offset = src_nd_helper.NdIndexToOffset(coordinate, ndim);
    coordinate[dim] = idx_elem;
    IDX_T output_offset = output_nd_helper.NdIndexToOffset(coordinate, ndim);
    Opt<IN_T>::apply(src + src_offset, output + output_offset);
  }
}

template<typename IN_T, typename IDX_T>
OF_DEVICE_FUNC void DoDimScatterBinOp(const DimOpIndexNdHelper<IDX_T>& src_nd_helper,
                                      const DimOpIndexNdHelper<IDX_T>& idx_nd_helper,
                                      const DimOpIndexNdHelper<IDX_T>& output_nd_helper,
                                      const int ndim, const int64_t elem_cnt, const int32_t dim,
                                      int64_t upper_bound, const IDX_T* index, const IN_T* src,
                                      IN_T* output, BinaryOpFn<IN_T> bin_op) {
  XPU_1D_KERNEL_LOOP(idx_offset, elem_cnt) {
    IDX_T coordinate[kDimGatherMaxDimCount] = {0};
    idx_nd_helper.OffsetToNdIndex(idx_offset, coordinate, ndim);  // idx_offset -> ijk
    IDX_T idx_elem = index[idx_offset];
    if (idx_elem >= upper_bound) {
#if __CUDA_ARCH__
      __trap();
#else
      std::cout << "The index element " << idx_elem << " is out of bounds for dimension " << dim
                << " with size " << upper_bound << std::endl;
      throw Error::CheckFailedError();
#endif
    }
    IDX_T src_offset = src_nd_helper.NdIndexToOffset(coordinate, ndim);
    coordinate[dim] = idx_elem;
    IDX_T output_offset = output_nd_helper.NdIndexToOffset(coordinate, ndim);
    bin_op(src + src_offset, output + output_offset);
  }
}

#define INSTANTIATE_DIM_SCATTER_FUNCTOR_TEST(devicetype, dtype, itype, opt) \
  template struct DimScatterFunctor<devicetype, dtype, itype, opt<dtype>>;

#define INSTANTIATE_DIM_SCATTER_GPUFUNCTORS_TEST(opt)                           \
  INSTANTIATE_DIM_SCATTER_FUNCTOR_TEST(DeviceType::kGPU, int32_t, int32_t, opt) \
  INSTANTIATE_DIM_SCATTER_FUNCTOR_TEST(DeviceType::kGPU, float, int32_t, opt)   \
  INSTANTIATE_DIM_SCATTER_FUNCTOR_TEST(DeviceType::kGPU, double, int32_t, opt)  \
  INSTANTIATE_DIM_SCATTER_FUNCTOR_TEST(DeviceType::kGPU, float16, int32_t, opt) \
                                                                                \
  INSTANTIATE_DIM_SCATTER_FUNCTOR_TEST(DeviceType::kGPU, int32_t, int64_t, opt) \
  INSTANTIATE_DIM_SCATTER_FUNCTOR_TEST(DeviceType::kGPU, float, int64_t, opt)   \
  INSTANTIATE_DIM_SCATTER_FUNCTOR_TEST(DeviceType::kGPU, double, int64_t, opt)  \
  INSTANTIATE_DIM_SCATTER_FUNCTOR_TEST(DeviceType::kGPU, float16, int64_t, opt)

#define INSTANTIATE_DIM_SCATTER_CPUFUNCTORS_TEST(opt)                           \
  INSTANTIATE_DIM_SCATTER_FUNCTOR_TEST(DeviceType::kCPU, int32_t, int32_t, opt) \
  INSTANTIATE_DIM_SCATTER_FUNCTOR_TEST(DeviceType::kCPU, float, int32_t, opt)   \
  INSTANTIATE_DIM_SCATTER_FUNCTOR_TEST(DeviceType::kCPU, double, int32_t, opt)  \
                                                                                \
  INSTANTIATE_DIM_SCATTER_FUNCTOR_TEST(DeviceType::kCPU, int32_t, int64_t, opt) \
  INSTANTIATE_DIM_SCATTER_FUNCTOR_TEST(DeviceType::kCPU, float, int64_t, opt)   \
  INSTANTIATE_DIM_SCATTER_FUNCTOR_TEST(DeviceType::kCPU, double, int64_t, opt)

//------------upper:new, below: old------------
#define INSTANTIATE_DIM_SCATTER_FUNCTOR(devicetype, dtype, itype, binop) \
  template struct DimScatter##binop##Functor<devicetype, dtype, itype>;

#define INSTANTIATE_DIM_SCATTER_GPUFUNCTORS(binop)                           \
  INSTANTIATE_DIM_SCATTER_FUNCTOR(DeviceType::kGPU, int32_t, int32_t, binop) \
  INSTANTIATE_DIM_SCATTER_FUNCTOR(DeviceType::kGPU, float, int32_t, binop)   \
  INSTANTIATE_DIM_SCATTER_FUNCTOR(DeviceType::kGPU, double, int32_t, binop)  \
  INSTANTIATE_DIM_SCATTER_FUNCTOR(DeviceType::kGPU, float16, int32_t, binop) \
                                                                             \
  INSTANTIATE_DIM_SCATTER_FUNCTOR(DeviceType::kGPU, int32_t, int64_t, binop) \
  INSTANTIATE_DIM_SCATTER_FUNCTOR(DeviceType::kGPU, float, int64_t, binop)   \
  INSTANTIATE_DIM_SCATTER_FUNCTOR(DeviceType::kGPU, double, int64_t, binop)  \
  INSTANTIATE_DIM_SCATTER_FUNCTOR(DeviceType::kGPU, float16, int64_t, binop)

#define INSTANTIATE_DIM_SCATTER_CPUFUNCTORS(binop)                           \
  INSTANTIATE_DIM_SCATTER_FUNCTOR(DeviceType::kCPU, int32_t, int32_t, binop) \
  INSTANTIATE_DIM_SCATTER_FUNCTOR(DeviceType::kCPU, float, int32_t, binop)   \
  INSTANTIATE_DIM_SCATTER_FUNCTOR(DeviceType::kCPU, double, int32_t, binop)  \
                                                                             \
  INSTANTIATE_DIM_SCATTER_FUNCTOR(DeviceType::kCPU, int32_t, int64_t, binop) \
  INSTANTIATE_DIM_SCATTER_FUNCTOR(DeviceType::kCPU, float, int64_t, binop)   \
  INSTANTIATE_DIM_SCATTER_FUNCTOR(DeviceType::kCPU, double, int64_t, binop)

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_DIM_GATHER_KERNEL_UTIL_H_
