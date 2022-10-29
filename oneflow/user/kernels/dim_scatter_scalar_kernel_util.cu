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
#ifdef WITH_CUDA
#include "oneflow/user/kernels/dim_scatter_scalar_kernel_util.h"

namespace oneflow {

namespace user_op {

template<typename IN_T, typename IDX_T, template<typename T> class Opt>
__global__ void DoCUDADimScatterScalar(const DimOpIndexNdHelper<IDX_T> idx_nd_helper,
                                       const DimOpIndexNdHelper<IDX_T> output_nd_helper,
                                       const int ndim, const int64_t elem_cnt, const int32_t dim,
                                       const int64_t upper_bound, const IDX_T* index,
                                       const IN_T src_scalar, IN_T* output) {
  DoScatterScalarFunctor<IN_T, IDX_T, Opt>(idx_nd_helper, output_nd_helper, ndim, elem_cnt, dim,
                                           upper_bound, index, src_scalar, output);
}

template<typename IN_T, typename IDX_T, template<typename T> class Opt>
struct DimScatterScalarFunctor<DeviceType::kCUDA, IN_T, IDX_T, Opt> final {
  void operator()(ep::Stream* stream, const DimOpIndexNdHelper<IDX_T>& idx_nd_helper,
                  const DimOpIndexNdHelper<IDX_T>& output_nd_helper, const int ndim,
                  const int64_t elem_cnt, const int32_t dim, int64_t upper_bound,
                  const IDX_T* index, const IN_T src, IN_T* output) {
    RUN_CUDA_KERNEL((DoCUDADimScatterScalar<IN_T, IDX_T, Opt>), stream,
                    BlocksNum4ThreadsNum(elem_cnt), idx_nd_helper, output_nd_helper, ndim, elem_cnt,
                    dim, upper_bound, index, src, output);
  }
};

template<typename IDX_T, template<typename T> class Opt>
struct DimScatterScalarFunctor<DeviceType::kCUDA, float16, IDX_T, Opt> final {
  void operator()(ep::Stream* stream, const DimOpIndexNdHelper<IDX_T>& idx_nd_helper,
                  const DimOpIndexNdHelper<IDX_T>& output_nd_helper, const int ndim,
                  const int64_t elem_cnt, const int32_t dim, int64_t upper_bound,
                  const IDX_T* index, const float16 src, float16* output) {
    RUN_CUDA_KERNEL((DoCUDADimScatterScalar<half, IDX_T, Opt>), stream,
                    BlocksNum4ThreadsNum(elem_cnt), idx_nd_helper, output_nd_helper, ndim, elem_cnt,
                    dim, upper_bound, index, src, reinterpret_cast<half*>(output));
  }
};

INSTANTIATE_DIM_SCATTER_SCARLAR_CUDA_FUNCTORS(DeviceType::kCUDA, UpdateScalarFunctor);
INSTANTIATE_DIM_SCATTER_SCARLAR_CUDA_FUNCTORS(DeviceType::kCUDA, AddScalarFunctor);

}  // namespace user_op
}  // namespace oneflow
#endif
