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
#include <type_traits>
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/user/kernels/to_contiguous_kernel.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {

namespace {

constexpr int kBlockSize = cuda::elementwise::kBlockSize;

int GetMinThreadNum(int64_t elem_cnt) { return std::min<int64_t>(elem_cnt, kBlockSize); }

int GetNumBlocks(int64_t elem_cnt) {
  int num_blocks = 0;
  OF_CUDA_CHECK(cuda::elementwise::GetNumBlocks(elem_cnt, &num_blocks));
  return num_blocks;
}

template<size_t ndim>
struct StrideParam {
  int stride[SHAPE_MAX_AXIS_SIZE];
  int coordinates[SHAPE_MAX_AXIS_SIZE];

  // NOLINTNEXTLINE
  StrideParam(const int64_t* stride_vec) {
    for (size_t i = 0; i < ndim; ++i) {
      stride[i] = stride_vec[i];
      coordinates[i] = 0;
    }
  }
};

template<typename IndexType, size_t ndim>
__device__ __forceinline__ IndexType compute_index(IndexType out_offset, StrideParam<ndim>& out_params,
                                 const StrideParam<ndim>& in_params) {
  IndexType in_offset = 0;
  IndexType remaining = out_offset;

#pragma unroll
  // compute coords(output offset to coords)
  for (int i = 0; i < ndim; ++i) {
    const IndexType idx = static_cast<IndexType>(remaining / out_params.stride[i]);
    out_params.coordinates[i] = idx;
    remaining = remaining - idx * out_params.stride[i];
  }
  // compute input offset
  for (int dim = 0; dim < ndim; ++dim) {
    in_offset = in_offset + out_params.coordinates[dim] * in_params.stride[dim];
  }
  return in_offset;
}


template<typename T, typename IndexType, size_t ndim>
__global__ void ToContiguousForwardGpu(IndexType count, StrideParam<ndim> in_stride,
                              StrideParam<ndim> out_stride, const T* in_dptr, T* out_dptr) {
  for (IndexType out_idx = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; out_idx < count; out_idx += step)
  {
    IndexType in_idx = compute_index<IndexType, ndim>(out_idx, out_stride, in_stride);
    out_dptr[out_idx] = in_dptr[in_idx];
  }
}

template<typename T, typename IndexType, size_t ndim, int32_t pack_size>
void LaunchToContiguousKernel(ep::Stream* stream, IndexType count, IndexType block_size, const std::vector<int64_t>& in_stride,
                           const StrideVector& out_stride, const char* in_dptr, char* out_dptr){

  StrideParam<ndim> param_in_stride(in_stride.data()), param_out_stride(out_stride.data());

  const size_t num_blocks = GetNumBlocks(count);
  const size_t num_threads = GetMinThreadNum(count);

  if (pack_size == 16 && block_size % 16 == 0) {
    ToContiguousForwardGpu<ulonglong2, IndexType, ndim><<<num_blocks, num_threads, 0,
                           stream->As<ep::CudaStream>()->cuda_stream()>>>(
      count, param_in_stride, param_out_stride, reinterpret_cast<const ulonglong2*>(in_dptr), reinterpret_cast<ulonglong2*>(out_dptr));
  } else if (pack_size == 8 && block_size % 8 == 0) {
    ToContiguousForwardGpu<uint64_t, IndexType, ndim><<<num_blocks, num_threads, 0,
                           stream->As<ep::CudaStream>()->cuda_stream()>>>(
      count, param_in_stride, param_out_stride, reinterpret_cast<const uint64_t*>(in_dptr), reinterpret_cast<uint64_t*>(out_dptr));
  } else if(pack_size == 4 && block_size % 4 == 0 ){
    ToContiguousForwardGpu<uint32_t, IndexType, ndim><<<num_blocks, num_threads, 0,
                           stream->As<ep::CudaStream>()->cuda_stream()>>>(
      count, param_in_stride, param_out_stride, reinterpret_cast<const uint32_t*>(in_dptr), reinterpret_cast<uint32_t*>(out_dptr));
  } else if(pack_size == 2 && block_size % 2 == 0 ){
    ToContiguousForwardGpu<uint16_t, IndexType, ndim><<<num_blocks, num_threads, 0,
                           stream->As<ep::CudaStream>()->cuda_stream()>>>(
      count, param_in_stride, param_out_stride, reinterpret_cast<const uint16_t*>(in_dptr), reinterpret_cast<uint16_t*>(out_dptr));
  } else {
    ToContiguousForwardGpu<T, IndexType, ndim><<<num_blocks, num_threads, 0,
                           stream->As<ep::CudaStream>()->cuda_stream()>>>(
      count, param_in_stride, param_out_stride, reinterpret_cast<const T*>(in_dptr), reinterpret_cast<T*>(out_dptr));
  }
}


} // namespace


template<typename T>
struct ToContiguousUtil<DeviceType::kCUDA, T> : ToContiguousUtilBase {
  using ToContiguousUtilBase::ToContiguousUtilBase;
  static constexpr size_t dsize = sizeof(T);
  void operator()() {
    const size_t ndims = contiguous_dim + 1;
    if (ndims == 0) {
      // 0-dim tensor
      OF_CUDA_CHECK(cudaMemcpyAsync(out_dptr, in_dptr, block_size * dsize, cudaMemcpyDeviceToDevice,
                                    stream->As<ep::CudaStream>()->cuda_stream()));
    } else {
      if(ndims > 8){
        UNIMPLEMENTED() << "It is not supported that the dimension of the input tensor is greater than 8 for now!";
      }
      bool is_same = true;
      for (int64_t i = contiguous_dim; i != -1; --i) {
        if (out_stride[i] != in_stride[i]) {
          is_same = false;
          break;
        }
      }
      if (is_same) {
        // if input tensor's strides equals to output's, than just copy one memory-contiguous tensor
        OF_CUDA_CHECK(cudaMemcpyAsync(out_dptr + out_offset * dsize, in_dptr + in_offset * dsize, element_count * dsize, cudaMemcpyDeviceToDevice,
                                    stream->As<ep::CudaStream>()->cuda_stream()));
      } else{
        // printf("\n >>> block_size:%ld; in stride:(%ld, %ld, %ld, %ld); out stride:(%ld, %ld, %ld, %ld)", block_size, in_stride[0], in_stride[1], in_stride[2], in_stride[3], out_stride[0], out_stride[1], out_stride[2], out_stride[3]);
        constexpr int pack_size = cuda::elementwise::PackSize<T>();
        if (element_count < GetMaxVal<int32_t>()) {
          if (ndims==1){
            LaunchToContiguousKernel<T, int32_t, 1, pack_size>(stream, element_count, block_size, in_stride, out_stride, in_dptr, out_dptr);
          } else if (ndims==2){
            LaunchToContiguousKernel<T, int32_t, 2, pack_size>(stream, element_count, block_size, in_stride, out_stride, in_dptr, out_dptr);
          } else if (ndims==3){
            LaunchToContiguousKernel<T, int32_t, 3, pack_size>(stream, element_count, block_size, in_stride, out_stride, in_dptr, out_dptr);
          } else if (ndims==4){
            LaunchToContiguousKernel<T, int32_t, 4, pack_size>(stream, element_count, block_size,in_stride, out_stride, in_dptr, out_dptr);
          } else if (ndims==5){
            LaunchToContiguousKernel<T, int32_t, 5, pack_size>(stream, element_count, block_size, in_stride, out_stride, in_dptr, out_dptr);
          } else if (ndims==6){
            LaunchToContiguousKernel<T, int32_t, 6, pack_size>(stream, element_count, block_size, in_stride, out_stride, in_dptr, out_dptr);
          } else if (ndims==7){
            LaunchToContiguousKernel<T, int32_t, 7, pack_size>(stream, element_count, block_size, in_stride, out_stride, in_dptr, out_dptr);
          } else {
            LaunchToContiguousKernel<T, int32_t, 8, pack_size>(stream, element_count, block_size, in_stride, out_stride, in_dptr, out_dptr);
          }    
        } else {
          if (ndims==1){
            LaunchToContiguousKernel<T, int64_t, 1, pack_size>(stream, element_count, block_size, in_stride, out_stride, in_dptr, out_dptr);
          } else if (ndims==2){
            LaunchToContiguousKernel<T, int64_t, 2, pack_size>(stream, element_count, block_size, in_stride, out_stride, in_dptr, out_dptr);
          } else if (ndims==3){
            LaunchToContiguousKernel<T, int64_t, 3, pack_size>(stream, element_count, block_size, in_stride, out_stride, in_dptr, out_dptr);
          } else if (ndims==4){
            LaunchToContiguousKernel<T, int64_t, 4, pack_size>(stream, element_count, block_size, in_stride, out_stride, in_dptr, out_dptr);
          } else if (ndims==5){
            LaunchToContiguousKernel<T, int64_t, 5, pack_size>(stream, element_count, block_size, in_stride, out_stride, in_dptr, out_dptr);
          } else if (ndims==6){
            LaunchToContiguousKernel<T, int64_t, 6, pack_size>(stream, element_count, block_size, in_stride, out_stride, in_dptr, out_dptr);
          } else if (ndims==7){
            LaunchToContiguousKernel<T, int64_t, 7, pack_size>(stream, element_count, block_size, in_stride, out_stride, in_dptr, out_dptr);
          } else {
            LaunchToContiguousKernel<T, int64_t, 8, pack_size>(stream, element_count, block_size, in_stride, out_stride, in_dptr, out_dptr);
          }
        }
      }
    }
  }
};


template<DeviceType device_type, typename T>
class ToContiguousKernel final : public user_op::OpKernel {
 public:
  ToContiguousKernel() = default;
  ~ToContiguousKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const ShapeView& in_shape = in->shape();
    CHECK_EQ(out->shape(), in_shape);
    const DataType in_data_type = in->data_type();
    CHECK_EQ(out->data_type(), in_data_type);

    const auto& in_stride = ctx->Attr<std::vector<int64_t>>("stride");

    const char* in_dptr = static_cast<const char*>(in->raw_dptr());
    char* out_dptr = static_cast<char*>(out->mut_raw_dptr());

    ToContiguousUtil<device_type, T>(ctx->stream(), in_shape, in_stride, in_dptr, out_dptr)();
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};


#define INSTANTIATE_TO_CONTIGUOUS_UTILS_FOR_CUDA(T) \
  template struct ToContiguousUtil<DeviceType::kCUDA, T>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_TO_CONTIGUOUS_UTILS_FOR_CUDA,
                     TO_CONTIGUOUS_TYPES TO_CONTIGUOUS_CUDA_SPECIAL_TYPE)

}  // namespace oneflow
