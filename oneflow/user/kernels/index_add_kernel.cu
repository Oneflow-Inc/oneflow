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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include "_deps/glog-build/glog/logging.h"
#include "oneflow/core/common/bfloat16.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/user_op_hob.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

bool IsContiguous(size_t num_dims, const ShapeView& t, const Stride& stride) {
  DimVector t_shape_dim;
  t.ToDimVector(&t_shape_dim);
  std::vector<int32_t> t_stride(stride.begin(), stride.end());
  for (int i = num_dims - 1; i >= 0; i--) {
    if ((i == num_dims - 1 && t_stride[i] != 1)
        || (i != num_dims - 1 && t_stride[i] != t_shape_dim[i + 1] * t_stride[i + 1])) {
      return false;
    }
  }
  return true;
}

// CUDA kernel argument that defines tensor layout
template <typename IndexType>
struct TensorInfo {
  TensorInfo();
  TensorInfo(int dim,
             IndexType sz[SHAPE_MAX_AXIS_SIZE],
             IndexType st[SHAPE_MAX_AXIS_SIZE]);

  // Set the size of the given dimension to 1, as if it were a
  // reduction dim (allows you to calculate offsets of the reduction
  // slice)
  void reduceDim(int dim);

  // Contiguous tensors of more than one dimension are collapsed down
  // to one tensor
  OF_DEVICE_FUNCTION bool isContiguous() const {
    return (dims == 1 && strides[0] == 1);
  }

  IndexType sizes[SHAPE_MAX_AXIS_SIZE];
  IndexType strides[SHAPE_MAX_AXIS_SIZE];
  int dims;
};

template <typename IndexType>
TensorInfo<IndexType>::TensorInfo() {
  dims = 0;
}

template <typename IndexType>
TensorInfo<IndexType>::TensorInfo(int dim,
                                     IndexType sz[SHAPE_MAX_AXIS_SIZE],
                                     IndexType st[SHAPE_MAX_AXIS_SIZE]) {
  dims = dim;
  CHECK_EQ(dims < SHAPE_MAX_AXIS_SIZE, true) << "CUDA Tensors cannot have more than 25 dimensions";

  for (int i = 0; i < dim; ++i) {
    sizes[i] = sz[i];
    strides[i] = st[i];
  }
}

template <typename IndexType>
void
TensorInfo<IndexType>::reduceDim(int dim) {
  CHECK_EQ(dim < dims && dim >= 0, true) << "expected dim between 0 and dims - 1";
  sizes[dim] = 1;
}

// TODO(add collapseDims function)

// Translate a linear index for the apply to a T* offset;
// specialized on `Dims` to reduce nvcc compilation time
template <typename IndexType, int Dims>
struct IndexToOffset {
  static OF_DEVICE_FUNCTION IndexType get(
    IndexType linearId,
    const TensorInfo<IndexType>& info) {

    IndexType offset = 0;

    // Uses static dims
    for (int i = Dims - 1; i > 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;
      linearId /= info.sizes[i];
    }

    return offset + linearId * info.strides[0];
  }
};

// Uses dynamic (runtime) instead of static (compiletime) dims
template <typename IndexType>
struct IndexToOffset<IndexType, -1> {
  static OF_DEVICE_FUNCTION IndexType get(
    IndexType linearId,
    const TensorInfo<IndexType>& info) {

      IndexType offset = 0;

      for (int i = info.dims - 1; i > 0; --i) {
        IndexType curDimIndex = linearId % info.sizes[i];
        IndexType curDimOffset = curDimIndex * info.strides[i];
        offset += curDimOffset;
        linearId /= info.sizes[i];
      }

      return offset + linearId * info.strides[0];
  }
};


template <typename IndexType>
TensorInfo<IndexType>
getTensorInfo(const ShapeView &t, const Stride& stride) {
  IndexType sz[SHAPE_MAX_AXIS_SIZE];
  IndexType st[SHAPE_MAX_AXIS_SIZE];

  DimVector t_shape_dim;
  t.ToDimVector(&t_shape_dim);
  std::vector<int32_t> t_stride(stride.begin(), stride.end());

  int dims = t_shape_dim.size();
  for (int i = 0; i < dims; ++i) {
    sz[i] = t_shape_dim[i];
    st[i] = t_stride[i];
  }

  return TensorInfo<IndexType>(dims, sz, st);
}

/**
   Computes ceil(a / b)
*/
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
OF_DEVICE_FUNCTION T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

bool canUse32BitIndexMath(const ShapeView& t, const Stride& stride, int32_t max_elem=std::numeric_limits<int32_t>::max()) {
  auto elements = t.elem_cnt();
  if (elements >= max_elem) {
    return false;
  }
  if (elements == 0) {
    return max_elem > 0;
  }

  size_t offset = 0;
  auto linearId = elements - 1;

  DimVector t_shape_dim;
  t.ToDimVector(&t_shape_dim);
  std::vector<int32_t> t_stride(stride.begin(), stride.end());

  // NOTE: Assumes all strides are positive, which is true for now
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  for (int i = t_shape_dim.size() - 1; i >= 0; --i) {
    auto curDimIndex = linearId % t_shape_dim[i];
    auto curDimOffset = curDimIndex * t_stride[i];
    offset += curDimOffset;
    linearId /= t_shape_dim[i];
  }

  if (offset >= max_elem) {
    return false;
  }

  return true;
}

class ReduceAdd {
public:
  template <typename T>
  constexpr __device__ void operator() (T* self_data_start, int32_t index, int32_t numel, const T * src_data) const {
    cuda::atomic::FastAdd(self_data_start, index, numel, *src_data);
  }
};
static ReduceAdd reduce_add;

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexFuncLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          typename func_t>
__global__ void indexFuncSmallIndex(TensorInfo<IndexType> dst,
                                    TensorInfo<IndexType> src,
                                    TensorInfo<IndexType> indices,
                                    const IndexType* indices_data, 
                                    const T* src_data,
                                    T* dst_data,
                                    int32_t dstAddDim,
                                    int32_t srcAddDim,
                                    IndexType innerSize,
                                    int32_t dstAddDimSize,
                                    int32_t dstNumel,
                                    const func_t& op,
                                    T alpha) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    // Lua indices begin at 1
    IndexType dstIndex =
        indices_data[IndexToOffset<IndexType, IdxDim>::get(srcIndex, indices)];
    assert(dstIndex < dstAddDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
          IndexToOffset<IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstAddDim];

      IndexType srcOffset =
          IndexToOffset<IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcAddDim];

      T val = src_data[srcOffset] * alpha;
      op(dst_data, dstOffset, dstNumel, &val);
    }

  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexFuncSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor, typename func_t>
__global__ void indexFuncLargeIndex(TensorInfo<IndexType> dst,
                                    TensorInfo<IndexType> src,
                                    TensorInfo<IndexType> indices,
                                    const IndexType* indices_data, 
                                    const T* src_data,
                                    T* dst_data,
                                    int32_t dstAddDim,
                                    int32_t srcAddDim,
                                    IndexType totalSize,
                                    IndexType innerSize,
                                    int32_t dstAddDimSize,
                                    int32_t dstNumel,
                                    const func_t& op,
                                    T alpha) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType srcIndex, elementInSlice;
    if (IndexIsMajor) {
      srcIndex = linearIndex / innerSize;
      elementInSlice = linearIndex % innerSize;
    }
    else {
      elementInSlice = linearIndex / innerSize;
      srcIndex = linearIndex % innerSize;
    }

    // Lua indices begin at 1
    IndexType dstIndex =
        indices_data[IndexToOffset<IndexType, IdxDim>::get(srcIndex, indices)];
    assert(dstIndex < dstAddDimSize);

    IndexType dstOffset =
      IndexToOffset<IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstAddDim];

    IndexType srcOffset =
      IndexToOffset<IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcAddDim];

    T val = src_data[srcOffset] * alpha;
    op(dst_data, dstOffset, dstNumel, &val);
  }
}

// Compare the stride between adjacent slices (sliceStride) with strides in the
// other dimensions (i.e., strides *inside* each slice).
//
// - Returns true if some dimension inside the slice has lower stride than
//   sliceStride.  The simplest example is a 2-D contiguous tensor with sliceDim
//   == 0 (that is, each slice is a row).
//
//   In this case, we choose the CUDA kernel that processes the data in
//   "index-major order".  For example, if thread count equals slice size, then
//   all threads process slice #0 in lockstep, and then slice #1, and so on.
//
// - Otherwise (i.e., sliceStride has the lowest value), this function returns
//   false.  The simplest example is a 2-D contiguous tensor with sliceDim == 1
//   (each slice is a column).
//
//   In this case, we choose the CUDA kernel that processes the data in
//   "elementInSlice-major order".  For example, each thread can process element
//   #0 of every slice, and then element #1 of every slice, and so on.
template <typename IndexT>
bool indexShouldBeMajor(TensorInfo<IndexT> &info,
                                    int sliceDim)
{
  // The stride between adjacent slices (e.g., between element #0 of slice #100
  // and element #0 of slice #101).
  unsigned int sliceStride = info.strides[sliceDim];

  for (size_t i = 0; i < info.dims; i++) {
    if (i != sliceDim && info.sizes[i] > 1 && info.strides[i] < sliceStride) {
      return true;
    }
  }

  return false;
}

};  // namespace

template<typename T, typename IndexT>
class IndexAddGpuKernel final : public user_op::OpKernel {
 public:
  IndexAddGpuKernel() = default;
  ~IndexAddGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* self = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* index = ctx->Tensor4ArgNameAndIndex("index", 0);
    const user_op::Tensor* source = ctx->Tensor4ArgNameAndIndex("source", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    const int32_t dim = ctx->Attr<int32_t>("dim");
    const float alpha = ctx->Attr<float>("alpha");
    const ShapeView& self_shape = self->shape_view();
    const ShapeView& source_shape = source->shape_view();
    const ShapeView& index_shape = index->shape_view();
    DimVector self_shape_dim, source_shape_dim, index_shape_dim;
    self_shape.ToDimVector(&self_shape_dim);
    source_shape.ToDimVector(&source_shape_dim);
    index_shape.ToDimVector(&index_shape_dim);
    const Stride& self_stride = self->stride();
    const Stride& index_stride = index->stride();
    const Stride& source_stride = source->stride();

    Memcpy<DeviceType::kCUDA>(
        ctx->stream(), output->mut_dptr<void>(), self->dptr<void>(),
        self->shape_view().elem_cnt() * GetSizeOfDataType(self->data_type()));

    int32_t sliceSize = 1;
    for (int i = 0; i < self_shape_dim.size(); i++){
      if (i != dim){
        sliceSize *= self_shape_dim[i];
      }
    }
    const int32_t sourceTotalSize = source_shape.elem_cnt();
    const int32_t selfAddDimSize = self_shape_dim[dim];
    const int32_t numIndex = index_shape.elem_cnt();
    const int32_t selfNumel = self_shape.elem_cnt();

    const cudaStream_t stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();

#define SMALL_INDEX(TENSOR_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM)     \
  indexFuncSmallIndex<TENSOR_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM>   \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                   \
      selfInfo, sourceInfo, indexInfo, index->dptr<TYPE>(), source->dptr<T>(), output->mut_dptr<T>(),                         \
      selfAddDim, sourceAddDim, sliceSize, selfAddDimSize,                              \
      selfNumel, reduce_add, alpha_value);                                              \

#define LARGE_INDEX(TENSOR_TYPE, TYPE,                        \
                    SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR)            \
  indexFuncLargeIndex<TENSOR_TYPE, TYPE,                      \
                      SELF_DIM, SOURCE_DIM, IDX_DIM, IDX_IS_MAJOR>          \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                       \
      selfInfo, sourceInfo, indexInfo, index->dptr<TYPE>(),  source->dptr<T>(), output->mut_dptr<T>(),                        \
      selfAddDim, sourceAddDim, sourceTotalSize,                            \
      (IDX_IS_MAJOR) ? sliceSize : numIndex,                                \
      selfAddDimSize, selfNumel, reduce_add, alpha_value);                  \

    const bool indContig = IsContiguous(numIndex, index_shape, index_stride);
    const int mpc = static_cast<uint32_t>(ctx->stream()->As<ep::CudaStream>()->device_properties().multiProcessorCount);
    const dim3 smallIndexGrid(std::min(ceil_div(sliceSize, (int32_t)128), (int32_t)(mpc * 8)));
    const dim3 smallIndexBlock(std::min(sliceSize, (int32_t)128));

    const dim3 largeIndexGrid(std::min(ceil_div(sourceTotalSize, (int32_t)128), (int32_t)(mpc * 8)));
    const dim3 largeIndexBlock(std::min(sourceTotalSize, (int32_t)128));
    const T alpha_value = static_cast<T>(alpha);

    if (canUse32BitIndexMath(self_shape, self_stride) && canUse32BitIndexMath(source_shape, source_stride) && canUse32BitIndexMath(index_shape, index_stride)) {
      TensorInfo<IndexT> selfInfo = getTensorInfo<IndexT>(self_shape, self_stride);
      const int32_t selfAddDim = dim;
      selfInfo.reduceDim(dim);
      TensorInfo<IndexT> sourceInfo = getTensorInfo<IndexT>(source_shape, source_stride);
      const int32_t sourceAddDim = dim;
      sourceInfo.reduceDim(dim);
      TensorInfo<IndexT> indexInfo = getTensorInfo<IndexT>(index_shape, index_stride);

      if(numIndex <= 16){
        if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
            SMALL_INDEX(T, IndexT, 1, 1, -2);
          } else if (selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
            SMALL_INDEX(T, IndexT, 2, 2, -2);
          } else if (selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
            SMALL_INDEX(T, IndexT, 3, 3, -2);
          } else {
            SMALL_INDEX(T, IndexT, -1, -1, -1);
          }
      }
      else {
          const bool indexIsMajor = indexShouldBeMajor<IndexT>(selfInfo, selfAddDim);

          if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
            LARGE_INDEX(T, IndexT, 1, 1, -2, true);
          } else if (selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
            if (indexIsMajor) {
              LARGE_INDEX(T, IndexT, 2, 2, -2, true);
            } else {
              LARGE_INDEX(T, IndexT, 2, 2, -2, false);
            }
          } else if (selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
            if (indexIsMajor) {
              LARGE_INDEX(T, IndexT, 3, 3, -2, true);
            } else {
              LARGE_INDEX(T, IndexT, 3, 3, -2, false);
            }
          } else {
            LARGE_INDEX(T, IndexT, -1, -1, -1, true);
          }
        }
    }
    else{
      UNIMPLEMENTED() << "index_add kernel only support tensor which can use 32 bit index math caculate for now.";
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_INDEX_ADD_CUDA_KERNEL(dtype, index_dtype)                          \
  REGISTER_USER_KERNEL("index_add")                                    \
      .SetCreateFn<IndexAddGpuKernel<dtype, index_dtype>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("output", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("index", 0) == GetDataType<index_dtype>::value));

REGISTER_INDEX_ADD_CUDA_KERNEL(float, int32_t)
REGISTER_INDEX_ADD_CUDA_KERNEL(float, int64_t)
REGISTER_INDEX_ADD_CUDA_KERNEL(half, int32_t)
REGISTER_INDEX_ADD_CUDA_KERNEL(half, int64_t)
REGISTER_INDEX_ADD_CUDA_KERNEL(double, int32_t)
REGISTER_INDEX_ADD_CUDA_KERNEL(double, int64_t)

}  // namespace oneflow
