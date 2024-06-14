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
#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/ep/include/primitive/gather.h"
#include "oneflow/core/ep/cuda/primitive/type_seq.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/device/cuda_util.h"
#include <cuda.h>

namespace oneflow {

namespace ep {
namespace primitive {

namespace gather {

namespace {

template<typename PackedType, typename IndicesType, typename IDX, int NumDims,
         typename std::enable_if<NumDims == 3 || NumDims == 2, int>::type = 0>
__global__ void GatherForwardGpu(IDX elem_cnt, NdIndexOffsetHelper<IDX, NumDims> in_helper,
                                 NdIndexOffsetHelper<IDX, NumDims> out_helper,
                                 NdIndexOffsetHelper<IDX, 2> indices_helper, IDX gather_dim_size,
                                 const PackedType* in, PackedType* out, const IndicesType* indices,
                                 IDX indices_size, IDX offset) {
  IDX nd_index[NumDims];
  int16_t gather_dim_index = NumDims - 2;
  CUDA_1D_KERNEL_LOOP_T(IDX, out_offset, elem_cnt) {
    out_helper.OffsetToNdIndex(out_offset, nd_index);
    nd_index[gather_dim_index] = indices[nd_index[gather_dim_index]] - offset;
    PackedType v{};
    if (nd_index[gather_dim_index] >= 0 && nd_index[gather_dim_index] < gather_dim_size) {
      v = in[in_helper.NdIndexToOffset(nd_index)];
    }
    out[out_offset] = v;
  }
}

template<typename PackedType, typename IndicesType, typename IDX, int NumDims,
         typename std::enable_if<NumDims == 4, int>::type = 0>
__global__ void GatherForwardGpu(IDX elem_cnt, NdIndexOffsetHelper<IDX, NumDims> in_helper,
                                 NdIndexOffsetHelper<IDX, NumDims> out_helper,
                                 NdIndexOffsetHelper<IDX, 2> indices_helper, IDX gather_dim_size,
                                 const PackedType* in, PackedType* out, const IndicesType* indices,
                                 IDX indices_size, IDX offset) {
  IDX nd_index[NumDims];
  IDX indices_nd_index[2];
  int16_t gather_dim_index = NumDims - 2;
  CUDA_1D_KERNEL_LOOP_T(IDX, out_offset, elem_cnt) {
    out_helper.OffsetToNdIndex(out_offset, nd_index);
    indices_nd_index[0] = nd_index[0];                 // batch dim size
    indices_nd_index[1] = nd_index[gather_dim_index];  // gather dim size
    nd_index[gather_dim_index] = indices[indices_helper.NdIndexToOffset(indices_nd_index)] - offset;
    PackedType v{};
    if (nd_index[gather_dim_index] >= 0 && nd_index[gather_dim_index] < gather_dim_size) {
      v = in[in_helper.NdIndexToOffset(nd_index)];
    }
    out[out_offset] = v;
  }
}

bool IsSafeUseIndex32(const size_t batch_dim_size, const size_t outer_dim_size,
                      const size_t gather_dim_size, const size_t inner_dim_size,
                      const size_t indices_size) {
  const int64_t in_elem_cnt = batch_dim_size * outer_dim_size * gather_dim_size * inner_dim_size;
  const int64_t out_elem_cnt = outer_dim_size * indices_size * inner_dim_size;
  return std::max(out_elem_cnt, in_elem_cnt) < GetMaxVal<int32_t>() / 2;
}

template<typename PackedType, typename IndicesType, typename IDX>
void LaunchGatherKernel(cudaStream_t cuda_stream, IDX batch_dim_size, IDX outer_dim_size,
                        IDX gather_dim_size, IDX inner_dim_size, const PackedType* in,
                        PackedType* out, const IndicesType* indices, IDX indices_size, IDX offset) {
  // out_elem_cnt = batch_dim_size * outer_dim_size * indices_size/batch_dim_size * inner_dim_size
  IDX out_elem_cnt = outer_dim_size * indices_size * inner_dim_size;
  IDX out_gather_dim_size = indices_size / batch_dim_size;
  IDX per_batch_indices_size = indices_size / batch_dim_size;
  NdIndexOffsetHelper<IDX, 2> indices_helper(batch_dim_size, per_batch_indices_size);

  if (batch_dim_size == 1 && outer_dim_size == 1) {
    NdIndexOffsetHelper<IDX, 2> in_helper(gather_dim_size, inner_dim_size);
    NdIndexOffsetHelper<IDX, 2> out_helper(out_gather_dim_size, inner_dim_size);
    GatherForwardGpu<PackedType, IndicesType, IDX, 2>
        <<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
            out_elem_cnt, in_helper, out_helper, indices_helper, gather_dim_size, in, out, indices,
            indices_size, offset);
  } else if (batch_dim_size == 1) {
    NdIndexOffsetHelper<IDX, 3> in_helper(outer_dim_size, gather_dim_size, inner_dim_size);
    NdIndexOffsetHelper<IDX, 3> out_helper(outer_dim_size, out_gather_dim_size, inner_dim_size);
    GatherForwardGpu<PackedType, IndicesType, IDX, 3>
        <<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
            out_elem_cnt, in_helper, out_helper, indices_helper, gather_dim_size, in, out, indices,
            indices_size, offset);
  } else {
    NdIndexOffsetHelper<IDX, 4> in_helper(batch_dim_size, outer_dim_size, gather_dim_size,
                                          inner_dim_size);
    NdIndexOffsetHelper<IDX, 4> out_helper(batch_dim_size, outer_dim_size, out_gather_dim_size,
                                           inner_dim_size);
    GatherForwardGpu<PackedType, IndicesType, IDX, 4>
        <<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(
            out_elem_cnt, in_helper, out_helper, indices_helper, gather_dim_size, in, out, indices,
            indices_size, offset);
  }
}

template<typename PackedType, typename IndicesType>
void DispatchIndexSize(cudaStream_t cuda_stream, size_t batch_dim_size, size_t outer_dim_size,
                       size_t gather_dim_size, size_t inner_dim_size, const PackedType* in,
                       PackedType* out, const IndicesType* indices, size_t indices_size,
                       int64_t offset) {
  if (IsSafeUseIndex32(batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size,
                       indices_size)) {
    LaunchGatherKernel<PackedType, IndicesType, int32_t>(
        cuda_stream, batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size, in, out,
        indices, indices_size, offset);
  } else {
    LaunchGatherKernel<PackedType, IndicesType, int64_t>(
        cuda_stream, batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size, in, out,
        indices, indices_size, offset);
  }
}

template<typename IndicesType, typename PackedType>
bool TryDispatchPackedType(cudaStream_t cuda_stream, size_t batch_dim_size, size_t outer_dim_size,
                           size_t gather_dim_size, size_t inner_dim_byte_size, const void* in,
                           void* out, const IndicesType* indices, size_t indices_size,
                           int64_t offset) {
#define isAligned(src, alignment) reinterpret_cast<uintptr_t>(src) % sizeof(alignment) == 0
  if (isAligned(in, PackedType) && isAligned(out, PackedType)
      && inner_dim_byte_size % sizeof(PackedType) == 0) {
    DispatchIndexSize<PackedType, IndicesType>(
        cuda_stream, batch_dim_size, outer_dim_size, gather_dim_size,
        inner_dim_byte_size / sizeof(PackedType), static_cast<const PackedType*>(in),
        static_cast<PackedType*>(out), indices, indices_size, offset);
    return true;
  } else {
    return false;
  }
#undef isAligned
}

template<typename IndicesType>
void DispatchPackedSize(cudaStream_t cuda_stream, size_t batch_dim_size, size_t outer_dim_size,
                        size_t gather_dim_size, unsigned long inner_dim_byte_size, const void* in,
                        void* out, const IndicesType* indices, size_t indices_size,
                        int64_t offset) {
  using Func = bool (*)(cudaStream_t cuda_stream, size_t batch_dim_size, size_t outer_dim_size,
                        size_t gather_dim_size, size_t inner_dim_byte_size, const void* in,
                        void* out, const IndicesType* indices, size_t indices_size, int64_t offset);

  Func func[] = {
      TryDispatchPackedType<IndicesType, ulonglong2>,  // 16-Bytes
      TryDispatchPackedType<IndicesType, uint64_t>,    // 8-Bytes
      TryDispatchPackedType<IndicesType, uint32_t>,    // 4-Bytes
      TryDispatchPackedType<IndicesType, uint16_t>,    // 2-Bytes
      TryDispatchPackedType<IndicesType, uint8_t>,     // 1-Bytes
  };

  for (size_t i = 0; i < sizeof(func) / sizeof(func[0]); i++) {
    if (func[i](cuda_stream, batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_byte_size,
                in, out, indices, indices_size, offset)) {
      break;
    }
  }
}

template<typename ParamsType, typename IndicesType>
class GatherImpl : public Gather {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherImpl);
  GatherImpl() = default;
  ~GatherImpl() override = default;

  using Gather::Launch;
  void Launch(Stream* stream, size_t batch_dim_size, size_t outer_dim_size, size_t gather_dim_size,
              size_t inner_dim_size, size_t offset, const void* in, size_t indices_size,
              const void* indices, void* out) override {
    cudaStream_t cuda_stream = stream->As<CudaStream>()->cuda_stream();
    DispatchPackedSize<IndicesType>(cuda_stream, batch_dim_size, outer_dim_size, gather_dim_size,
                                    inner_dim_size * sizeof(ParamsType), in, out,
                                    reinterpret_cast<IndicesType*>(const_cast<void*>(indices)),
                                    indices_size, offset);
  }
};

template<typename ParamsType, typename IndicesType>
std::unique_ptr<Gather> NewGather() {
  return std::unique_ptr<Gather>(new GatherImpl<ParamsType, IndicesType>());
}

class GatherFactoryImpl : public GatherFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherFactoryImpl);
  GatherFactoryImpl() = default;
  ~GatherFactoryImpl() override = default;

  std::unique_ptr<Gather> New(DataType params_type, DataType indices_type) override {
#define MAKE_NEW_GATHER_ENTRY(params_type_pair, indices_type_pair)                            \
  {std::make_pair(OF_PP_PAIR_SECOND(params_type_pair), OF_PP_PAIR_SECOND(indices_type_pair)), \
   NewGather<OF_PP_PAIR_FIRST(params_type_pair), OF_PP_PAIR_FIRST(indices_type_pair)>},

    static const std::map<std::pair<DataType, DataType>, std::function<std::unique_ptr<Gather>()>>
        new_gather_handle{OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
            MAKE_NEW_GATHER_ENTRY, CUDA_PRIMITIVE_ALL_TYPE_SEQ, CUDA_PRIMITIVE_INT_TYPE_SEQ)};

#undef MAKE_NEW_GATHER_ENTRY

    const auto it = new_gather_handle.find(std::make_pair(params_type, indices_type));
    if (it != new_gather_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, GatherFactory, GatherFactoryImpl);

}  // namespace

}  // namespace gather

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
