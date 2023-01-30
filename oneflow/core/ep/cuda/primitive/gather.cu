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

namespace internal {

template<typename PackedType, typename IndicesType, typename IDX>
__global__ void GatherForwardGpu(const IDX elem_cnt,
                                 NdIndexOffsetHelper<IDX, 3> per_batch_in_helper,
                                 NdIndexOffsetHelper<IDX, 3> per_batch_out_helper,
                                 const IDX batch_dim_size, const IDX per_batch_out_size,
                                 const IDX per_batch_in_size, const IDX per_batch_indices_size,
                                 const IDX gather_dim_size, const PackedType* in, const IDX in_size,
                                 PackedType* out, const IDX out_size, const IndicesType* indices,
                                 const IDX indices_size, const IDX offset) {
  IDX batch_index[3];
  CUDA_1D_KERNEL_LOOP_T(IDX, out_offset, elem_cnt) {
    const IDX batch_id = out_offset / per_batch_out_size;
    const IDX batch_offset = out_offset % per_batch_out_size;
    per_batch_out_helper.OffsetToNdIndex(batch_offset, batch_index);
    batch_index[1] = indices[per_batch_indices_size * batch_id + batch_index[1]] - offset;
    PackedType v{};
    if (batch_index[1] >= 0 && batch_index[1] < gather_dim_size) {
      v = in[per_batch_in_size * batch_id + per_batch_in_helper.NdIndexToOffset(batch_index)];
    }
    out[out_offset] = v;
  }
}

bool IsSafeUseIndex32(const size_t batch_dim_size, const size_t outer_dim_size,
                      const size_t gather_dim_size, const size_t inner_dim_size,
                      const size_t indices_size) {
  const int64_t in_elem_cnt = batch_dim_size * outer_dim_size * gather_dim_size * inner_dim_size;
  const int64_t out_elem_cnt = outer_dim_size * indices_size * inner_dim_size;
  // out_elem_cnt = batch_dim_size * outer_dim_size * indices_size/batch_dim_size * inner_dim_size
  return std::max(out_elem_cnt, in_elem_cnt) < GetMaxVal<int32_t>() / 2;
}

template<typename PackedType, typename IndicesType>
void DispatchIndexSize(cudaStream_t cuda_stream, const size_t batch_dim_size,
                       const size_t outer_dim_size, const size_t gather_dim_size,
                       const size_t inner_dim_size, const PackedType* in, const size_t in_size,
                       PackedType* out, const size_t out_size, const IndicesType* indices,
                       const size_t indices_size, const int64_t offset) {
  const int64_t out_elem_cnt = outer_dim_size * indices_size * inner_dim_size;
  // out_elem_cnt = batch_dim_size * outer_dim_size * indices_size/batch_dim_size * inner_dim_size
  const int64_t per_batch_out_size = out_size / batch_dim_size;
  const int64_t per_batch_in_size = in_size / batch_dim_size;
  const int64_t per_batch_indices_size = indices_size / batch_dim_size;

#define LAUNCH_GATHER_KERNEL(IDX)                                                                 \
  NdIndexOffsetHelper<IDX, 3> per_batch_in_helper(outer_dim_size, gather_dim_size,                \
                                                  inner_dim_size);                                \
  NdIndexOffsetHelper<IDX, 3> per_batch_out_helper(outer_dim_size, indices_size / batch_dim_size, \
                                                   inner_dim_size);                               \
  GatherForwardGpu<PackedType, IndicesType, IDX>                                                  \
      <<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(          \
          out_elem_cnt, per_batch_in_helper, per_batch_out_helper, batch_dim_size,                \
          per_batch_out_size, per_batch_in_size, per_batch_indices_size, gather_dim_size, in,     \
          in_size, out, out_size, indices, indices_size, offset);

  if (IsSafeUseIndex32(batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size,
                       indices_size)) {
    LAUNCH_GATHER_KERNEL(int32_t)
  } else {
    LAUNCH_GATHER_KERNEL(int64_t)
  }

#undef LAUNCH_GATHER_KERNEL
}

template<typename IndicesType, typename PackedType>
bool TryDispatchPackedType(cudaStream_t cuda_stream, const size_t batch_dim_size,
                           const size_t outer_dim_size, const size_t gather_dim_size,
                           const size_t inner_dim_byte_size, const void* in, const size_t in_size,
                           void* out, const size_t out_size, const IndicesType* indices,
                           const size_t indices_size, const int64_t offset) {
#define isAligned(src, alignment) reinterpret_cast<uintptr_t>(src) % sizeof(alignment) == 0
  if (isAligned(in, PackedType) && isAligned(out, PackedType)
      && inner_dim_byte_size % sizeof(PackedType) == 0) {
    DispatchIndexSize<PackedType, IndicesType>(
        cuda_stream, batch_dim_size, outer_dim_size, gather_dim_size,
        inner_dim_byte_size / sizeof(PackedType), static_cast<const PackedType*>(in), in_size,
        static_cast<PackedType*>(out), out_size, indices, indices_size, offset);
    return true;
  } else {
    return false;
  }
#undef isAligned
}

template<typename IndicesType>
void DispatchPackedSize(cudaStream_t cuda_stream, const size_t batch_dim_size,
                        const size_t outer_dim_size, const size_t gather_dim_size,
                        const unsigned long inner_dim_byte_size, const void* in,
                        const size_t in_size, void* out, const size_t out_size,
                        const IndicesType* indices, const size_t indices_size,
                        const int64_t offset) {
  using Func =
      bool (*)(cudaStream_t cuda_stream, const size_t batch_dim_size, const size_t outer_dim_size,
               const size_t gather_dim_size, const size_t inner_dim_byte_size, const void* in,
               const size_t in_size, void* out, const size_t out_size, const IndicesType* indices,
               const size_t indices_size, const int64_t offset);

  Func func[] = {
      TryDispatchPackedType<IndicesType, ulonglong2>,  // 16-Bytes
      TryDispatchPackedType<IndicesType, uint64_t>,    // 8-Bytes
      TryDispatchPackedType<IndicesType, uint32_t>,    // 4-Bytes
      TryDispatchPackedType<IndicesType, uint16_t>,    // 2-Bytes
      TryDispatchPackedType<IndicesType, uint8_t>,     // 1-Bytes
  };

  for (size_t i = 0; i < sizeof(func) / sizeof(func[0]); i++) {
    if (func[i](cuda_stream, batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_byte_size,
                in, in_size, out, out_size, indices, indices_size, offset)) {
      break;
    }
  }
}

template<typename T, typename IndicesType>
class GatherImpl : public Gather {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherImpl);
  GatherImpl() = default;
  ~GatherImpl() override = default;

  using Gather::Launch;
  void Launch(Stream* stream, const size_t batch_dim_size, const size_t outer_dim_size,
              const size_t gather_dim_size, const size_t inner_dim_size, const void* in,
              const size_t in_size, void* out, const size_t out_size, const void* indices,
              const size_t indices_size, const int64_t offset) override {
    cudaStream_t cuda_stream = stream->As<CudaStream>()->cuda_stream();
    DispatchPackedSize<IndicesType>(cuda_stream, batch_dim_size, outer_dim_size, gather_dim_size,
                                    inner_dim_size * sizeof(T), in, in_size, out, out_size,
                                    reinterpret_cast<IndicesType*>(const_cast<void*>(indices)),
                                    indices_size, offset);
  }
};

template<typename T, typename IndicesType>
std::unique_ptr<Gather> NewGather() {
  return std::unique_ptr<Gather>(new GatherImpl<T, IndicesType>());
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

}  // namespace internal

}  // namespace gather

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
