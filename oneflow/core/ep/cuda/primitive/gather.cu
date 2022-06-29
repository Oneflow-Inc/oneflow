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
#include "oneflow/core/ep/include/primitive/gather.h"
#include "oneflow/core/ep/cpu/primitive/type_seq.h"
#include "oneflow/core/ep/common/primitive/util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
namespace oneflow {
namespace ep {
namespace primitive {
namespace {
template<typename T, typename K, typename IDX>
__global__ void GatherForwardGpu(const IDX elem_cnt, NdIndexOffsetHelper<IDX, 3> in_helper,
                                 NdIndexOffsetHelper<IDX, 3> out_helper, const K* indices,
                                 const T* in, const IDX gather_dim_size, T* out, const IDX offset) {
  IDX index[3];
  CUDA_1D_KERNEL_LOOP_T(IDX, i, elem_cnt) {
    out_helper.OffsetToNdIndex(i, index);
    index[1] = indices[index[1]] - offset;
    T v{};
    if (index[1] >= 0 && index[1] < gather_dim_size) { v = in[in_helper.NdIndexToOffset(index)]; }
    out[i] = v;
  }
}

bool IsSafeUseIndex32(int64_t outer_dim_size, int64_t gather_dim_size, int64_t inner_dim_size,
                      int64_t num_indices) {
  const int64_t in_elem_cnt = outer_dim_size * gather_dim_size * inner_dim_size;
  const int64_t out_elem_cnt = outer_dim_size * num_indices * inner_dim_size;
  return std::max(out_elem_cnt, in_elem_cnt) < GetMaxVal<int32_t>() / 2;
}

template<typename T, typename K>
void DispatchIndexSize(ep::Stream* stream, int64_t outer_dim_size, int64_t gather_dim_size,
                       int64_t inner_dim_size, int64_t num_indices, int64_t offset,
                       const K* indices, const T* in, T* out) {
  const int64_t out_elem_cnt = outer_dim_size * num_indices * inner_dim_size;
  if (IsSafeUseIndex32(outer_dim_size, gather_dim_size, inner_dim_size, num_indices)) {
    NdIndexOffsetHelper<int32_t, 3> in_helper(outer_dim_size, gather_dim_size, inner_dim_size);
    NdIndexOffsetHelper<int32_t, 3> out_helper(outer_dim_size, num_indices, inner_dim_size);
    GatherForwardGpu<T, K, int32_t><<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock,
                                      0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
        out_elem_cnt, in_helper, out_helper, indices, in, gather_dim_size, out, offset);
  } else {
    NdIndexOffsetHelper<int64_t, 3> in_helper(outer_dim_size, gather_dim_size, inner_dim_size);
    NdIndexOffsetHelper<int64_t, 3> out_helper(outer_dim_size, num_indices, inner_dim_size);
    GatherForwardGpu<T, K, int64_t><<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock,
                                      0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
        out_elem_cnt, in_helper, out_helper, indices, in, gather_dim_size, out, offset);
  }
}

template<typename K, typename T>
bool TryDispatchMovementType(ep::Stream* stream, int64_t outer_dim_size, int64_t gather_dim_size,
                             int64_t inner_dim_size, int64_t num_indices, int64_t offset,
                             const K* indices, const void* in, void* out) {
  if (reinterpret_cast<uintptr_t>(in) % sizeof(T) == 0
      && reinterpret_cast<uintptr_t>(out) % sizeof(T) == 0 && inner_dim_size % sizeof(T) == 0) {
    DispatchIndexSize<T, K>(stream, outer_dim_size, gather_dim_size, inner_dim_size / sizeof(T),
                            num_indices, offset, indices, static_cast<const T*>(in),
                            static_cast<T*>(out));
    return true;
  } else {
    return false;
  }
}

template<typename K>
void DispatchMovementSize(ep::Stream* stream, int64_t outer_dim_size, int64_t gather_dim_size,
                          int64_t inner_dim_size, int64_t num_indices, int64_t offset,
                          const K* indices, const void* in, void* out) {
  using Func = bool (*)(ep::Stream * stream, int64_t outer_dim_size, int64_t gather_dim_size,
                        int64_t inner_dim_size, int64_t num_indices, int64_t offset,
                        const K* indices, const void* in, void* out);
  Func funcs[] = {
      TryDispatchMovementType<K, ulonglong2>,  // 16B
      TryDispatchMovementType<K, uint64_t>,    // 8B
      TryDispatchMovementType<K, uint32_t>,    // 4B
      TryDispatchMovementType<K, uint16_t>,    // 2B
      TryDispatchMovementType<K, uint8_t>,     // 1B
  };
  for (size_t i = 0; i < sizeof(funcs) / sizeof(funcs[0]); ++i) {
    if (funcs[i](stream, outer_dim_size, gather_dim_size, inner_dim_size, num_indices, offset,
                 indices, in, out)) {
      break;
    }
  }
}

template<typename T, typename K>
class GatherImpl : public Gather {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherImpl);
  GatherImpl() = default;
  ~GatherImpl() = default;
  void Launch(Stream* stream, const void* src, void* dst, const void* indice,
              const size_t num_indices, const size_t src_dim0, const size_t src_dim1,
              const size_t src_dim2, const size_t offset) override {
    DispatchMovementSize(stream, src_dim0, src_dim1, src_dim2 * sizeof(T), num_indices, offset,
                         indice, src, dst);
  }
};
template<typename T, typename K>
std::unique_ptr<Gather> NewGather() {
  return std::unique_ptr<Gather>(new GatherImpl<T, K>());
}
class GatherFactoryImpl : public GatherFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherFactoryImpl);
  GatherFactoryImpl() = default;
  ~GatherFactoryImpl() = default;
  std::unique_ptr<Gather> New(DataType data_type) override {
#define MAKE_NEW_GATHER_ENTRY(in_type, indice_type) \
  {OF_PP_PAIR_SECOND(in_type), NewGather<OF_PP_PAIR_FIRST(in_type), OF_PP_PAIR_FIRST(indice_type)>},

    static const std::map<DataType, std::function<std::unique_ptr<Gather>()>> new_gather_handle{
        OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_GATHER_ENTRY, CPU_PRIMITIVE_FLOATING_TYPE_SEQ)};

#undef MAKE_NEW_GATHER_ENTRY
    return NewPrimitiveFromHandlers(new_gather_handle, data_type);
  }
};
REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, GatherFactory, GatherFactoryImpl);
}  // namespace
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
