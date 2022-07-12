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
template<typename T, typename K, typename IDX, int N>
__global__ void GatherForwardGpu(const IDX batch_dim_size, const IDX out_instance_size,
                                 NdIndexOffsetHelper<IDX, N> in_helper,
                                 NdIndexOffsetHelper<IDX, N> out_helper, const K* indices,
                                 const T* in, const IDX gather_dim_size, T* out, const IDX offset) {
  IDX index[N];
  CUDA_1D_KERNEL_LOOP_T(IDX, i, batch_dim_size * out_instance_size) {
    out_helper.OffsetToNdIndex(i, index);
    index[2] = indices[index[2] + i % out_instance_size] - offset;
    T v{};
    if (index[2] >= 0 && index[2] < gather_dim_size) { v = in[in_helper.NdIndexToOffset(index)]; }
    out[i] = v;
  }
}

bool IsSafeUseIndex32(int64_t batch_dim_size, int64_t outer_dim_size, int64_t gather_dim_size,
                      int64_t inner_dim_size, int64_t num_indices) {
  const int64_t in_elem_cnt = batch_dim_size * outer_dim_size * gather_dim_size * inner_dim_size;
  const int64_t out_elem_cnt = batch_dim_size * outer_dim_size * num_indices * inner_dim_size;
  return std::max(out_elem_cnt, in_elem_cnt) < GetMaxVal<int32_t>() / 2;
}

template<typename T, typename K, int N>
void DispatchIndexSize(ep::Stream* stream, int64_t batch_dim_size, int64_t outer_dim_size,
                       int64_t gather_dim_size, int64_t inner_dim_size, int64_t num_indices,
                       int64_t offset, const K* indices, const T* in, T* out) {
  const int64_t out_instance_size = outer_dim_size * num_indices * inner_dim_size;
  const int64_t out_elem_cnt = batch_dim_size * out_instance_size;
  if (IsSafeUseIndex32(batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size,
                       num_indices)) {
    NdIndexOffsetHelper<int32_t, N> in_helper(batch_dim_size, outer_dim_size, gather_dim_size,
                                              inner_dim_size);
    NdIndexOffsetHelper<int32_t, N> out_helper(batch_dim_size, outer_dim_size, num_indices,
                                               inner_dim_size);
    GatherForwardGpu<T, K, int32_t, N>
        <<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(batch_dim_size, out_instance_size,
                                                          in_helper, out_helper, indices, in,
                                                          gather_dim_size, out, offset);
  } else {
    NdIndexOffsetHelper<int64_t, N> in_helper(batch_dim_size, outer_dim_size, gather_dim_size,
                                              inner_dim_size);
    NdIndexOffsetHelper<int64_t, N> out_helper(batch_dim_size, outer_dim_size, num_indices,
                                               inner_dim_size);
    GatherForwardGpu<T, K, int64_t, N>
        <<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(batch_dim_size, out_instance_size,
                                                          in_helper, out_helper, indices, in,
                                                          gather_dim_size, out, offset);
  }
}

template<typename K, typename T>
bool TryDispatchMovementType(ep::Stream* stream, int64_t batch_dim_size, int64_t outer_dim_size,
                             int64_t gather_dim_size, int64_t inner_dim_size, int64_t num_indices,
                             int64_t offset, const K* indices, const void* in, void* out) {
  if (reinterpret_cast<uintptr_t>(in) % sizeof(T) == 0
      && reinterpret_cast<uintptr_t>(out) % sizeof(T) == 0 && inner_dim_size % sizeof(T) == 0) {
    DispatchIndexSize<T, K, 4>(stream, batch_dim_size, outer_dim_size, gather_dim_size,
                               inner_dim_size / sizeof(T), num_indices, offset, indices,
                               static_cast<const T*>(in), static_cast<T*>(out));
    return true;
  } else {
    return false;
  }
}

template<typename K>
void DispatchMovementSize(ep::Stream* stream, int64_t batch_dim_size, int64_t outer_dim_size,
                          int64_t gather_dim_size, int64_t inner_dim_size, int64_t num_indices,
                          int64_t offset, const K* indices, const void* in, void* out) {
  using Func = bool (*)(ep::Stream * stream, int64_t batch_dim_size, int64_t outer_dim_size,
                        int64_t gather_dim_size, int64_t inner_dim_size, int64_t num_indices,
                        int64_t offset, const K* indices, const void* in, void* out);
  Func funcs[] = {
      TryDispatchMovementType<K, ulonglong2>,  // 16B
      TryDispatchMovementType<K, uint64_t>,    // 8B
      TryDispatchMovementType<K, uint32_t>,    // 4B
      TryDispatchMovementType<K, uint16_t>,    // 2B
      TryDispatchMovementType<K, uint8_t>,     // 1B
  };
  for (size_t i = 0; i < sizeof(funcs) / sizeof(funcs[0]); ++i) {
    if (funcs[i](stream, batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size,
                 num_indices, offset, indices, in, out)) {
      break;
    }
  }
}

template<typename T, typename K>
void GatherGpuKernel(Stream* stream, const void* src, void* dst, const void* indice,
                     const size_t num_indices, const size_t batch_dim_size,
                     const size_t outer_dim_size, const size_t gather_dim_size,
                     const size_t inner_dim_size) {
  DispatchMovementSize(stream, batch_dim_size, outer_dim_size, gather_dim_size,
                       inner_dim_size * sizeof(T), num_indices, 0, static_cast<const K*>(indice),
                       src, dst);
}

template<typename T, typename K>
class GatherImpl : public Gather {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherImpl);
  GatherImpl() = default;
  ~GatherImpl() = default;
  void Launch(Stream* stream, const void* src, void* dst, const void* indice,
              const size_t num_indices, const size_t batch_dim_size, const size_t outer_dim_size,
              const size_t gather_dim_size, const size_t inner_dim_size) override {
    GatherGpuKernel<T, K>(stream, src, dst, indice, num_indices, batch_dim_size, outer_dim_size,
                          gather_dim_size, inner_dim_size);
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
  std::unique_ptr<Gather> New(std::tuple<DataType, DataType> type_tuple) override {
#define MAKE_NEW_GATHER_ENTRY(in_type_pair, indice_type_pair)                             \
  {std::make_tuple(OF_PP_PAIR_SECOND(in_type_pair), OF_PP_PAIR_SECOND(indice_type_pair)), \
   NewGather<OF_PP_PAIR_FIRST(in_type_pair), OF_PP_PAIR_FIRST(indice_type_pair)>},

    static const std::map<std::tuple<DataType, DataType>, std::function<std::unique_ptr<Gather>()>>
        new_gather_handle{OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
            MAKE_NEW_GATHER_ENTRY, GATHER_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)};

#undef MAKE_NEW_GATHER_ENTRY
    return NewPrimitiveFromHandlers(new_gather_handle, type_tuple);
  }
};
REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, GatherFactory, GatherFactoryImpl);
}  // namespace
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
