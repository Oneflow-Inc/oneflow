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
                                 const IDX indice_instance_size,
                                 NdIndexOffsetHelper<IDX, N> in_helper,
                                 NdIndexOffsetHelper<IDX, N> out_helper, const K* indices,
                                 const T* data, const IDX gather_dim_size, T* output,
                                 const IDX offset) {
  IDX index[N];
  constexpr int index_axis = N - 2;
  CUDA_1D_KERNEL_LOOP_T(IDX, i, batch_dim_size * out_instance_size) {
    out_helper.OffsetToNdIndex(i, index);
    index[index_axis] =
        indices[index[index_axis] + i / out_instance_size * indice_instance_size] - offset;
    T v{};
    if (index[index_axis] >= 0 && index[index_axis] < gather_dim_size) {
      v = data[in_helper.NdIndexToOffset(index)];
    }
    output[i] = v;
  }
}

bool IsSafeUseIndex32(int64_t batch_dim_size, int64_t outer_dim_size, int64_t gather_dim_size,
                      int64_t inner_dim_size, int64_t num_indices) {
  const int64_t in_elem_cnt = batch_dim_size * outer_dim_size * gather_dim_size * inner_dim_size;
  const int64_t out_elem_cnt = batch_dim_size * outer_dim_size * num_indices * inner_dim_size;
  return std::max(out_elem_cnt, in_elem_cnt) < GetMaxVal<int32_t>() / 2;
}

template<typename T, typename K, typename IDX>
void Dispatch2DimNum(ep::Stream* stream, int64_t gather_dim_size, int64_t inner_dim_size,
                     int64_t num_indices, int64_t offset, const K* indices, const T* data,
                     T* output) {
  constexpr int N = 2;
  const int64_t out_instance_size = num_indices * inner_dim_size;
  const int64_t out_elem_cnt = out_instance_size;
  NdIndexOffsetHelper<IDX, N> in_helper(gather_dim_size, inner_dim_size);
  NdIndexOffsetHelper<IDX, N> out_helper(num_indices, inner_dim_size);
  GatherForwardGpu<T, K, IDX, N><<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0,
                                   stream->As<ep::CudaStream>()->cuda_stream()>>>(
      /*batch_dim_size*/ 1, out_instance_size, num_indices, in_helper, out_helper, indices, data,
      gather_dim_size, output, offset);
}

template<typename T, typename K, typename IDX>
void Dispatch3DimNum(ep::Stream* stream, int64_t batch_dim_size, int64_t outer_dim_size,
                     int64_t gather_dim_size, int64_t inner_dim_size, int64_t num_indices,
                     int64_t offset, const K* indices, const T* data, T* output) {
  constexpr int N = 3;
  const int64_t indice_instance_size = num_indices / batch_dim_size;
  const int64_t out_instance_size = outer_dim_size * indice_instance_size * inner_dim_size;
  const int64_t out_elem_cnt = batch_dim_size * out_instance_size;

  const int64_t outest_dim_size = batch_dim_size * outer_dim_size;

  NdIndexOffsetHelper<IDX, N> in_helper(outest_dim_size, gather_dim_size, inner_dim_size);
  NdIndexOffsetHelper<IDX, N> out_helper(outest_dim_size, indice_instance_size, inner_dim_size);
  GatherForwardGpu<T, K, IDX, N><<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0,
                                   stream->As<ep::CudaStream>()->cuda_stream()>>>(
      batch_dim_size, out_instance_size, indice_instance_size, in_helper, out_helper, indices, data,
      gather_dim_size, output, offset);
}

template<typename T, typename K, typename IDX>
void Dispatch4DimNum(ep::Stream* stream, int64_t batch_dim_size, int64_t outer_dim_size,
                     int64_t gather_dim_size, int64_t inner_dim_size, int64_t num_indices,
                     int64_t offset, const K* indices, const T* data, T* output) {
  constexpr int N = 4;
  const int64_t indice_instance_size = num_indices / batch_dim_size;
  const int64_t out_instance_size = outer_dim_size * indice_instance_size * inner_dim_size;
  const int64_t out_elem_cnt = batch_dim_size * out_instance_size;
  NdIndexOffsetHelper<IDX, N> in_helper(batch_dim_size, outer_dim_size, gather_dim_size,
                                        inner_dim_size);
  NdIndexOffsetHelper<IDX, N> out_helper(batch_dim_size, outer_dim_size, indice_instance_size,
                                         inner_dim_size);
  GatherForwardGpu<T, K, IDX, N><<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0,
                                   stream->As<ep::CudaStream>()->cuda_stream()>>>(
      batch_dim_size, out_instance_size, indice_instance_size, in_helper, out_helper, indices, data,
      gather_dim_size, output, offset);
}

template<typename T, typename K, typename IDX>
void DispatchNumDims(ep::Stream* stream, int64_t batch_dim_size, int64_t outer_dim_size,
                     int64_t gather_dim_size, int64_t inner_dim_size, int64_t num_indices,
                     int64_t offset, const K* indices, const void* data, void* output) {
  if (batch_dim_size == 1 && outer_dim_size == 1) {
    Dispatch2DimNum<T, K, IDX>(stream, gather_dim_size, inner_dim_size, num_indices, offset,
                               indices, static_cast<const T*>(data), static_cast<T*>(output));
  } else if (batch_dim_size == 1 || outer_dim_size == 1) {
    Dispatch3DimNum<T, K, IDX>(stream, batch_dim_size, outer_dim_size, gather_dim_size,
                               inner_dim_size, num_indices, offset, indices,
                               static_cast<const T*>(data), static_cast<T*>(output));
  } else {
    Dispatch4DimNum<T, K, IDX>(stream, batch_dim_size, outer_dim_size, gather_dim_size,
                               inner_dim_size, num_indices, offset, indices,
                               static_cast<const T*>(data), static_cast<T*>(output));
  }
}

template<typename K, typename T>
void DispatchIndexSize(ep::Stream* stream, int64_t batch_dim_size, int64_t outer_dim_size,
                       int64_t gather_dim_size, int64_t inner_dim_size, int64_t num_indices,
                       int64_t offset, const K* indices, const void* data, void* output) {
  if (IsSafeUseIndex32(batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size,
                       num_indices)) {
    DispatchNumDims<T, K, int32_t>(stream, batch_dim_size, outer_dim_size, gather_dim_size,
                                   inner_dim_size / sizeof(T), num_indices, offset, indices,
                                   static_cast<const T*>(data), static_cast<T*>(output));
  } else {
    DispatchNumDims<T, K, int64_t>(stream, batch_dim_size, outer_dim_size, gather_dim_size,
                                   inner_dim_size / sizeof(T), num_indices, offset, indices,
                                   static_cast<const T*>(data), static_cast<T*>(output));
  }
}

template<typename K, typename T>
bool TryDispatchMovementType(ep::Stream* stream, int64_t batch_dim_size, int64_t outer_dim_size,
                             int64_t gather_dim_size, int64_t inner_dim_size, int64_t num_indices,
                             int64_t offset, const K* indices, const void* data, void* output) {
  if (reinterpret_cast<uintptr_t>(data) % sizeof(T) == 0
      && reinterpret_cast<uintptr_t>(output) % sizeof(T) == 0 && inner_dim_size % sizeof(T) == 0) {
    DispatchIndexSize<K, T>(stream, batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size,
                            num_indices, offset, indices, data, output);
    return true;
  } else {
    return false;
  }
}

template<typename K>
void DispatchMovementSize(ep::Stream* stream, int64_t batch_dim_size, int64_t outer_dim_size,
                          int64_t gather_dim_size, int64_t inner_dim_size, int64_t num_indices,
                          int64_t offset, const K* indices, const void* data, void* output) {
  using Func = bool (*)(ep::Stream * stream, int64_t batch_dim_size, int64_t outer_dim_size,
                        int64_t gather_dim_size, int64_t inner_dim_size, int64_t num_indices,
                        int64_t offset, const K* indices, const void* data, void* output);
  Func funcs[] = {
      TryDispatchMovementType<K, ulonglong2>,  // 16B
      TryDispatchMovementType<K, uint64_t>,    // 8B
      TryDispatchMovementType<K, uint32_t>,    // 4B
      TryDispatchMovementType<K, uint16_t>,    // 2B
      TryDispatchMovementType<K, uint8_t>,     // 1B
  };
  for (size_t i = 0; i < sizeof(funcs) / sizeof(funcs[0]); ++i) {
    if (funcs[i](stream, batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size,
                 num_indices, offset, indices, data, output)) {
      break;
    }
  }
}

template<typename T, typename K>
void GatherGpuKernel(Stream* stream, int64_t batch_dim_size, int64_t outer_dim_size,
                     int64_t gather_dim_size, int64_t inner_dim_size, const void* data,
                     int64_t num_indices, const void* indice, int64_t offset, void* output) {
  DispatchMovementSize(stream, batch_dim_size, outer_dim_size, gather_dim_size,
                       inner_dim_size * sizeof(T), num_indices, offset,
                       static_cast<const K*>(indice), data, output);
}

template<typename T, typename K>
class GatherImpl : public Gather {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherImpl);
  GatherImpl() = default;
  ~GatherImpl() = default;
  void Launch(Stream* stream, int64_t batch_dim_size, int64_t outer_dim_size,
              int64_t gather_dim_size, int64_t inner_dim_size, const void* data,
              int64_t num_indices, const void* indice, void* output) override {
    GatherGpuKernel<T, K>(stream, batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size,
                          data, num_indices, indice, /*offset*/ 0, output);
  }
  void Launch(Stream* stream, int64_t batch_dim_size, int64_t outer_dim_size,
              int64_t gather_dim_size, int64_t inner_dim_size, const void* data,
              int64_t num_indices, const void* indice, int64_t offset, void* output) override {
    GatherGpuKernel<T, K>(stream, batch_dim_size, outer_dim_size, gather_dim_size, inner_dim_size,
                          data, num_indices, indice, offset, output);
  }
};
template<typename T, typename K>
std::unique_ptr<Gather> NewGather() {
  return std::unique_ptr<Gather>(new GatherImpl<T, K>());
}
#define GATHER_DATA_TYPE_SEQ ARITHMETIC_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(bool, DataType::kBool)
#define GATHER_INDEX_TYPE_SEQ INDEX_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32)
class GatherFactoryImpl : public GatherFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherFactoryImpl);
  GatherFactoryImpl() = default;
  ~GatherFactoryImpl() override = default;
  std::unique_ptr<Gather> New(DataType data_dtype, DataType indice_type) override {
    std::tuple<DataType, DataType> type_tuple = std::make_tuple(data_dtype, indice_type);
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
#undef GATHER_INDEX_TYPE_SEQ
#undef GATHER_DATA_TYPE_SEQ
REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, GatherFactory, GatherFactoryImpl);
}  // namespace
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
