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
#include "oneflow/core/ep/include/primitive/permute.h"
#include "oneflow/core/ep/common/primitive/permute_impl.h"
#include "oneflow/core/ep/cpu/cpu_stream.h"
#include "oneflow/core/ep/cpu/cpu_device.h"
#include "oneflow/core/ep/common/onednn.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace permute {

namespace internal {

namespace {

template<size_t num_dims, size_t movement_size, typename IndexType>
void PermuteKernel(PermuteKernelParams<num_dims, IndexType> params) {
  using T = typename std::aligned_storage<movement_size, movement_size>::type;
  const T* src = reinterpret_cast<const T*>(params.src);
  T* dst = reinterpret_cast<T*>(params.dst);
  for (IndexType i = 0; i < params.count; ++i) {
    IndexType src_index[num_dims];
    IndexType dst_index[num_dims];
    params.dst_index_helper.OffsetToNdIndex(i, dst_index);
    for (size_t dim = 0; dim < num_dims; ++dim) {
      src_index[params.permutation[dim]] = dst_index[dim];
    }
    IndexType src_offset = params.src_index_helper.NdIndexToOffset(src_index);
    dst[i] = src[src_offset];
  }
}

template<size_t num_dims, size_t movement_size, typename IndexType>
void LaunchKernel(Stream* stream, const int64_t* src_dims, const void* src, const int* permutation,
                  void* dst, size_t count) {
  PermuteKernelParams<num_dims, IndexType> params =
      MakePermuteParams<num_dims, IndexType>(src_dims, src, permutation, dst, count);
  PermuteKernel<num_dims, movement_size, IndexType>(params);
}
class PermuteImpl : public Permute {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PermuteImpl);
  PermuteImpl() = default;
  ~PermuteImpl() override = default;

  using Permute::Launch;
  void Launch(Stream* stream, DataType data_type, size_t num_dims, const int64_t* src_dims,
              const void* src, const int* permutation, void* dst) override {
    SimplifyThenLaunch(stream, data_type, num_dims, src_dims, src, permutation, dst);
  }
};

#ifdef WITH_ONEDNN
constexpr size_t kMaxOneDnnMovementSize = 4;
constexpr size_t kMaxOneDnnMapSize = 5;
uint32_t OnednnDatatypeTagMap[kMaxOneDnnMapSize] = {0, dnnl_u8, dnnl_f16, 0, dnnl_s32};
class OneDnnPermuteImpl : public Permute {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OneDnnPermuteImpl);
  OneDnnPermuteImpl() = default;
  ~OneDnnPermuteImpl() override = default;

  using Permute::Launch;
  void Launch(Stream* stream, DataType data_type, size_t num_dims, const int64_t* src_dims,
              const void* src, const int* permutation, void* dst) override {
    CHECK_LE(num_dims, kMaxNumDims);
    CHECK_GT(num_dims, 0);

    stream->As<CpuStream>()->onednn_executor()->Launch([&](dnnl::engine* onednn_engine,
                                                           dnnl::stream* onednn_stream) {
      size_t onednn_num_dims = num_dims;
      dnnl::memory::dims onednn_dims(kMaxNumDims + 1, 0);
      dnnl::memory::dims onednn_permute(kMaxNumDims + 1, 0);
      dnnl::memory::dims src_stride(kMaxNumDims + 1, 0);
      dnnl::memory::dims dst_stride(kMaxNumDims + 1, 0);
      for (int64_t dim = onednn_num_dims - 1; dim >= 0; dim--) {
        onednn_dims[dim] = src_dims[dim];
        onednn_permute[dim] = permutation[dim];
      }
      size_t movement_size = GetSizeOfDataType(data_type);
      if (movement_size > kMaxOneDnnMovementSize) {
        onednn_dims[onednn_num_dims] = movement_size / kMaxOneDnnMovementSize;
        onednn_permute[onednn_num_dims] = onednn_num_dims;
        onednn_num_dims = onednn_num_dims + 1;
        movement_size = kMaxOneDnnMovementSize;
      }
      onednn_dims.resize(onednn_num_dims);

      src_stride[onednn_num_dims - 1] = 1;
      dst_stride[onednn_permute[onednn_num_dims - 1]] = 1;
      for (int64_t i = onednn_num_dims - 2; i >= 0; i--) {
        src_stride[i] = src_stride[i + 1] * onednn_dims[i + 1];
        dst_stride[onednn_permute[i]] =
            dst_stride[onednn_permute[i + 1]] * onednn_dims[onednn_permute[i + 1]];
      }

      dnnl::memory::data_type onednn_data_type =
          static_cast<dnnl::memory::data_type>(OnednnDatatypeTagMap[movement_size]);
      // The reorder primitive requires the source and destination tensors to have the same shape.
      // Implicit broadcasting is not supported.
      auto src_mem_desc = dnnl::memory::desc(onednn_dims, onednn_data_type, src_stride);
      auto dst_mem_desc = dnnl::memory::desc(onednn_dims, onednn_data_type, dst_stride);
      auto src_mem = dnnl::memory(src_mem_desc, *onednn_engine, const_cast<void*>(src));
      auto dst_mem = dnnl::memory(dst_mem_desc, *onednn_engine, dst);
      auto reorder_primitive_desc =
          dnnl::reorder::primitive_desc(*onednn_engine, src_mem_desc, *onednn_engine, dst_mem_desc);
      auto reorder_primitive = dnnl::reorder(reorder_primitive_desc);

      reorder_primitive.execute(*onednn_stream, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
    });
  }
};

#endif  // WITH_ONEDNN

class PermuteFactoryImpl : public PermuteFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PermuteFactoryImpl);
  PermuteFactoryImpl() = default;
  ~PermuteFactoryImpl() override = default;

  std::unique_ptr<Permute> New(size_t max_num_dims) override {
    if (max_num_dims <= kMaxNumDims) {
#ifdef WITH_ONEDNN
      if (OneDnnIsEnabled()) { return std::unique_ptr<Permute>(new OneDnnPermuteImpl()); }
#endif
      return std::unique_ptr<Permute>(new PermuteImpl());
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, PermuteFactory, PermuteFactoryImpl);

}  // namespace

}  // namespace internal

}  // namespace permute

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
