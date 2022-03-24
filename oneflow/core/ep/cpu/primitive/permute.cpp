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

enum PermuteKey {
  HWC2CHW = 0x102,
  CHW2HWC = 0x201,
};

size_t perm2key(size_t num_dims, const int* permutation) {
  uint64_t key = 0;
  for (int i = 0; i < num_dims; i++) { key |= permutation[i] << i * 4; }
  return key;
}

template<size_t movement_size>
void PermuteSpecialCaseHWC2CHW(size_t num_dims, const int64_t* src_dims, const void* src,
                               const int* permutation, void* dst) {
  using T = typename std::aligned_storage<movement_size, movement_size>::type;
  size_t dim0_num = src_dims[0] * src_dims[1];
  size_t dim1_num = src_dims[2];
  T* src_ptr = (T*)src;
  T* dst_ptr = (T*)dst;

  if (dim1_num == 3) {
    T* r = dst_ptr;
    T* g = dst_ptr + (dim0_num);
    T* b = dst_ptr + 2 * (dim0_num);

    for (int i = 0; i < dim0_num; i++) {
      r[i] = src_ptr[3 * i];
      g[i] = src_ptr[3 * i + 1];
      b[i] = src_ptr[3 * i + 2];
    }

  } else {
    for (int64_t i = 0; i < dim0_num; i++) {
      T* c_ptr = src_ptr + (i * dim1_num);
      for (int64_t j = 0; j < dim1_num; j++) { dst_ptr[j * dim0_num + i] = c_ptr[j]; }
    }
  }
}

template<size_t movement_size>
void PermuteSpecialCaseCHW2HWC(size_t num_dims, const int64_t* src_dims, const void* src,
                               const int* permutation, void* dst) {
  using T = typename std::aligned_storage<movement_size, movement_size>::type;
  size_t dim1_num = src_dims[1] * src_dims[2];
  size_t dim0_num = src_dims[0];
  T* src_ptr = (T*)src;
  T* dst_ptr = (T*)dst;

  if (dim0_num == 3) {
    T* r = src_ptr;
    T* g = src_ptr + (dim1_num);
    T* b = src_ptr + 2 * (dim1_num);

    for (int i = 0; i < dim1_num; i++) {
      dst_ptr[3 * i] = r[i];
      dst_ptr[3 * i + 1] = g[i];
      dst_ptr[3 * i + 2] = b[i];
    }
  } else {
    for (int64_t i = 0; i < dim1_num; i++) {
      T* c_ptr = dst_ptr + (i * dim1_num);
      for (int64_t j = 0; j < dim0_num; j++) { c_ptr[j] = src_ptr[i * dim0_num + j]; }
    }
  }
}

void LaunchSpecialCaseHWC2CHW(DataType data_type, size_t num_dims, const int64_t* src_dims,
                              const void* src, const int* permutation, void* dst) {
  void (*func)(size_t num_dims, const int64_t* src_dims, const void* src, const int* permutation,
               void* dst) = nullptr;
  size_t movement_size = GetSizeOfDataType(data_type);
  if (movement_size == 1) {
    func = PermuteSpecialCaseHWC2CHW<1>;
  } else if (movement_size == 2) {
    func = PermuteSpecialCaseHWC2CHW<2>;
  } else if (movement_size == 4) {
    func = PermuteSpecialCaseHWC2CHW<4>;
  } else if (movement_size == 8) {
    func = PermuteSpecialCaseHWC2CHW<8>;
  } else if (movement_size == 16) {
    func = PermuteSpecialCaseHWC2CHW<16>;
  } else {
    UNIMPLEMENTED();
  }
  func(num_dims, src_dims, src, permutation, dst);
  printf("hwc-chw \n");
}

void LaunchSpecialCaseCHW2HWC(DataType data_type, size_t num_dims, const int64_t* src_dims,
                              const void* src, const int* permutation, void* dst) {
  void (*func)(size_t num_dims, const int64_t* src_dims, const void* src, const int* permutation,
               void* dst) = nullptr;
  size_t movement_size = GetSizeOfDataType(data_type);
  if (movement_size == 1) {
    func = PermuteSpecialCaseCHW2HWC<1>;
  } else if (movement_size == 2) {
    func = PermuteSpecialCaseCHW2HWC<2>;
  } else if (movement_size == 4) {
    func = PermuteSpecialCaseCHW2HWC<4>;
  } else if (movement_size == 8) {
    func = PermuteSpecialCaseCHW2HWC<8>;
  } else if (movement_size == 16) {
    func = PermuteSpecialCaseCHW2HWC<16>;
  } else {
    UNIMPLEMENTED();
  }
  func(num_dims, src_dims, src, permutation, dst);
  printf("chw-hwc \n");
}

bool PermuteSpecialCase(DataType data_type, size_t num_dims, const int64_t* src_dims,
                        const void* src, const int* permutation, void* dst) {
  size_t key = perm2key(num_dims, permutation);
  if (HWC2CHW == key) {
    LaunchSpecialCaseHWC2CHW(data_type, num_dims, src_dims, src, permutation, dst);
    return true;
  } else if (CHW2HWC == key) {
    LaunchSpecialCaseCHW2HWC(data_type, num_dims, src_dims, src, permutation, dst);
    return true;
  }
  return false;
}

constexpr size_t kMaxOneDnnMapSize = 5;
constexpr size_t kMaxOneDNNMovementSize = 4;
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

    // SpecialCase
    if (PermuteSpecialCase(data_type, num_dims, src_dims, src, permutation, dst)) { return; }

    CpuStream* cpu_stream = stream->As<CpuStream>();
    size_t num_threads = static_cast<CpuDevice*>(cpu_stream->device())->GetNumThreads();
    CpuNumThreadsGuard guard(num_threads);

    dnnl::engine* onednn_engine = stream->As<CpuStream>()->onednn_engine();
    dnnl::stream* onednn_stream = stream->As<CpuStream>()->onednn_stream();

    size_t onednn_num_dims = num_dims;
    dnnl::memory::dims onednn_dims(onednn_num_dims + 1, 0);
    dnnl::memory::dims onednn_perm(onednn_num_dims + 1, 0);
    dnnl::memory::dims src_stride(onednn_num_dims + 1, 0);
    dnnl::memory::dims dst_stride(onednn_num_dims + 1, 0);

    for (int64_t dim = onednn_num_dims - 1; dim >= 0; dim--) {
      onednn_dims[dim] = src_dims[dim];
      onednn_perm[dim] = permutation[dim];
    }

    size_t movement_size = GetSizeOfDataType(data_type);
    if (movement_size > kMaxOneDNNMovementSize) {
      onednn_dims[onednn_num_dims] = movement_size / kMaxOneDNNMovementSize;
      onednn_perm[onednn_num_dims] = onednn_num_dims;
      onednn_num_dims = onednn_num_dims + 1;
      movement_size = kMaxOneDNNMovementSize;
    }

    src_stride[onednn_num_dims - 1] = 1;
    dst_stride[onednn_perm[onednn_num_dims - 1]] = 1;

    for (int64_t dim = onednn_num_dims - 2; dim >= 0; dim--) {
      int index = onednn_perm[dim + 1];
      src_stride[dim] = src_stride[dim + 1] * onednn_dims[dim + 1];
      dst_stride[onednn_perm[dim]] = dst_stride[index] * onednn_dims[index];
    }

    dnnl::memory::data_type onednn_data_type =
        static_cast<dnnl::memory::data_type>(OnednnDatatypeTagMap[movement_size]);
    auto src_md = dnnl::memory::desc(onednn_dims, onednn_data_type, src_stride);
    auto dst_md = dnnl::memory::desc(onednn_dims, onednn_data_type, dst_stride);
    auto src_mem = dnnl::memory(src_md, *onednn_engine, const_cast<void*>(src));
    auto dst_mem = dnnl::memory(dst_md, *onednn_engine, dst);
    auto reorder_pd = dnnl::reorder::primitive_desc(*onednn_engine, src_md, *onednn_engine, dst_md);
    auto reorder_prim = dnnl::reorder(reorder_pd);
    std::unordered_map<int, dnnl::memory> reorder_args{{DNNL_ARG_SRC, src_mem},
                                                       {DNNL_ARG_DST, dst_mem}};
    reorder_prim.execute(*onednn_stream, reorder_args);
    onednn_stream->wait();
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
      return std::unique_ptr<Permute>(new OneDnnPermuteImpl());
#else
      return std::unique_ptr<Permute>(new PermuteImpl());
#endif  // WITH_ONEDNN

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
