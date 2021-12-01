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
#include "oneflow/core/ep/include/primitive//broadcast_elementwise_binary.h"
#include "oneflow/core/ep/common/primitive/broadcast_elementwise_binary.h"
#include "oneflow/core/ep/cuda/primitive/type_seq.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/cuda/primitive/binary_functor.cuh"

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

template<typename T, int N>
struct GetPackType {
  using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  OF_DEVICE_FUNC Pack() {
    // do nothing
  }
  PackType<T, N> storage;
  T elem[N];
};

template<size_t max_dims, typename IndexType>
struct BroadcastElementwiseBinaryParams {
  NdIndexOffsetHelper<IndexType, max_dims> src0_index_helper;
  NdIndexOffsetHelper<IndexType, max_dims> src1_index_helper;
  NdIndexOffsetHelper<IndexType, max_dims> dst_index_helper;
  size_t num_dims;
  IndexType src0_dims[max_dims];
  IndexType src1_dims[max_dims];
  IndexType count{};
  const void* src0{};
  const void* src1{};
  void* dst{};
};

template<BinaryOp binary_op, typename Src, typename Dst, size_t max_dims, size_t src0_pack_size,
         size_t src1_pack_size, typename IndexType>
__global__ void BroadcastElementwiseBinaryGpu(
    BroadcastElementwiseBinaryParams<max_dims, IndexType> params) {
  constexpr size_t dst_pack_size =
      src0_pack_size > src1_pack_size ? src0_pack_size : src1_pack_size;
  static_assert(src0_pack_size == dst_pack_size || src0_pack_size == 1, "");
  static_assert(src1_pack_size == dst_pack_size || src1_pack_size == 1, "");

  const PackType<Src, src0_pack_size>* src0 =
      reinterpret_cast<const PackType<Src, src0_pack_size>*>(params.src0);
  const PackType<Src, src1_pack_size>* src1 =
      reinterpret_cast<const PackType<Src, src1_pack_size>*>(params.src1);
  PackType<Dst, dst_pack_size>* dst = reinterpret_cast<PackType<Dst, dst_pack_size>*>(params.dst);

  IndexType src0_index[max_dims];
  IndexType src1_index[max_dims];
  IndexType dst_index[max_dims];
  size_t num_dims = params.num_dims;
  CUDA_1D_KERNEL_LOOP_T(IndexType, offset, params.count) {
    params.dst_index_helper.OffsetToNdIndex(offset, dst_index, num_dims);
    for (int64_t i = 0; i < num_dims; ++i) {
      if (params.src0_dims[i] == 1) {
        src0_index[i] = 0;
      } else {
        src0_index[i] = dst_index[i];
      }
      if (params.src1_dims[i] == 1) {
        src1_index[i] = 0;
      } else {
        src1_index[i] = dst_index[i];
      }
    }
    const IndexType src0_offset = params.src0_index_helper.NdIndexToOffset(src0_index, num_dims);
    const IndexType src1_offset = params.src1_index_helper.NdIndexToOffset(src1_index, num_dims);
    Pack<Src, src0_pack_size> src0_pack;
    src0_pack.storage = src0[src0_offset];
    Pack<Src, src1_pack_size> src1_pack;
    src1_pack.storage = src1[src1_offset];
    Pack<Dst, dst_pack_size> dst_pack;

#pragma unroll
    for (int j = 0; j < dst_pack_size; ++j) {
      const Src src0_val =
          (src0_pack_size == dst_pack_size) ? src0_pack.elem[j] : src0_pack.elem[0];
      const Src src1_val =
          (src1_pack_size == dst_pack_size) ? src1_pack.elem[j] : src1_pack.elem[0];
      dst_pack.elem[j] =
          BinaryFunctor<DeviceType::kCUDA, binary_op, Src, Dst>()(src0_val, src1_val);
    }
    dst[offset] = dst_pack.storage;
  }
}

template<BinaryOp op, typename T, typename R, size_t max_dims, size_t src0_pack_size,
         size_t src1_pack_size, typename IndexType>
void LaunchKernel(Stream* stream, int num_dims, const int64_t* src0_dims, const void* src0,
                  const int64_t* src1_dims, const void* src1, const int64_t* dst_dims, void* dst,
                  size_t count) {
  BroadcastElementwiseBinaryParams<max_dims, IndexType> params;
  IndexType params_dst_dims[max_dims];
  for (size_t i = 0; i < num_dims; ++i) {
    params.src0_dims[i] = src0_dims[i];
    params.src1_dims[i] = src1_dims[i];
    params_dst_dims[i] = dst_dims[i];
  }
  params.src0_index_helper = NdIndexOffsetHelper<IndexType, max_dims>(params.src0_dims, num_dims);
  params.src1_index_helper = NdIndexOffsetHelper<IndexType, max_dims>(params.src1_dims, num_dims);
  params.dst_index_helper = NdIndexOffsetHelper<IndexType, max_dims>(params_dst_dims, num_dims);
  params.num_dims = num_dims;
  params.src0 = src0;
  params.src1 = src1;
  params.dst = dst;
  params.count = static_cast<IndexType>(count);
  auto* cuda_stream = stream->As<CudaStream>();
  BroadcastElementwiseBinaryGpu<op, T, R, max_dims, src0_pack_size, src1_pack_size, IndexType>
      <<<BlocksNum4ThreadsNum(params.count), kCudaThreadsNumPerBlock, 0,
         cuda_stream->cuda_stream()>>>(params);
}

template<BinaryOp op, typename T, typename R, size_t max_dims, size_t src0_pack_size,
         size_t src1_pack_size>
void DispatchIndexType(Stream* stream, size_t num_dims, const int64_t* src0_dims, const void* src0,
                       const int64_t* src1_dims, const void* src1, const int64_t* dst_dims,
                       void* dst) {
  size_t count = 1;
  for (size_t i = 0; i < num_dims; ++i) { count *= dst_dims[i]; }
  if (count < GetMaxVal<int32_t>()) {
    LaunchKernel<op, T, R, max_dims, src0_pack_size, src1_pack_size, int32_t>(
        stream, num_dims, src0_dims, src0, src1_dims, src1, dst_dims, dst, count);
  } else {
    LaunchKernel<op, T, R, max_dims, src0_pack_size, src1_pack_size, int64_t>(
        stream, num_dims, src0_dims, src0, src1_dims, src1, dst_dims, dst, count);
  }
}

template<BinaryOp op, typename T, typename R, size_t max_dims>
void DispatchPackSize(Stream* stream, size_t src0_pack_size, size_t src1_pack_size, size_t num_dims,
                      const int64_t* src0_dims, const void* src0, const int64_t* src1_dims,
                      const void* src1, const int64_t* dst_dims, void* dst) {
  void (*func)(Stream* /*stream*/, size_t /*num_dims*/, const int64_t* /*src0_dims*/,
               const void* /*src0*/, const int64_t* /*src1_dims*/, const void* /*src1*/,
               const int64_t* /*dst_dims*/, void* /*dst*/) = nullptr;
  if (src0_pack_size == 1 && src1_pack_size == 1) {
    func = DispatchIndexType<op, T, R, max_dims, 1, 1>;
  } else if (src0_pack_size == 4 && src1_pack_size == 4) {
    func = DispatchIndexType<op, T, R, max_dims, 4, 4>;
  } else if (src0_pack_size == 1 && src1_pack_size == 4) {
    func = DispatchIndexType<op, T, R, max_dims, 1, 4>;
  } else if (src0_pack_size == 4 && src1_pack_size == 1) {
    func = DispatchIndexType<op, T, R, max_dims, 4, 1>;
  } else {
    UNIMPLEMENTED();
  }
  func(stream, num_dims, src0_dims, src0, src1_dims, src1, dst_dims, dst);
}

template<BinaryOp op, typename T, typename R>
void LaunchWithSimplified(Stream* stream, size_t src0_pack_size, size_t src1_pack_size,
                          size_t num_dims, const int64_t* src0_dims, const void* src0,
                          const int64_t* src1_dims, const void* src1, const int64_t* dst_dims,
                          void* dst) {
  void (*func)(Stream* /*stream*/, size_t /*src0_pack_size*/, size_t /*src1_pack_size*/,
               size_t /*num_dims*/, const int64_t* /*src0_dims*/, const void* /*src0*/,
               const int64_t* /*src1_dims*/, const void* /*src1*/, const int64_t* /*dst_dims*/,
               void* /*dst*/) = nullptr;
  CHECK_NE(num_dims, 1);
  if (num_dims == 2) {
    func = DispatchPackSize<op, T, R, 2>;
  } else if (num_dims == 3) {
    func = DispatchPackSize<op, T, R, 3>;
  } else if (num_dims == 4) {
    func = DispatchPackSize<op, T, R, 4>;
  } else if (num_dims <= 8) {
    func = DispatchPackSize<op, T, R, 8>;
  } else {
    UNIMPLEMENTED();
  }
  func(stream, src0_pack_size, src1_pack_size, num_dims, src0_dims, src0, src1_dims, src1, dst_dims,
       dst);
}

template<size_t max_pack_size, typename T, typename R>
size_t GetPackSize(size_t num_src_dims, const int64_t* src0_dims, const void* src0,
                   const int64_t* src1_dims, const void* src1, void* dst) {
  static_assert(max_pack_size > 0 && (max_pack_size & (max_pack_size - 1)) == 0, "");
  auto src0_ptr = reinterpret_cast<std::uintptr_t>(src0);
  auto src1_ptr = reinterpret_cast<std::uintptr_t>(src1);
  auto dst_ptr = reinterpret_cast<std::uintptr_t>(dst);
  const auto is_pack_size_supported = [&](const size_t pack_size, const int64_t* src_dims,
                                          std::uintptr_t src_ptr) -> bool {
    if (src_dims[num_src_dims - 1] == 1) { return true; }
    if ((src_dims[num_src_dims - 1] % pack_size == 0) && (src_ptr % (pack_size * sizeof(T)) == 0)) {
      return true;
    }
    return false;
  };
  for (size_t pack_size = max_pack_size; pack_size > 2; pack_size /= 2) {
    if (is_pack_size_supported(pack_size, src0_dims, src0_ptr)
        && is_pack_size_supported(pack_size, src1_dims, src1_ptr)
        && (dst_ptr % (pack_size * sizeof(R))) == 0) {
      return pack_size;
    }
  }
  return 1;
}

constexpr size_t kMaxPackSize = 4;

template<BinaryOp op, typename T, typename R>
void SimplifyThenLaunch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims,
                        const void* src0, size_t num_src1_dims, const int64_t* src1_dims,
                        const void* src1, void* dst) {
  CHECK_LE(num_src0_dims, kMaxNumDims);
  CHECK_LE(num_src1_dims, kMaxNumDims);
  size_t simplified_num_dims = 0;
  int64_t simplified_src0_dims[kMaxNumDims];
  int64_t simplified_src1_dims[kMaxNumDims];
  SimplifyDims(num_src0_dims, src0_dims, num_src1_dims, src1_dims, &simplified_num_dims,
               simplified_src0_dims, simplified_src1_dims);
  size_t pack_size = GetPackSize<kMaxPackSize, T, R>(simplified_num_dims, simplified_src0_dims,
                                                     src0, simplified_src1_dims, src1, dst);
  int64_t simplified_dst_dims[kMaxNumDims];
  for (int64_t i = 0; i < simplified_num_dims; ++i) {
    simplified_dst_dims[i] = std::max(simplified_src0_dims[i], simplified_src1_dims[i]);
    // inplace check
    if (src0 == dst) { CHECK_EQ(simplified_src0_dims[i], simplified_dst_dims[i]); }
    if (src1 == dst) { CHECK_EQ(simplified_src1_dims[i], simplified_dst_dims[i]); }
  }
  size_t src0_pack_size = 1;
  size_t src1_pack_size = 1;
  if (simplified_src0_dims[simplified_num_dims - 1] != 1) {
    simplified_src0_dims[simplified_num_dims - 1] /= pack_size;
    src0_pack_size = pack_size;
  }
  if (simplified_src1_dims[simplified_num_dims - 1] != 1) {
    simplified_src1_dims[simplified_num_dims - 1] /= pack_size;
    src1_pack_size = pack_size;
  }
  simplified_dst_dims[simplified_num_dims - 1] /= pack_size;
  LaunchWithSimplified<op, T, R>(stream, src0_pack_size, src1_pack_size, simplified_num_dims,
                                 simplified_src0_dims, src0, simplified_src1_dims, src1,
                                 simplified_dst_dims, dst);
}

template<BinaryOp binary_op, typename Src, typename Dst>
struct BinaryLhsScalarFunctor {
  __host__ __device__ explicit BinaryLhsScalarFunctor(Src scalar) : scalar(scalar) {}
  __device__ Dst operator()(Src src) const {
    return BinaryFunctor<DeviceType::kCUDA, binary_op, Src, Dst>()(scalar, src);
  }
  const Src scalar;
};

template<BinaryOp binary_op, typename Src, typename Dst>
struct BinaryRhsScalarFunctor {
  __host__ __device__ explicit BinaryRhsScalarFunctor(Src scalar) : scalar(scalar) {}
  __device__ Dst operator()(Src src) const {
    return BinaryFunctor<DeviceType::kCUDA, binary_op, Src, Dst>()(src, scalar);
  }
  const Src scalar;
};

template<BinaryOp binary_op, typename Src, typename Dst>
struct BinaryLhsScalarPtrFunctorFactory {
  __host__ __device__ explicit BinaryLhsScalarPtrFunctorFactory(const Src* scalar_ptr)
      : scalar_ptr(scalar_ptr) {}
  __device__ BinaryLhsScalarFunctor<binary_op, Src, Dst> operator()() const {
    return BinaryLhsScalarFunctor<binary_op, Src, Dst>(*scalar_ptr);
  }
  const Src* scalar_ptr;
};

template<BinaryOp binary_op, typename Src, typename Dst>
struct BinaryRhsScalarPtrFunctorFactory {
  __host__ __device__ explicit BinaryRhsScalarPtrFunctorFactory(const Src* scalar_ptr)
      : scalar_ptr(scalar_ptr) {}
  __device__ BinaryRhsScalarFunctor<binary_op, Src, Dst> operator()() const {
    return BinaryRhsScalarFunctor<binary_op, Src, Dst>(*scalar_ptr);
  }
  const Src* scalar_ptr;
};

inline bool IsDimsEquals(size_t num_src0_dims, const int64_t* src0_dims, size_t num_src1_dims,
                         const int64_t* src1_dims) {
  if (num_src0_dims != num_src1_dims) { return false; }
  for (size_t i = 0; i < num_src1_dims; ++i) {
    if (src0_dims[i] != src1_dims[i]) { return false; }
  }
  return true;
}

template<BinaryOp binary_op, typename Src, typename Dst>
void DispatchLaunch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const Src* src0,
                    size_t num_src1_dims, const int64_t* src1_dims, const Src* src1, Dst* dst) {
  auto* cuda_stream = stream->As<CudaStream>();
  size_t src0_count = GetElementCount(num_src0_dims, src0_dims);
  size_t src1_count = GetElementCount(num_src1_dims, src1_dims);
  const size_t elem_cnt = std::max(src0_count, src1_count);
  if (IsDimsEquals(num_src0_dims, src0_dims, num_src1_dims, src1_dims)) {
    OF_CUDA_CHECK(
        (cuda::elementwise::Binary(BinaryFunctor<DeviceType::kCUDA, binary_op, Src, Dst>(),
                                   elem_cnt, dst, src0, src1, cuda_stream->cuda_stream())));
  } else if (src0_count == 1) {
    OF_CUDA_CHECK((cuda::elementwise::UnaryWithFactory(
        BinaryLhsScalarPtrFunctorFactory<binary_op, Src, Dst>(src0), elem_cnt, dst, src1,
        cuda_stream->cuda_stream())));
  } else if (src1_count == 1) {
    OF_CUDA_CHECK((cuda::elementwise::UnaryWithFactory(
        BinaryRhsScalarPtrFunctorFactory<binary_op, Src, Dst>(src1), elem_cnt, dst, src0,
        cuda_stream->cuda_stream())));
  } else {
    SimplifyThenLaunch<binary_op, Src, Dst>(stream, num_src0_dims, src0_dims, src0, num_src1_dims,
                                            src1_dims, src1, dst);
  }
}

template<typename T>
T GetValue(Scalar value) {
  return value.Value<T>();
}

template<>
half GetValue<half>(Scalar value) {
  return static_cast<half>(GetValue<float>(value));
}

#if CUDA_VERSION >= 11000

template<>
nv_bfloat16 GetValue<nv_bfloat16>(Scalar value) {
  return static_cast<nv_bfloat16>(GetValue<float>(value));
}

#endif  // CUDA_VERSION >= 11000

template<BinaryOp binary_op, typename Src, typename Dst>
class BroadcastElementwiseBinaryImpl : public BroadcastElementwiseBinary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseBinaryImpl);
  BroadcastElementwiseBinaryImpl() = default;
  ~BroadcastElementwiseBinaryImpl() override = default;

  void Launch(Stream* stream, Scalar src0, size_t num_src1_dims, const int64_t* src1_dims,
              const void* src1, void* dst) override {
    auto* cuda_stream = stream->As<CudaStream>();
    const size_t elem_cnt = GetElementCount(num_src1_dims, src1_dims);
    OF_CUDA_CHECK(
        (cuda::elementwise::Unary(BinaryLhsScalarFunctor<binary_op, Src, Dst>(GetValue<Src>(src0)),
                                  elem_cnt, reinterpret_cast<Dst*>(dst),
                                  reinterpret_cast<const Src*>(src1), cuda_stream->cuda_stream())));
  }
  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              Scalar src1, void* dst) override {
    auto* cuda_stream = stream->As<CudaStream>();
    const size_t elem_cnt = GetElementCount(num_src0_dims, src0_dims);
    OF_CUDA_CHECK(
        (cuda::elementwise::Unary(BinaryRhsScalarFunctor<binary_op, Src, Dst>(GetValue<Src>(src1)),
                                  elem_cnt, reinterpret_cast<Dst*>(dst),
                                  reinterpret_cast<const Src*>(src0), cuda_stream->cuda_stream())));
  }
  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              size_t num_src1_dims, const int64_t* src1_dims, const void* src1,
              void* dst) override {
    DispatchLaunch<binary_op, Src, Dst>(
        stream, num_src0_dims, src0_dims, reinterpret_cast<const Src*>(src0), num_src1_dims,
        src1_dims, reinterpret_cast<const Src*>(src1), reinterpret_cast<Dst*>(dst));
  }
};

}  // namespace

template<BinaryOp binary_op, typename Src, typename Dst>
std::unique_ptr<BroadcastElementwiseBinary> NewBroadcastElementwiseBinary() {
  return std::unique_ptr<BroadcastElementwiseBinary>(
      new BroadcastElementwiseBinaryImpl<binary_op, Src, Dst>());
}

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
