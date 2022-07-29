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
namespace broadcast_elementwise_binary {

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
  IndexType src0_index_mask[max_dims];
  IndexType src1_index_mask[max_dims];
  IndexType count{};
  const void* src0{};
  const void* src1{};
  void* dst{};
  Scalar attr0;
  Scalar attr1;
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
#pragma unroll
    for (int i = 0; i < max_dims; ++i) {
      if (i < num_dims) {
        src0_index[i] = params.src0_index_mask[i] * dst_index[i];
        src1_index[i] = params.src1_index_mask[i] * dst_index[i];
      } else {
        src0_index[i] = 0;
        src1_index[i] = 0;
      }
    }
    const IndexType src0_offset = params.src0_index_helper.NdIndexToOffset(src0_index, num_dims);
    const IndexType src1_offset = params.src1_index_helper.NdIndexToOffset(src1_index, num_dims);
    Pack<Src, src0_pack_size> src0_pack;
    src0_pack.storage = src0[src0_offset];
    Pack<Src, src1_pack_size> src1_pack;
    src1_pack.storage = src1[src1_offset];
    Pack<Dst, dst_pack_size> dst_pack;
    BinaryFunctor<DeviceType::kCUDA, binary_op, Src, Dst> functor(params.attr0, params.attr1);
#pragma unroll
    for (int j = 0; j < dst_pack_size; ++j) {
      const Src src0_val =
          (src0_pack_size == dst_pack_size) ? src0_pack.elem[j] : src0_pack.elem[0];
      const Src src1_val =
          (src1_pack_size == dst_pack_size) ? src1_pack.elem[j] : src1_pack.elem[0];
      dst_pack.elem[j] = functor(src0_val, src1_val);
    }
    dst[offset] = dst_pack.storage;
  }
}

template<BinaryOp op, typename T, typename R, size_t max_dims, size_t src0_pack_size,
         size_t src1_pack_size, typename IndexType>
void LaunchKernel(Stream* stream, int num_dims, const int64_t* src0_dims, const void* src0,
                  const int64_t* src1_dims, const void* src1, const int64_t* dst_dims, void* dst,
                  size_t count, Scalar attr0, Scalar attr1) {
  BroadcastElementwiseBinaryParams<max_dims, IndexType> params;
  for (size_t i = 0; i < num_dims; ++i) {
    params.src0_index_mask[i] = (src0_dims[i] == 1) ? 0 : 1;
    params.src1_index_mask[i] = (src1_dims[i] == 1) ? 0 : 1;
  }
  params.src0_index_helper = NdIndexOffsetHelper<IndexType, max_dims>(src0_dims, num_dims);
  params.src1_index_helper = NdIndexOffsetHelper<IndexType, max_dims>(src1_dims, num_dims);
  params.dst_index_helper = NdIndexOffsetHelper<IndexType, max_dims>(dst_dims, num_dims);
  params.num_dims = num_dims;
  params.src0 = src0;
  params.src1 = src1;
  params.dst = dst;
  params.count = static_cast<IndexType>(count);
  params.attr0 = attr0;
  params.attr1 = attr1;
  auto* cuda_stream = stream->As<CudaStream>();
  BroadcastElementwiseBinaryGpu<op, T, R, max_dims, src0_pack_size, src1_pack_size, IndexType>
      <<<BlocksNum4ThreadsNum(params.count), kCudaThreadsNumPerBlock, 0,
         cuda_stream->cuda_stream()>>>(params);
}

template<BinaryOp op, typename T, typename R, size_t max_dims, size_t src0_pack_size,
         size_t src1_pack_size>
void DispatchIndexType(Stream* stream, size_t num_dims, const int64_t* src0_dims, const void* src0,
                       const int64_t* src1_dims, const void* src1, const int64_t* dst_dims,
                       void* dst, Scalar attr0, Scalar attr1) {
  size_t count = GetElementCount(num_dims, dst_dims);
  if (count < GetMaxVal<int32_t>()) {
    LaunchKernel<op, T, R, max_dims, src0_pack_size, src1_pack_size, int32_t>(
        stream, num_dims, src0_dims, src0, src1_dims, src1, dst_dims, dst, count, attr0, attr1);
  } else {
    LaunchKernel<op, T, R, max_dims, src0_pack_size, src1_pack_size, int64_t>(
        stream, num_dims, src0_dims, src0, src1_dims, src1, dst_dims, dst, count, attr0, attr1);
  }
}

template<BinaryOp op, typename T, typename R, size_t max_dims>
void DispatchPackSize(Stream* stream, size_t src0_pack_size, size_t src1_pack_size, size_t num_dims,
                      const int64_t* src0_dims, const void* src0, const int64_t* src1_dims,
                      const void* src1, const int64_t* dst_dims, void* dst, Scalar attr0,
                      Scalar attr1) {
  void (*func)(Stream* /*stream*/, size_t /*num_dims*/, const int64_t* /*src0_dims*/,
               const void* /*src0*/, const int64_t* /*src1_dims*/, const void* /*src1*/,
               const int64_t* /*dst_dims*/, void* /*dst*/, Scalar /*attr0*/, Scalar /*attr1*/) =
      nullptr;
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
  func(stream, num_dims, src0_dims, src0, src1_dims, src1, dst_dims, dst, attr0, attr1);
}

template<BinaryOp op, typename T, typename R>
void DispatchNumDims(Stream* stream, size_t src0_pack_size, size_t src1_pack_size, size_t num_dims,
                     const int64_t* src0_dims, const void* src0, const int64_t* src1_dims,
                     const void* src1, const int64_t* dst_dims, void* dst, Scalar attr0,
                     Scalar attr1) {
  void (*func)(Stream* /*stream*/, size_t /*src0_pack_size*/, size_t /*src1_pack_size*/,
               size_t /*num_dims*/, const int64_t* /*src0_dims*/, const void* /*src0*/,
               const int64_t* /*src1_dims*/, const void* /*src1*/, const int64_t* /*dst_dims*/,
               void* /*dst*/, Scalar /*attr0*/, Scalar /*attr1*/) = nullptr;
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
       dst, attr0, attr1);
}

template<size_t max_pack_size, typename T, typename R>
size_t GetPackSize(size_t num_src_dims, const int64_t* src0_dims, const void* src0,
                   const int64_t* src1_dims, const void* src1, void* dst) {
  static_assert(max_pack_size > 0 && (max_pack_size & (max_pack_size - 1)) == 0, "");
  CHECK(src0_dims[num_src_dims - 1] != 1 || src1_dims[num_src_dims - 1] != 1);
  auto dst_ptr = reinterpret_cast<std::uintptr_t>(dst);
  for (size_t pack_size = max_pack_size; pack_size > 2; pack_size /= 2) {
    bool is_src0_supported = (src0_dims[num_src_dims - 1] == 1)
                             || IsPackSizeSupported<T>(pack_size, num_src_dims, src0_dims, src0);
    bool is_src1_supported = (src1_dims[num_src_dims - 1] == 1)
                             || IsPackSizeSupported<T>(pack_size, num_src_dims, src1_dims, src1);
    if (is_src0_supported && is_src1_supported && (dst_ptr % (pack_size * sizeof(R))) == 0) {
      return pack_size;
    }
  }
  return 1;
}

constexpr size_t kMaxPackSize = 4;

template<BinaryOp op, typename T, typename R>
void LaunchWithSimplified(Stream* stream, size_t simplified_num_dims, int64_t* simplified_src0_dims,
                          const void* src0, int64_t* simplified_src1_dims, const void* src1,
                          int64_t* simplified_dst_dims, void* dst, Scalar attr0, Scalar attr1) {
  CHECK_LE(simplified_num_dims, kMaxNumDims);
  size_t pack_size = GetPackSize<kMaxPackSize, T, R>(simplified_num_dims, simplified_src0_dims,
                                                     src0, simplified_src1_dims, src1, dst);
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
  DispatchNumDims<op, T, R>(stream, src0_pack_size, src1_pack_size, simplified_num_dims,
                            simplified_src0_dims, src0, simplified_src1_dims, src1,
                            simplified_dst_dims, dst, attr0, attr1);
}

template<BinaryOp binary_op, typename Src, typename Dst>
struct BinaryLhsScalarFunctor {
  __host__ __device__ BinaryLhsScalarFunctor(Src scalar, Scalar attr0, Scalar attr1)
      : scalar(scalar), functor(attr0, attr1) {}
  __device__ Dst operator()(Src src) const { return functor(scalar, src); }
  const Src scalar;
  BinaryFunctor<DeviceType::kCUDA, binary_op, Src, Dst> functor;
};

template<BinaryOp binary_op, typename Src, typename Dst>
struct BinaryRhsScalarFunctor {
  __host__ __device__ BinaryRhsScalarFunctor(Src scalar, Scalar attr0, Scalar attr1)
      : scalar(scalar), functor(attr0, attr1) {}
  __device__ Dst operator()(Src src) const { return functor(src, scalar); }
  const Src scalar;
  BinaryFunctor<DeviceType::kCUDA, binary_op, Src, Dst> functor;
};

template<BinaryOp binary_op, typename Src, typename Dst>
struct BinaryLhsScalarPtrFunctorFactory {
  __host__ __device__ BinaryLhsScalarPtrFunctorFactory(const Src* scalar_ptr, Scalar attr0,
                                                       Scalar attr1)
      : scalar_ptr(scalar_ptr), attr0(attr0), attr1(attr1) {}
  __device__ BinaryLhsScalarFunctor<binary_op, Src, Dst> operator()() const {
    return BinaryLhsScalarFunctor<binary_op, Src, Dst>(*scalar_ptr, attr0, attr1);
  }
  const Src* scalar_ptr;
  Scalar attr0, attr1;
};

template<BinaryOp binary_op, typename Src, typename Dst>
struct BinaryRhsScalarPtrFunctorFactory {
  __host__ __device__ explicit BinaryRhsScalarPtrFunctorFactory(const Src* scalar_ptr, Scalar attr0,
                                                                Scalar attr1)
      : scalar_ptr(scalar_ptr), attr0(attr0), attr1(attr1) {}
  __device__ BinaryRhsScalarFunctor<binary_op, Src, Dst> operator()() const {
    return BinaryRhsScalarFunctor<binary_op, Src, Dst>(*scalar_ptr, attr0, attr1);
  }
  const Src* scalar_ptr;
  Scalar attr0, attr1;
};

template<BinaryOp binary_op, typename Src, typename Dst>
void DispatchLaunch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const Src* src0,
                    size_t num_src1_dims, const int64_t* src1_dims, const Src* src1, Dst* dst,
                    Scalar attr0, Scalar attr1) {
  auto* cuda_stream = stream->As<CudaStream>();
  size_t simplified_num_dims = 0;
  int64_t simplified_src0_dims[kMaxNumDims];
  int64_t simplified_src1_dims[kMaxNumDims];
  int64_t simplified_dst_dims[kMaxNumDims];
  SimplifyBroadcastDims<kMaxNumDims>(num_src0_dims, src0_dims, num_src1_dims, src1_dims,
                                     &simplified_num_dims, simplified_src0_dims,
                                     simplified_src1_dims, simplified_dst_dims);
  CheckInplace(simplified_num_dims, simplified_src0_dims, src0, simplified_dst_dims, dst);
  CheckInplace(simplified_num_dims, simplified_src1_dims, src1, simplified_dst_dims, dst);
  if (IsDimsEquals(simplified_num_dims, simplified_src0_dims, simplified_num_dims,
                   simplified_src1_dims)) {
    const int64_t elem_cnt = GetElementCount(simplified_num_dims, simplified_src0_dims);
    OF_CUDA_CHECK((cuda::elementwise::Binary(
        BinaryFunctor<DeviceType::kCUDA, binary_op, Src, Dst>(attr0, attr1), elem_cnt, dst, src0,
        src1, cuda_stream->cuda_stream())));
  } else {
    if (simplified_num_dims == 1 && simplified_src0_dims[0] == 1) {
      OF_CUDA_CHECK((cuda::elementwise::UnaryWithFactory(
          BinaryLhsScalarPtrFunctorFactory<binary_op, Src, Dst>(src0, attr0, attr1),
          simplified_src1_dims[0], dst, src1, cuda_stream->cuda_stream())));
    } else if (simplified_num_dims == 1 && simplified_src1_dims[0] == 1) {
      OF_CUDA_CHECK((cuda::elementwise::UnaryWithFactory(
          BinaryRhsScalarPtrFunctorFactory<binary_op, Src, Dst>(src1, attr0, attr1),
          simplified_src0_dims[0], dst, src0, cuda_stream->cuda_stream())));
    } else {
      LaunchWithSimplified<binary_op, Src, Dst>(stream, simplified_num_dims, simplified_src0_dims,
                                                src0, simplified_src1_dims, src1,
                                                simplified_dst_dims, dst, attr0, attr1);
    }
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
  BroadcastElementwiseBinaryImpl(Scalar attr0, Scalar attr1) : attr0(attr0), attr1(attr1) {}
  ~BroadcastElementwiseBinaryImpl() override = default;

  void Launch(Stream* stream, Scalar src0, size_t num_src1_dims, const int64_t* src1_dims,
              const void* src1, void* dst) override {
    auto* cuda_stream = stream->As<CudaStream>();
    const size_t elem_cnt = GetElementCount(num_src1_dims, src1_dims);
    OF_CUDA_CHECK((cuda::elementwise::Unary(
        BinaryLhsScalarFunctor<binary_op, Src, Dst>(GetValue<Src>(src0), attr0, attr1), elem_cnt,
        reinterpret_cast<Dst*>(dst), reinterpret_cast<const Src*>(src1),
        cuda_stream->cuda_stream())));
  }
  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              Scalar src1, void* dst) override {
    auto* cuda_stream = stream->As<CudaStream>();
    const size_t elem_cnt = GetElementCount(num_src0_dims, src0_dims);
    OF_CUDA_CHECK((cuda::elementwise::Unary(
        BinaryRhsScalarFunctor<binary_op, Src, Dst>(GetValue<Src>(src1), attr0, attr1), elem_cnt,
        reinterpret_cast<Dst*>(dst), reinterpret_cast<const Src*>(src0),
        cuda_stream->cuda_stream())));
  }
  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              size_t num_src1_dims, const int64_t* src1_dims, const void* src1,
              void* dst) override {
    DispatchLaunch<binary_op, Src, Dst>(
        stream, num_src0_dims, src0_dims, reinterpret_cast<const Src*>(src0), num_src1_dims,
        src1_dims, reinterpret_cast<const Src*>(src1), reinterpret_cast<Dst*>(dst), attr0, attr1);
  }

 private:
  Scalar attr0, attr1;
};

}  // namespace

template<BinaryOp binary_op, typename Src, typename Dst>
std::unique_ptr<BroadcastElementwiseBinary> NewBroadcastElementwiseBinary(Scalar attr0,
                                                                          Scalar attr1) {
  return std::unique_ptr<BroadcastElementwiseBinary>(
      new BroadcastElementwiseBinaryImpl<binary_op, Src, Dst>(attr0, attr1));
}

}  // namespace broadcast_elementwise_binary
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
