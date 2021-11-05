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
#include "oneflow/core/primitive/include/broadcast_elementwise_binary.h"
#include "oneflow/core/primitive/common/broadcast_elementwise_binary.h"
#include "oneflow/core/primitive/cuda/type_seq.h"
#include "oneflow/core/stream/cuda_stream_context.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/primitive/common/binary_functor.h"

namespace oneflow {

namespace primitive {

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kGPU, BinaryOp::kFloorDiv, Src, Dst> {
  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const {
    return static_cast<Dst>(floor(static_cast<float>(src0) / static_cast<float>(src1)));
  }
};

template<>
struct BinaryFunctor<DeviceType::kGPU, BinaryOp::kFloorDiv, double, double> {
  OF_DEVICE_FUNC double operator()(double src0, double src1) const { return floor(src0 / src1); }
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kGPU, BinaryOp::kPow, Src, Dst> {
  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return pow(src0, src1); }
};

template<>
struct BinaryFunctor<DeviceType::kGPU, BinaryOp::kPow, half, half> {
  OF_DEVICE_FUNC half operator()(half src0, half src1) const {
    return pow(static_cast<float>(src0), static_cast<float>(src1));
  }
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kGPU, BinaryOp::kFmod, Src, Dst> {
  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return fmod(src0, src1); }
};

template<>
struct BinaryFunctor<DeviceType::kGPU, BinaryOp::kFmod, half, half> {
  OF_DEVICE_FUNC half operator()(half src0, half src1) const {
    return fmod(static_cast<float>(src0), static_cast<float>(src1));
  }
};

namespace {

template<BinaryOp binary_op, typename Src, typename Dst, size_t num_dims, size_t pack_size,
         typename IndexType>
__global__ void BroadcastElementwiseBinaryGpu(
    BroadcastElementwiseBinaryParams<num_dims, IndexType> params) {
  const PackType<Src, pack_size>* src0 =
      reinterpret_cast<const PackType<Src, pack_size>*>(params.src0);
  const PackType<Src, pack_size>* src1 =
      reinterpret_cast<const PackType<Src, pack_size>*>(params.src1);
  PackType<Dst, pack_size>* dst = reinterpret_cast<PackType<Dst, pack_size>*>(params.dst);
  IndexType src0_index[num_dims];
  IndexType src1_index[num_dims];
  IndexType dst_index[num_dims];
  CUDA_1D_KERNEL_LOOP_T(IndexType, offset, params.count) {
    params.dst_index_helper.OffsetToNdIndex(offset, dst_index);
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
    const IndexType src0_offset = params.src0_index_helper.NdIndexToOffset(src0_index);
    const IndexType src1_offset = params.src1_index_helper.NdIndexToOffset(src1_index);
    Pack<Src, pack_size> src0_pack;
    src0_pack.storage = src0[src0_offset];
    Pack<Src, pack_size> src1_pack;
    src1_pack.storage = src1[src1_offset];
    Pack<Dst, pack_size> dst_pack;
#pragma unroll
    for (int j = 0; j < pack_size; ++j) {
      dst_pack.elem[j] = BinaryFunctor<DeviceType::kGPU, binary_op, Src, Dst>()(src0_pack.elem[j],
                                                                                src1_pack.elem[j]);
    }
    dst[offset] = dst_pack.storage;
  }
}

template<BinaryOp op, typename Src, typename Dst, size_t num_dims, size_t pack_size,
         typename IndexType>
void LaunchKernel(StreamContext* stream_ctx,
                  BroadcastElementwiseBinaryParams<num_dims, IndexType> params) {
  cudaStream_t cuda_stream =
      CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();
  BroadcastElementwiseBinaryGpu<op, Src, Dst, num_dims, pack_size, IndexType>
      <<<BlocksNum4ThreadsNum(params.count), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(params);
}

template<BinaryOp binary_op, typename Src, typename Dst, bool scalar_left>
struct UnaryByScalarFunctor {
  __host__ __device__ explicit UnaryByScalarFunctor(Src scalar) : scalar(scalar) {}
  __device__ Dst operator()(Src a) const {
    if (scalar_left) {
      return BinaryFunctor<DeviceType::kGPU, binary_op, Src, Dst>()(scalar, a);
    } else {
      return BinaryFunctor<DeviceType::kGPU, binary_op, Src, Dst>()(a, scalar);
    }
  }
  const Src scalar;
};

template<BinaryOp binary_op, typename Src, typename Dst, bool scalar_left>
struct UnaryByScalarPtrFunctorFactory {
  __host__ __device__ explicit UnaryByScalarPtrFunctorFactory(const Src* scalar_ptr)
      : scalar_ptr(scalar_ptr) {}
  __device__ UnaryByScalarFunctor<binary_op, Src, Dst, scalar_left> operator()() const {
    return UnaryByScalarFunctor<binary_op, Src, Dst, scalar_left>(*scalar_ptr);
  }
  const Src* scalar_ptr;
};

bool IsDimsEquals(size_t num_src0_dims, const int64_t* src0_dims, size_t num_src1_dims,
                  const int64_t* src1_dims) {
  if (num_src0_dims != num_src1_dims) { return false; }
  for (size_t i = 0; i < num_src1_dims; ++i) {
    if (src0_dims[i] != src1_dims[i]) { return false; }
  }
  return true;
}

size_t GetElementCount(size_t num_dims, const int64_t* dims) {
  size_t count = 1;
  for (size_t i = 0; i < num_dims; ++i) { count *= dims[i]; }
  return count;
}

template<BinaryOp binary_op, typename Src, typename Dst>
void DispatchLaunch(StreamContext* stream_ctx, size_t num_src0_dims, const int64_t* src0_dims,
                    const Src* src0, size_t num_src1_dims, const int64_t* src1_dims,
                    const Src* src1, Dst* dst) {
  cudaStream_t cuda_stream =
      CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();
  size_t src0_count = GetElementCount(num_src0_dims, src0_dims);
  size_t src1_count = GetElementCount(num_src1_dims, src1_dims);
  const size_t elem_cnt = std::max(src0_count, src1_count);
  if (IsDimsEquals(num_src0_dims, src0_dims, num_src1_dims, src1_dims)) {
    LOG(ERROR) << "elementwise";
    OF_CUDA_CHECK((cuda::elementwise::Binary(BinaryFunctor<DeviceType::kGPU, binary_op, Src, Dst>(),
                                             elem_cnt, dst, src0, src1, cuda_stream)));
  } else if (src0_count == 1) {
    LOG(ERROR) << "UnaryWithFactory left scalar ptr";
    OF_CUDA_CHECK((cuda::elementwise::UnaryWithFactory(
        UnaryByScalarPtrFunctorFactory<binary_op, Src, Dst, true>(src0), elem_cnt, dst, src1,
        cuda_stream)));
  } else if (src1_count == 1) {
    LOG(ERROR) << "UnaryWithFactory right scalar ptr";
    OF_CUDA_CHECK((cuda::elementwise::UnaryWithFactory(
        UnaryByScalarPtrFunctorFactory<binary_op, Src, Dst, false>(src1), elem_cnt, dst, src0,
        cuda_stream)));
  } else {
    LOG(ERROR) << "SimplifyThenLaunch";
    SimplifyThenLaunch<binary_op, Src, Dst>(stream_ctx, num_src0_dims, src0_dims, src0,
                                            num_src1_dims, src1_dims, src1, dst);
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

  void Launch(StreamContext* stream_ctx, Scalar src0, size_t num_src1_dims,
              const int64_t* src1_dims, const void* src1, void* dst) override {
    LOG(ERROR) << "UnaryWithFactory left scalar";
    cudaStream_t cuda_stream =
        CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();
    const size_t elem_cnt = GetElementCount(num_src1_dims, src1_dims);
    OF_CUDA_CHECK((cuda::elementwise::Unary(
        UnaryByScalarFunctor<binary_op, Src, Dst, true>(GetValue<Src>(src0)), elem_cnt,
        reinterpret_cast<Dst*>(dst), reinterpret_cast<const Src*>(src1), cuda_stream)));
  }
  void Launch(StreamContext* stream_ctx, size_t num_src0_dims, const int64_t* src0_dims,
              const void* src0, Scalar src1, void* dst) override {
    LOG(ERROR) << "UnaryWithFactory right scalar";
    cudaStream_t cuda_stream =
        CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();
    const size_t elem_cnt = GetElementCount(num_src0_dims, src0_dims);
    OF_CUDA_CHECK((cuda::elementwise::Unary(
        UnaryByScalarFunctor<binary_op, Src, Dst, false>(GetValue<Src>(src1)), elem_cnt,
        reinterpret_cast<Dst*>(dst), reinterpret_cast<const Src*>(src0), cuda_stream)));
  }
  void Launch(StreamContext* stream_ctx, size_t num_src0_dims, const int64_t* src0_dims,
              const void* src0, size_t num_src1_dims, const int64_t* src1_dims, const void* src1,
              void* dst) override {
    DispatchLaunch<binary_op, Src, Dst>(
        stream_ctx, num_src0_dims, src0_dims, reinterpret_cast<const Src*>(src0), num_src1_dims,
        src1_dims, reinterpret_cast<const Src*>(src1), reinterpret_cast<Dst*>(dst));
  }
};

template<BinaryOp binary_op, typename Src, typename Dst>
std::unique_ptr<BroadcastElementwiseBinary> NewBroadcastElementwiseBinary() {
  return std::unique_ptr<BroadcastElementwiseBinary>(
      new BroadcastElementwiseBinaryImpl<binary_op, Src, Dst>());
}

class BroadcastElementwiseBinaryFactoryImpl : public BroadcastElementwiseBinaryFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseBinaryFactoryImpl);
  BroadcastElementwiseBinaryFactoryImpl() = default;
  ~BroadcastElementwiseBinaryFactoryImpl() override = default;

  std::unique_ptr<BroadcastElementwiseBinary> New(BinaryOp binary_op, DataType src_type,
                                                  DataType dst_type, size_t max_num_dims) override {
    if (max_num_dims > kMaxNumDims) { return nullptr; }
#define MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_SAME_DTYPE_ENTRY(binary_op, data_type_pair) \
  {std::make_tuple(binary_op, OF_PP_PAIR_SECOND(data_type_pair),                          \
                   OF_PP_PAIR_SECOND(data_type_pair)),                                    \
   NewBroadcastElementwiseBinary<binary_op, OF_PP_PAIR_FIRST(data_type_pair),             \
                                 OF_PP_PAIR_FIRST(data_type_pair)>},

#define MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_DIFFERENT_DTYPE_ENTRY(binary_op, src_data_type_pair, \
                                                                    dst_data_type_pair)            \
  {std::make_tuple(binary_op, OF_PP_PAIR_SECOND(src_data_type_pair),                               \
                   OF_PP_PAIR_SECOND(dst_data_type_pair)),                                         \
   NewBroadcastElementwiseBinary<binary_op, OF_PP_PAIR_FIRST(src_data_type_pair),                  \
                                 OF_PP_PAIR_FIRST(dst_data_type_pair)>},

    static const std::map<std::tuple<BinaryOp, DataType, DataType>,
                          std::function<std::unique_ptr<BroadcastElementwiseBinary>()>>
        new_broadcast_elementwise_binary_handle{
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_SAME_DTYPE_ENTRY,
                                             BINARY_MATH_OP_SEQ, CUDA_PRIMITIVE_ALL_TYPE_SEQ)
                OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                    MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_DIFFERENT_DTYPE_ENTRY,
                    BINARY_LOGICAL_OP_SEQ, CUDA_PRIMITIVE_ALL_TYPE_SEQ,
                    CUDA_PRIMITIVE_INT8_TYPE_SEQ)};

#undef MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_DIFFERENT_DTYPE_ENTRY
#undef MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_SAME_DTYPE_ENTRY

    const auto it = new_broadcast_elementwise_binary_handle.find(
        std::make_tuple(binary_op, src_type, dst_type));
    if (it != new_broadcast_elementwise_binary_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, BroadcastElementwiseBinaryFactory,
                           BroadcastElementwiseBinaryFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow
