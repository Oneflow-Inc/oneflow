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

#include "oneflow/core/ep/common/primitive/broadcast_elementwise_unary.h"
#include "oneflow/core/ep/include/primitive/permute.h"
#include "oneflow/core/ep/cuda/primitive/unary_functor.cuh"
#include "oneflow/core/ep/cuda/primitive/type_seq.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace ep {
namespace primitive {
namespace broadcast_elementwise_unary {

namespace {

#define CUDA_PRIMITIVE_CAST_ALL_TYPE_SEQ \
  CUDA_PRIMITIVE_UINT32_TYPE_SEQ         \
  CUDA_PRIMITIVE_ALL_TYPE_SEQ

constexpr size_t kMaxPackSize = 4;

template<size_t max_pack_size, typename Src, typename Dst>
size_t GetPackSize(size_t num_dims, const int64_t* src_dims, const void* src,
                   const int64_t* dst_dims, const void* dst) {
  static_assert(max_pack_size > 0 && (max_pack_size & (max_pack_size - 1)) == 0, "");
  for (size_t pack_size = max_pack_size; pack_size > 2; pack_size /= 2) {
    bool is_src_supported = IsPackSizeSupported<Src>(pack_size, num_dims, src_dims, src);
    bool is_dst_supported = IsPackSizeSupported<Dst>(pack_size, num_dims, dst_dims, dst);
    if (is_src_supported && is_dst_supported) { return pack_size; }
  }
  return 1;
}

template<typename Src, typename Dst, size_t max_dims, typename IndexType>
struct BroadcastElementwiseUnaryParams {
  OffsetToIndexWithStrideCalculator<IndexType, max_dims> dst_offset_to_index_helper;
  size_t num_dims;
  int64_t src_strides[max_dims];
  int64_t dst_strides[max_dims];
  IndexType src_index_mask[max_dims];
  IndexType count{};
  const Src* src{};
  Dst* dst{};
  bool dst_is_contiguous;
  Scalar attr0;
  Scalar attr1;
};

template<UnaryOp unary_op, typename Src, typename Dst>
struct UnaryScalarFunctor {
  __host__ __device__ explicit UnaryScalarFunctor(Src scalar) : scalar(scalar) {}
  __device__ Dst operator()() const {
    return UnaryFunctor<DeviceType::kCUDA, unary_op, Dst, Src>()(scalar);
  }
  const Src scalar;
};

template<UnaryOp unary_op, typename Src, typename Dst>
struct UnaryScalarPtrFunctorFactory {
  __host__ __device__ explicit UnaryScalarPtrFunctorFactory(const Src* scalar_ptr)
      : scalar_ptr(scalar_ptr) {}
  __device__ UnaryScalarFunctor<unary_op, Src, Dst> operator()() const {
    return UnaryScalarFunctor<unary_op, Src, Dst>(*scalar_ptr);
  }
  const Src* scalar_ptr;
};

template<UnaryOp op, typename Src, typename Dst, size_t max_dims, size_t pack_size,
         typename IndexType>
__global__ void BroadcastElementwiseUnaryGpu(
    BroadcastElementwiseUnaryParams<Src, Dst, max_dims, IndexType> params) {
  using LoadPack = cuda::elementwise::Packed<Src, pack_size>;
  using StorePack = cuda::elementwise::Packed<Dst, pack_size>;
  const LoadPack* src = reinterpret_cast<const LoadPack*>(params.src);
  StorePack* dst = reinterpret_cast<StorePack*>(params.dst);

  size_t num_dims = params.num_dims;
  const int64_t* src_strides = params.src_strides;
  const int64_t* dst_strides = params.dst_strides;
  auto functor = UnaryFunctor<DeviceType::kCUDA, op, Dst, Src>(params.attr0, params.attr1);

  CUDA_1D_KERNEL_LOOP_T(IndexType, offset, params.count) {
    IndexType src_offset = 0;
    IndexType dst_offset = 0;
    IndexType remaining = offset;
#pragma unroll
    for (int i = 0; i < max_dims; ++i) {
      if (i < num_dims - 1) {
        IndexType dst_index = params.dst_offset_to_index_helper.divides(remaining, i);
        remaining = remaining - params.dst_offset_to_index_helper.mul(dst_index, i);
        dst_offset += dst_index * dst_strides[i];
        src_offset += params.src_index_mask[i] * dst_index * src_strides[i];
      } else if (i == num_dims - 1) {
        dst_offset += remaining * dst_strides[i];
        src_offset += params.src_index_mask[i] * remaining * src_strides[i];
      } else {
        break;
      }
    }

    LoadPack src_pack = src[src_offset];
    StorePack dst_pack;
#pragma unroll
    for (int j = 0; j < pack_size; ++j) { dst_pack.elem[j] = functor(src_pack.elem[j]); }
    dst[dst_offset] = dst_pack;
  }
}

template<UnaryOp op, typename Src, typename Dst, size_t max_dims, size_t pack_size,
         typename IndexType>
void LaunchKernel(CudaStream* stream, size_t num_dims, const int64_t* src_dims,
                  const int64_t* src_strides, const Src* src, const int64_t* dst_dims,
                  const int64_t* dst_strides, Dst* dst, bool continuous_output, Scalar attr0,
                  Scalar attr1, size_t count) {
  BroadcastElementwiseUnaryParams<Src, Dst, max_dims, IndexType> params;
  for (size_t i = 0; i < num_dims; ++i) {
    params.src_index_mask[i] = (src_dims[i] == 1) ? 0 : 1;
    params.src_strides[i] = src_strides[i];
    params.dst_strides[i] = dst_strides[i];
  }
  params.dst_offset_to_index_helper =
      OffsetToIndexWithStrideCalculator<IndexType, max_dims>(dst_dims, num_dims);
  params.num_dims = num_dims;
  params.src = src;
  params.dst = dst;
  params.count = static_cast<IndexType>(count);
  params.attr0 = attr0;
  params.attr1 = attr1;
  params.dst_is_contiguous = continuous_output;

  BroadcastElementwiseUnaryGpu<op, Src, Dst, max_dims, pack_size, IndexType>
      <<<BlocksNum4ThreadsNum(params.count), kCudaThreadsNumPerBlock, 0, stream->cuda_stream()>>>(
          params);
}

template<UnaryOp op, typename Src, typename Dst, size_t max_dims, size_t pack_size>
void DispatchIndexType(CudaStream* stream, size_t num_dims, const int64_t* src_dims,
                       const int64_t* src_strides, const Src* src, const int64_t* dst_dims,
                       const int64_t* dst_strides, Dst* dst, bool continuous_output, Scalar attr0,
                       Scalar attr1) {
  size_t count = GetElementCount(num_dims, dst_dims);
  if (count < GetMaxVal<int32_t>() / 2) {
    LaunchKernel<op, Src, Dst, max_dims, pack_size, int32_t>(
        stream, num_dims, src_dims, src_strides, src, dst_dims, dst_strides, dst, continuous_output,
        attr0, attr1, count);
  } else {
    LaunchKernel<op, Src, Dst, max_dims, pack_size, int64_t>(
        stream, num_dims, src_dims, src_strides, src, dst_dims, dst_strides, dst, continuous_output,
        attr0, attr1, count);
  }
}

template<UnaryOp op, typename Src, typename Dst, size_t max_dims>
void DispatchPackSize(CudaStream* stream, size_t pack_size, size_t num_dims,
                      const int64_t* src_dims, const int64_t* src_strides, const Src* src,
                      const int64_t* dst_dims, const int64_t* dst_strides, Dst* dst,
                      bool continuous_output, Scalar attr0, Scalar attr1) {
  void (*func)(CudaStream* /*stream*/, size_t /*num_dims*/, const int64_t* /*src_dims*/,
               const int64_t* /*src_strides*/, const Src* /*src*/, const int64_t* /*dst_dims*/,
               const int64_t* /*dst_strides*/, Dst* /*dst*/, bool /*continuous_output*/,
               Scalar /*attr0*/, Scalar /*attr1*/) = nullptr;
  if (pack_size == 1) {
    func = DispatchIndexType<op, Src, Dst, max_dims, 1>;
  } else if (pack_size == 4) {
    func = DispatchIndexType<op, Src, Dst, max_dims, 4>;
  } else {
    UNIMPLEMENTED();
  }
  func(stream, num_dims, src_dims, src_strides, src, dst_dims, dst_strides, dst, continuous_output,
       attr0, attr1);
}

template<UnaryOp op, typename Src, typename Dst>
void DispatchNumDims(CudaStream* stream, size_t pack_size, size_t num_dims, const int64_t* src_dims,
                     const int64_t* src_strides, const Src* src, const int64_t* dst_dims,
                     const int64_t* dst_strides, Dst* dst, bool continuous_output, Scalar attr0,
                     Scalar attr1) {
  void (*func)(CudaStream* /*stream*/, size_t /*pack_size*/, size_t /*num_dims*/,
               const int64_t* /*src_dims*/, const int64_t* /*src_strides*/, const Src* /*src*/,
               const int64_t* /*dst_dims*/, const int64_t* /*dst_strides*/, Dst* /*dst*/,
               bool /*continuous_output*/, Scalar /*attr0*/, Scalar /*attr1*/) = nullptr;
  if (num_dims == 1) {
    func = DispatchPackSize<op, Src, Dst, 1>;
  } else if (num_dims == 2) {
    func = DispatchPackSize<op, Src, Dst, 2>;
  } else if (num_dims == 3) {
    func = DispatchPackSize<op, Src, Dst, 3>;
  } else if (num_dims == 4) {
    func = DispatchPackSize<op, Src, Dst, 4>;
  } else if (num_dims <= kMaxNumDims) {
    func = DispatchPackSize<op, Src, Dst, kMaxNumDims>;
  } else {
    UNIMPLEMENTED();
  }
  func(stream, pack_size, num_dims, src_dims, src_strides, src, dst_dims, dst_strides, dst,
       continuous_output, attr0, attr1);
}

template<UnaryOp op, typename Src, typename Dst>
void LaunchWithSimplified(CudaStream* stream, size_t simplified_num_dims,
                          int64_t* simplified_src_dims, int64_t* simplified_src_strides,
                          const Src* src, int64_t* simplified_dst_dims,
                          int64_t* simplified_dst_strides, Dst* dst, Scalar attr0, Scalar attr1) {
  CHECK_LE(simplified_num_dims, kMaxNumDims);
  bool src_enable_pack = (simplified_src_strides[simplified_num_dims - 1] == 1);
  bool dst_enable_pack = (simplified_dst_strides[simplified_num_dims - 1] == 1);
  size_t pack_size = 1;
  if (src_enable_pack && dst_enable_pack) {
    pack_size = GetPackSize<kMaxPackSize, Src, Dst>(simplified_num_dims, simplified_src_dims, src,
                                                    simplified_dst_dims, dst);
  }
  bool continuous_output = true;
  for (int i = simplified_num_dims - 1; i >= 0; i--) {
    if ((i == simplified_num_dims - 1 && simplified_dst_strides[i] != 1)
        || (i != simplified_num_dims - 1
            && simplified_dst_strides[i]
                   != simplified_dst_strides[i + 1] * simplified_dst_dims[i + 1])) {
      continuous_output = false;
      break;
    }
  }
  simplified_src_dims[simplified_num_dims - 1] /= pack_size;
  simplified_dst_dims[simplified_num_dims - 1] /= pack_size;
  for (int i = 0; i < simplified_num_dims - 1; i++) {
    simplified_src_strides[i] /= pack_size;
    simplified_dst_strides[i] /= pack_size;
  }
  DispatchNumDims<op, Src, Dst>(stream, pack_size, simplified_num_dims, simplified_src_dims,
                                simplified_src_strides, src, simplified_dst_dims,
                                simplified_dst_strides, dst, continuous_output, attr0, attr1);
}

template<UnaryOp op, typename Src, typename Dst, size_t pack, bool tail>
__global__ void LaunchFillKernel(UnaryFunctor<DeviceType::kCUDA, op, Dst, Src> functor, Dst* dst,
                                 const Src* src, size_t pack_count, size_t count, size_t tail_count,
                                 Dst* tail_dst) {
  using StorePack = cuda::elementwise::Packed<Dst, pack>;
  StorePack pack_value;
  Dst value = functor(*src);
#pragma unroll
  for (size_t i = 0; i < pack; ++i) { pack_value.elem[i] = value; }
  StorePack* pack_dst = reinterpret_cast<StorePack*>(dst);
  CUDA_1D_KERNEL_LOOP_T(size_t, i, pack_count) { pack_dst[i] = pack_value; }
  if (tail) {
    CUDA_1D_KERNEL_LOOP_T(size_t, i, tail_count) { tail_dst[i] = value; }
  }
}

template<UnaryOp op, typename Src, typename Dst, size_t pack>
typename std::enable_if<(pack != 0), void>::type LaunchPackFill(CudaStream* stream, Dst* dst,
                                                                const Src* src, size_t count,
                                                                Scalar attr0, Scalar attr1) {
  const size_t pack_count = count / pack;
  const size_t tail_offset = pack_count * pack;
  const size_t tail_count = count - tail_offset;
  auto functor = UnaryFunctor<DeviceType::kCUDA, op, Dst, Src>(attr0, attr1);
  if (tail_count > 0) {
    LaunchFillKernel<op, Src, Dst, pack, true>
        <<<BlocksNum4ThreadsNum(pack_count), kCudaThreadsNumPerBlock, 0, stream->cuda_stream()>>>(
            functor, dst, src, pack_count, count, tail_count, dst + tail_offset);
  } else {
    LaunchFillKernel<op, Src, Dst, pack, false>
        <<<BlocksNum4ThreadsNum(pack_count), kCudaThreadsNumPerBlock, 0, stream->cuda_stream()>>>(
            functor, dst, src, pack_count, count, tail_count, dst + tail_offset);
  }
}

template<UnaryOp op, typename Src, typename Dst, size_t pack>
typename std::enable_if<(pack == 0), void>::type LaunchPackFill(CudaStream* stream, Dst* dst,
                                                                const Src* src, size_t count,
                                                                Scalar attr0, Scalar attr1) {
  LOG(FATAL) << "wrong alignment";
}

template<UnaryOp op, typename Src, typename Dst>
void LaunchFill(CudaStream* stream, Dst* dst, const Src* src, size_t count, Scalar attr0,
                Scalar attr1) {
  auto uintptr = reinterpret_cast<std::uintptr_t>(dst);
  if (uintptr % 16 == 0 && count * sizeof(Dst) >= 16) {
    LaunchPackFill<op, Src, Dst, 16 / sizeof(Dst)>(stream, dst, src, count, attr0, attr1);
  } else if (uintptr % 8 == 0 && count * sizeof(Dst) >= 8) {
    LaunchPackFill<op, Src, Dst, 8 / sizeof(Dst)>(stream, dst, src, count, attr0, attr1);
  } else if (uintptr % 4 == 0 && count * sizeof(Dst) >= 4) {
    LaunchPackFill<op, Src, Dst, 4 / sizeof(Dst)>(stream, dst, src, count, attr0, attr1);
  } else if (uintptr % 2 == 0 && count * sizeof(Dst) >= 2) {
    LaunchPackFill<op, Src, Dst, 2 / sizeof(Dst)>(stream, dst, src, count, attr0, attr1);
  } else {
    LaunchPackFill<op, Src, Dst, 1 / sizeof(Dst)>(stream, dst, src, count, attr0, attr1);
  }
}

template<UnaryOp unary_op, typename Src, DataType src_type, typename Dst, DataType dst_type>
class BroadcastElementwiseUnaryImpl : public BroadcastElementwiseUnary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseUnaryImpl);
  BroadcastElementwiseUnaryImpl(Scalar attr0, Scalar attr1) : attr0(attr0), attr1(attr1) {}
  ~BroadcastElementwiseUnaryImpl() override = default;

  void Launch(Stream* stream, size_t num_src_dims, const int64_t* src_dims, const void* src,
              size_t num_dst_dims, const int64_t* dst_dims, void* dst) override {
    CHECK_GT(num_src_dims, 0) << "num_src_dims must greater than 0";
    CHECK_GT(num_dst_dims, 0) << "num_dst_dims must greater than 0";
    int64_t src_strides[kMaxNumDims];
    int64_t dst_strides[kMaxNumDims];
    // init stride
    for (int i = num_src_dims - 1; i < kMaxNumDims; ++i) { src_strides[i] = 1; }
    for (int i = num_src_dims - 2; i >= 0; --i) {
      src_strides[i] = src_dims[i + 1] * src_strides[i + 1];
    }

    for (int i = num_dst_dims - 1; i < kMaxNumDims; ++i) { dst_strides[i] = 1; }
    for (int i = num_dst_dims - 2; i >= 0; --i) {
      dst_strides[i] = dst_dims[i + 1] * dst_strides[i + 1];
    }
    Launch(stream, num_src_dims, src_dims, src_strides, src, num_dst_dims, dst_dims, dst_strides,
           dst);
  }

  void Launch(Stream* stream, size_t num_src_dims, const int64_t* src_dims,
              const int64_t* src_strides, const void* src_ptr, size_t num_dst_dims,
              const int64_t* dst_dims, const int64_t* dst_strides, void* dst_ptr) override {
    CHECK_GT(num_src_dims, 0) << "num_src_dims must greater than 0";
    CHECK_GT(num_dst_dims, 0) << "num_dst_dims must greater than 0";
    auto* cuda_stream = stream->As<CudaStream>();
    Dst* dst = reinterpret_cast<Dst*>(dst_ptr);
    const Src* src = reinterpret_cast<const Src*>(src_ptr);
    size_t simplified_num_dims = 0;
    int permutation_list[kMaxNumDims];
    int64_t permutation_src_dims[kMaxNumDims];
    int64_t simplified_src_dims[kMaxNumDims];
    int64_t simplified_dst_dims[kMaxNumDims];
    int64_t simplified_src_strides[kMaxNumDims];
    int64_t simplified_dst_strides[kMaxNumDims];
    SimplifyBroadcastDims<kMaxNumDims>(num_src_dims, src_dims, src_strides, num_dst_dims, dst_dims,
                                       dst_strides, &simplified_num_dims, simplified_src_dims,
                                       simplified_src_strides, simplified_dst_dims,
                                       simplified_dst_strides);
    bool permutable = InferPermutable<kMaxNumDims>(
        simplified_num_dims, simplified_src_strides, simplified_dst_strides, simplified_src_dims,
        simplified_dst_dims, permutation_list, permutation_src_dims, unary_op);
    std::unique_ptr<Permute> permute =
        NewPrimitive<PermuteFactory>(DeviceType::kCUDA, simplified_num_dims);
    CheckInplace(simplified_num_dims, simplified_src_dims, src, simplified_dst_dims, dst);
    CheckInplace(simplified_num_dims, simplified_src_strides, src, simplified_dst_strides, dst);
    if (simplified_num_dims == 1 && simplified_src_dims[0] == 1) {
      const int64_t elem_cnt = simplified_dst_dims[0];
      LaunchFill<unary_op, Src, Dst>(cuda_stream, dst, src, elem_cnt, attr0, attr1);
    } else if (simplified_num_dims == 1 && simplified_src_strides[0] == 1
               && simplified_dst_strides[0] == 1) {
      const int64_t elem_cnt = simplified_src_dims[0];
      auto functor = UnaryFunctor<DeviceType::kCUDA, unary_op, Dst, Src>(attr0, attr1);
      OF_CUDA_CHECK((cuda::elementwise::Unary<decltype(functor), Dst, Src>(
          functor, elem_cnt, dst, src, cuda_stream->cuda_stream())));
    } else if (permutable && src_type == dst_type && permute) {
      permute->Launch(stream, dst_type, simplified_num_dims, permutation_src_dims, src_ptr,
                      permutation_list, dst_ptr);
    } else {
      // fall back to normal cases
      LaunchWithSimplified<unary_op, Src, Dst>(
          cuda_stream, simplified_num_dims, simplified_src_dims, simplified_src_strides, src,
          simplified_dst_dims, simplified_dst_strides, dst, attr0, attr1);
    }
  }

 protected:
  Scalar attr0, attr1;
};

template<UnaryOp unary_op, typename Src, DataType src_type, typename Dst, DataType dst_type>
std::unique_ptr<BroadcastElementwiseUnary> NewBroadcastElementwiseUnary(Scalar attr0,
                                                                        Scalar attr1) {
  return std::unique_ptr<BroadcastElementwiseUnary>(
      new BroadcastElementwiseUnaryImpl<unary_op, Src, src_type, Dst, dst_type>(attr0, attr1));
}

class BroadcastElementwiseUnaryFactoryImpl : public BroadcastElementwiseUnaryFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseUnaryFactoryImpl);
  BroadcastElementwiseUnaryFactoryImpl() = default;
  ~BroadcastElementwiseUnaryFactoryImpl() override = default;

  std::unique_ptr<BroadcastElementwiseUnary> New(UnaryOp op, DataType src_type, DataType dst_type,
                                                 size_t max_num_dims) override {
    return New(op, src_type, dst_type, max_num_dims, Scalar(), Scalar());
  }

  std::unique_ptr<BroadcastElementwiseUnary> New(UnaryOp op, DataType src_type, DataType dst_type,
                                                 size_t max_num_dims, Scalar attr0) override {
    return New(op, src_type, dst_type, max_num_dims, attr0, Scalar());
  }

  std::unique_ptr<BroadcastElementwiseUnary> New(UnaryOp unary_op, DataType src_type,
                                                 DataType dst_type, size_t max_num_dims,
                                                 Scalar attr0, Scalar attr1) override {
    if (max_num_dims > kMaxNumDims) { return nullptr; }
#define MAKE_NEW_SAME_DTYPE_BROADCAST_ELEMENTWISE_UNARY_ENTRY(unary_op, dtype_pair)          \
  {std::make_tuple(unary_op, OF_PP_PAIR_SECOND(dtype_pair), OF_PP_PAIR_SECOND(dtype_pair)),  \
   NewBroadcastElementwiseUnary<unary_op, OF_PP_PAIR_FIRST(dtype_pair),                      \
                                OF_PP_PAIR_SECOND(dtype_pair), OF_PP_PAIR_FIRST(dtype_pair), \
                                OF_PP_PAIR_SECOND(dtype_pair)>},

#define MAKE_NEW_BROADCAST_ELEMENTWISE_UNARY_ENTRY(unary_op, src_dtype_pair, dst_dtype_pair) \
  {std::make_tuple(unary_op, OF_PP_PAIR_SECOND(src_dtype_pair),                              \
                   OF_PP_PAIR_SECOND(dst_dtype_pair)),                                       \
   NewBroadcastElementwiseUnary<                                                             \
       unary_op, OF_PP_PAIR_FIRST(src_dtype_pair), OF_PP_PAIR_SECOND(src_dtype_pair),        \
       OF_PP_PAIR_FIRST(dst_dtype_pair), OF_PP_PAIR_SECOND(dst_dtype_pair)>},

    static const std::map<std::tuple<UnaryOp, DataType, DataType>,
                          std::function<std::unique_ptr<BroadcastElementwiseUnary>(Scalar, Scalar)>>
        new_broadcast_elementwise_unary_handle{
            // For All Type OP
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_SAME_DTYPE_BROADCAST_ELEMENTWISE_UNARY_ENTRY,
                                             UNARY_IDENTITY_SEQ, CUDA_PRIMITIVE_ALL_TYPE_SEQ)

            // For Cast OP
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                MAKE_NEW_BROADCAST_ELEMENTWISE_UNARY_ENTRY, BROADCAST_ELEMENTWISE_CAST_OP_SEQ,
                CUDA_PRIMITIVE_CAST_ALL_TYPE_SEQ, CUDA_PRIMITIVE_CAST_ALL_TYPE_SEQ)};

#undef MAKE_NEW_BROADCAST_ELEMENTWISE_UNARY_ENTRY
#undef MAKE_NEW_SAME_DTYPE_BROADCAST_ELEMENTWISE_UNARY_ENTRY

    const auto iter =
        new_broadcast_elementwise_unary_handle.find(std::make_tuple(unary_op, src_type, dst_type));
    if (iter != new_broadcast_elementwise_unary_handle.end()) {
      return iter->second(attr0, attr1);
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, BroadcastElementwiseUnaryFactory,
                           BroadcastElementwiseUnaryFactoryImpl);

}  // namespace
}  // namespace broadcast_elementwise_unary
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
