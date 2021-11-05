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
#include "oneflow/core/primitive/common/binary_functor.h"
#include "oneflow/core/primitive/cpu/type_seq.h"

namespace oneflow {

namespace primitive {

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFloorDiv, Src, Dst> {
  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return std::floor(src0 / src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFloorDiv, float16, float16> {
  OF_DEVICE_FUNC float16 operator()(float16 src0, float16 src1) const {
    return static_cast<float16>(std::floor(static_cast<float>(src0) / static_cast<float>(src1)));
  }
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kPow, Src, Dst> {
  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return std::pow(src0, src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kPow, float16, float16> {
  OF_DEVICE_FUNC float16 operator()(float16 src0, float16 src1) const {
    return static_cast<float16>(std::pow(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

template<typename Src, typename Dst>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFmod, Src, Dst> {
  OF_DEVICE_FUNC Dst operator()(Src src0, Src src1) const { return std::fmod(src0, src1); }
};

template<>
struct BinaryFunctor<DeviceType::kCPU, BinaryOp::kFmod, float16, float16> {
  OF_DEVICE_FUNC float16 operator()(float16 src0, float16 src1) const {
    return static_cast<float16>(std::fmod(static_cast<float>(src0), static_cast<float>(src1)));
  }
};

namespace {

template<typename T>
T GetValue(Scalar value) {
  return value.Value<T>();
}

template<>
float16 GetValue<float16>(Scalar value) {
  return static_cast<float16>(GetValue<float>(value));
}

template<BinaryOp binary_op, typename Src, typename Dst, size_t num_dims, size_t pack_size,
         typename IndexType>
void LaunchKernel(StreamContext* stream_ctx,
                  BroadcastElementwiseBinaryParams<num_dims, IndexType> params) {
  const PackType<Src, pack_size>* src0 =
      reinterpret_cast<const PackType<Src, pack_size>*>(params.src0);
  const PackType<Src, pack_size>* src1 =
      reinterpret_cast<const PackType<Src, pack_size>*>(params.src1);
  PackType<Dst, pack_size>* dst = reinterpret_cast<PackType<Dst, pack_size>*>(params.dst);
  IndexType src0_index[num_dims];
  IndexType src1_index[num_dims];
  IndexType dst_index[num_dims];
  for (IndexType offset = 0; offset < params.count; ++offset) {
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
    for (int j = 0; j < pack_size; ++j) {
      dst_pack.elem[j] = BinaryFunctor<DeviceType::kCPU, binary_op, Src, Dst>()(src0_pack.elem[j],
                                                                                src1_pack.elem[j]);
    }
    dst[offset] = dst_pack.storage;
  }
}

size_t GetElementCount(size_t num_dims, const int64_t* dims) {
  size_t count = 1;
  for (size_t i = 0; i < num_dims; ++i) { count *= dims[i]; }
  return count;
}

template<BinaryOp binary_op, typename Src, typename Dst, bool scalar_left>
void LaunchScalarBinary(size_t n, Src scalar, const Src* src, Dst* dst) {
  for (size_t i = 0; i < n; ++i) {
    if (scalar_left) {
      dst[i] = BinaryFunctor<DeviceType::kCPU, binary_op, Src, Dst>()(scalar, src[i]);
    } else {
      dst[i] = BinaryFunctor<DeviceType::kCPU, binary_op, Src, Dst>()(src[i], scalar);
    }
  }
}

template<BinaryOp binary_op, typename Src, typename Dst>
class BroadcastElementwiseBinaryImpl : public BroadcastElementwiseBinary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseBinaryImpl);
  BroadcastElementwiseBinaryImpl() = default;
  ~BroadcastElementwiseBinaryImpl() override = default;

  void Launch(StreamContext* stream_ctx, Scalar src0, size_t num_src1_dims,
              const int64_t* src1_dims, const void* src1, void* dst) override {
    LOG(ERROR) << "left scalar";
    size_t elem_cnt = GetElementCount(num_src1_dims, src1_dims);
    LaunchScalarBinary<binary_op, Src, Dst, true>(elem_cnt, GetValue<Src>(src0),
                                                  reinterpret_cast<const Src*>(src1),
                                                  reinterpret_cast<Dst*>(dst));
  }
  void Launch(StreamContext* stream_ctx, size_t num_src0_dims, const int64_t* src0_dims,
              const void* src0, Scalar src1, void* dst) override {
    LOG(ERROR) << "right scalar";
    size_t elem_cnt = GetElementCount(num_src0_dims, src0_dims);
    LaunchScalarBinary<binary_op, Src, Dst, false>(elem_cnt, GetValue<Src>(src1),
                                                   reinterpret_cast<const Src*>(src0),
                                                   reinterpret_cast<Dst*>(dst));
  }
  void Launch(StreamContext* stream_ctx, size_t num_src0_dims, const int64_t* src0_dims,
              const void* src0, size_t num_src1_dims, const int64_t* src1_dims, const void* src1,
              void* dst) override {
    LOG(ERROR) << "SimplifyThenLaunch";
    SimplifyThenLaunch<binary_op, Src, Dst>(stream_ctx, num_src0_dims, src0_dims, src0,
                                            num_src1_dims, src1_dims, src1, dst);
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
                                             BINARY_MATH_OP_SEQ, CPU_PRIMITIVE_ALL_TYPE_SEQ)
                OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                    MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_DIFFERENT_DTYPE_ENTRY,
                    BINARY_LOGICAL_OP_SEQ, CPU_PRIMITIVE_ALL_TYPE_SEQ,
                    CPU_PRIMITIVE_INT8_TYPE_SEQ)};

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

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, BroadcastElementwiseBinaryFactory,
                           BroadcastElementwiseBinaryFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow
