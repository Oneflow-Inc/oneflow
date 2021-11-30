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

#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"
#include "oneflow/core/ep/common/primitive/broadcast_elementwise_binary.h"
#include "oneflow/core/ep/cpu/primitive/binary_functor.h"
#include "oneflow/core/ep/cpu/primitive/type_seq.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

template<typename T>
T GetValue(Scalar value) {
  return value.Value<T>();
}

template<>
float16 GetValue<float16>(Scalar value) {
  return static_cast<float16>(GetValue<float>(value));
}

template<BinaryOp binary_op, typename Src, typename Dst>
void LaunchLhsScalarBinary(size_t n, Src scalar, const Src* src, Dst* dst) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] = BinaryFunctor<DeviceType::kCPU, binary_op, Src, Dst>()(scalar, src[i]);
  }
}

template<BinaryOp binary_op, typename Src, typename Dst>
void LaunchRhsScalarBinary(size_t n, Src scalar, const Src* src, Dst* dst) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] = BinaryFunctor<DeviceType::kCPU, binary_op, Src, Dst>()(src[i], scalar);
  }
}

template<BinaryOp binary_op, typename Src, typename Dst>
void LaunchLhsScalarPtrBinary(size_t n, const Src* scalar, const Src* src, Dst* dst) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] = BinaryFunctor<DeviceType::kCPU, binary_op, Src, Dst>()(scalar[0], src[i]);
  }
}

template<BinaryOp binary_op, typename Src, typename Dst>
void LaunchRhsScalarPtrBinary(size_t n, const Src* scalar, const Src* src, Dst* dst) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] = BinaryFunctor<DeviceType::kCPU, binary_op, Src, Dst>()(src[i], scalar[0]);
  }
}

template<BinaryOp binary_op, typename Src, typename Dst>
class BroadcastElementwiseBinaryImpl : public BroadcastElementwiseBinary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseBinaryImpl);
  BroadcastElementwiseBinaryImpl() = default;
  ~BroadcastElementwiseBinaryImpl() override = default;

  void Launch(Stream* stream, Scalar src0, size_t num_src1_dims, const int64_t* src1_dims,
              const void* src1, void* dst) override {
    size_t elem_cnt = GetElementCount(num_src1_dims, src1_dims);
    LaunchLhsScalarBinary<binary_op, Src, Dst>(elem_cnt, GetValue<Src>(src0),
                                               reinterpret_cast<const Src*>(src1),
                                               reinterpret_cast<Dst*>(dst));
  }
  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              Scalar src1, void* dst) override {
    size_t elem_cnt = GetElementCount(num_src0_dims, src0_dims);
    LaunchRhsScalarBinary<binary_op, Src, Dst>(elem_cnt, GetValue<Src>(src1),
                                               reinterpret_cast<const Src*>(src0),
                                               reinterpret_cast<Dst*>(dst));
  }
  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              size_t num_src1_dims, const int64_t* src1_dims, const void* src1,
              void* dst) override {
    const size_t src0_elem_cnt = GetElementCount(num_src0_dims, src0_dims);
    const size_t src1_elem_cnt = GetElementCount(num_src1_dims, src1_dims);
    if (src0_elem_cnt == 1) {
      LaunchLhsScalarPtrBinary<binary_op, Src, Dst>(
          src1_elem_cnt, reinterpret_cast<const Src*>(src0), reinterpret_cast<const Src*>(src1),
          reinterpret_cast<Dst*>(dst));
    } else if (src1_elem_cnt == 1) {
      LaunchRhsScalarPtrBinary<binary_op, Src, Dst>(
          src0_elem_cnt, reinterpret_cast<const Src*>(src1), reinterpret_cast<const Src*>(src0),
          reinterpret_cast<Dst*>(dst));
    } else {
      UNIMPLEMENTED();
    }
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
#define MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY(binary_op, data_type_pair) \
  {std::make_tuple(binary_op, OF_PP_PAIR_SECOND(data_type_pair),                    \
                   OF_PP_PAIR_SECOND(data_type_pair)),                              \
   NewBroadcastElementwiseBinary<binary_op, OF_PP_PAIR_FIRST(data_type_pair),       \
                                 OF_PP_PAIR_FIRST(data_type_pair)>},

#define MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_COMPARASION_AND_LOGICAL_ENTRY(      \
    binary_op, src_data_type_pair, dst_data_type_pair)                            \
  {std::make_tuple(binary_op, OF_PP_PAIR_SECOND(src_data_type_pair),              \
                   OF_PP_PAIR_SECOND(dst_data_type_pair)),                        \
   NewBroadcastElementwiseBinary<binary_op, OF_PP_PAIR_FIRST(src_data_type_pair), \
                                 OF_PP_PAIR_FIRST(dst_data_type_pair)>},

    static const std::map<std::tuple<BinaryOp, DataType, DataType>,
                          std::function<std::unique_ptr<BroadcastElementwiseBinary>()>>
        new_broadcast_elementwise_binary_handle{
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY,
                                             BINARY_MATH_OP_SEQ, CPU_PRIMITIVE_ALL_TYPE_SEQ)
                OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                    MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_COMPARASION_AND_LOGICAL_ENTRY,
                    BINARY_COMPARISION_OP_SEQ BINARY_LOGICAL_OP_SEQ, CPU_PRIMITIVE_ALL_TYPE_SEQ,
                    CPU_PRIMITIVE_INT8_TYPE_SEQ)};

#undef MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_COMPARASION_AND_LOGICAL_ENTRY
#undef MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY

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
}  // namespace ep

}  // namespace oneflow
