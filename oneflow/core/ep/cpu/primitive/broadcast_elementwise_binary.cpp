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
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"

namespace oneflow {

namespace ep {
namespace primitive {
namespace broadcast_elementwise_binary {

namespace {

template<typename T>
T GetValue(Scalar value) {
  return value.Value<T>();
}

template<>
float16 GetValue<float16>(Scalar value) {
  return static_cast<float16>(GetValue<float>(value));
}

template<BinaryOp binary_op, typename Src, typename Dst,
         void (*binary_func)(ep::Stream* stream, const XpuVarNdarray<Dst>& z,
                             const XpuVarNdarray<const Src>& x, const XpuVarNdarray<const Src>& y)>
class BroadcastElementwiseBinaryImpl : public BroadcastElementwiseBinary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseBinaryImpl);
  BroadcastElementwiseBinaryImpl() = default;
  ~BroadcastElementwiseBinaryImpl() override = default;

  void Launch(Stream* stream, Scalar src0, size_t num_src1_dims, const int64_t* src1_dims,
              const void* src1, void* dst) override {
    int64_t elem_cnt = GetElementCount(num_src1_dims, src1_dims);
    Src src0_val = GetValue<Src>(src0);
    binary_func(stream, XpuVarNdarray<Dst>(Shape({elem_cnt}), reinterpret_cast<Dst*>(dst), 1),
                XpuVarNdarray<const Src>(Shape({1}), &src0_val, 1),
                XpuVarNdarray<const Src>(Shape({elem_cnt}), reinterpret_cast<const Src*>(src1), 1));
  }
  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              Scalar src1, void* dst) override {
    int64_t elem_cnt = GetElementCount(num_src0_dims, src0_dims);
    Src src1_val = GetValue<Src>(src1);
    binary_func(stream, XpuVarNdarray<Dst>(Shape({elem_cnt}), reinterpret_cast<Dst*>(dst), 1),
                XpuVarNdarray<const Src>(Shape({elem_cnt}), reinterpret_cast<const Src*>(src0), 1),
                XpuVarNdarray<const Src>(Shape({1}), &src1_val, 1));
  }
  void Launch(Stream* stream, size_t num_src0_dims, const int64_t* src0_dims, const void* src0,
              size_t num_src1_dims, const int64_t* src1_dims, const void* src1,
              void* dst) override {
    DimVector src0_dim_vec;
    DimVector src1_dim_vec;
    DimVector dst_dim_vec;
    size_t num_dims = 0;
    int64_t simplified_src0_dims[kMaxNumDims];
    int64_t simplified_src1_dims[kMaxNumDims];
    int64_t simplified_dst_dims[kMaxNumDims];
    SimplifyBroadcastDims<kMaxNumDims>(num_src0_dims, src0_dims, num_src1_dims, src1_dims,
                                       &num_dims, simplified_src0_dims, simplified_src1_dims,
                                       simplified_dst_dims);
    CheckInplace(num_dims, simplified_src0_dims, src0, simplified_src1_dims, src1,
                 simplified_dst_dims, dst);
    for (int64_t i = 0; i < num_dims; ++i) {
      src0_dim_vec.push_back(simplified_src0_dims[i]);
      src1_dim_vec.push_back(simplified_src1_dims[i]);
      dst_dim_vec.push_back(simplified_dst_dims[i]);
    }
    binary_func(
        stream, XpuVarNdarray<Dst>(Shape(dst_dim_vec), reinterpret_cast<Dst*>(dst), num_dims),
        XpuVarNdarray<const Src>(Shape(src0_dim_vec), reinterpret_cast<const Src*>(src0), num_dims),
        XpuVarNdarray<const Src>(Shape(src1_dim_vec), reinterpret_cast<const Src*>(src1),
                                 num_dims));
  }
};

template<BinaryOp binary_op, typename Src, typename Dst,
         void (*binary_func)(ep::Stream* stream, const XpuVarNdarray<Dst>& z,
                             const XpuVarNdarray<const Src>& x, const XpuVarNdarray<const Src>& y)>
std::unique_ptr<BroadcastElementwiseBinary> NewBroadcastElementwiseBinary() {
  return std::unique_ptr<BroadcastElementwiseBinary>(
      new BroadcastElementwiseBinaryImpl<binary_op, Src, Dst, binary_func>());
}

#define BINARY_MATH_OP_NDARRAY_PAIR         \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kAdd, Add) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kSub, Sub) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kMul, Mul) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kDiv, Div) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kMax, Max) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kMin, Min) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kPow, Pow)

#define NDARRAY_BINARY_TYPE_SEQ \
  CPU_PRIMITIVE_INT8_TYPE_SEQ   \
  CPU_PRIMITIVE_UINT8_TYPE_SEQ  \
  CPU_PRIMITIVE_INT32_TYPE_SEQ  \
  CPU_PRIMITIVE_INT64_TYPE_SEQ  \
  CPU_PRIMITIVE_FLOAT_TYPE_SEQ  \
  CPU_PRIMITIVE_DOUBLE_TYPE_SEQ \
  CPU_PRIMITIVE_FLOAT16_TYPE_SEQ

#define BINARY_LOGICAL_COMPARISION_OP_NDARRAY_PAIR  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kEqual, EQ)        \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kNotEqual, NE)     \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLessThan, LT)     \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLessEqual, LE)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kGreaterThan, GT)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kGreaterEqual, GE) \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalAnd, AND)  \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalOr, OR)    \
  OF_PP_MAKE_TUPLE_SEQ(BinaryOp::kLogicalXor, XOR)

class BroadcastElementwiseBinaryFactoryImpl : public BroadcastElementwiseBinaryFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastElementwiseBinaryFactoryImpl);
  BroadcastElementwiseBinaryFactoryImpl() = default;
  ~BroadcastElementwiseBinaryFactoryImpl() override = default;

  std::unique_ptr<BroadcastElementwiseBinary> New(BinaryOp binary_op, DataType src_type,
                                                  DataType dst_type, size_t max_num_dims) override {
    if (max_num_dims > kMaxNumDims) { return nullptr; }
#define MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY(binary_op_pair, data_type_pair) \
  {std::make_tuple(OF_PP_PAIR_FIRST(binary_op_pair), OF_PP_PAIR_SECOND(data_type_pair),  \
                   OF_PP_PAIR_SECOND(data_type_pair)),                                   \
   NewBroadcastElementwiseBinary<                                                        \
       OF_PP_PAIR_FIRST(binary_op_pair), OF_PP_PAIR_FIRST(data_type_pair),               \
       OF_PP_PAIR_FIRST(data_type_pair),                                                 \
       &NdarrayUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(data_type_pair)>::OF_PP_CAT(      \
           Broadcast, OF_PP_PAIR_SECOND(binary_op_pair))>},

#define MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_COMPARASION_AND_LOGICAL_ENTRY(                \
    binary_op_pair, src_data_type_pair, dst_data_type_pair)                                 \
  {std::make_tuple(OF_PP_PAIR_FIRST(binary_op_pair), OF_PP_PAIR_SECOND(src_data_type_pair), \
                   OF_PP_PAIR_SECOND(dst_data_type_pair)),                                  \
   NewBroadcastElementwiseBinary<                                                           \
       OF_PP_PAIR_FIRST(binary_op_pair), OF_PP_PAIR_FIRST(src_data_type_pair),              \
       OF_PP_PAIR_FIRST(dst_data_type_pair),                                                \
       &NdarrayUtil<DeviceType::kCPU, OF_PP_PAIR_FIRST(src_data_type_pair)>::OF_PP_CAT(     \
           Broadcast, OF_PP_PAIR_SECOND(binary_op_pair))>},

    static const std::map<std::tuple<BinaryOp, DataType, DataType>,
                          std::function<std::unique_ptr<BroadcastElementwiseBinary>()>>
        new_broadcast_elementwise_binary_handle{
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_MATH_ENTRY,
                                             BINARY_MATH_OP_NDARRAY_PAIR, NDARRAY_BINARY_TYPE_SEQ)
                OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                    MAKE_NEW_BROADCAST_ELEMENTWISE_BINARY_COMPARASION_AND_LOGICAL_ENTRY,
                    BINARY_LOGICAL_COMPARISION_OP_NDARRAY_PAIR, NDARRAY_BINARY_TYPE_SEQ,
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
}  // namespace broadcast_elementwise_binary
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
