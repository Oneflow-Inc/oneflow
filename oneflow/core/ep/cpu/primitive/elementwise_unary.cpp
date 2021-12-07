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
#include "oneflow/core/ep/common/primitive/elementwise_unary.h"
#include "oneflow/core/ep/common/primitive/unary_functor.h"
#include "oneflow/core/ep/cpu/primitive/type_seq.h"
#include "oneflow/core/ep/cpu/cpu_stream.h"

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

template<UnaryOp unary_op, typename Src, typename Dst>
class ElementwiseUnaryImpl : public ElementwiseUnary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnaryImpl);
  ElementwiseUnaryImpl() = default;
  ~ElementwiseUnaryImpl() override = default;

  void Launch(Stream* stream, const void* src_ptr, void* dst_ptr, size_t count) override {
    Dst* dst = reinterpret_cast<Dst*>(dst_ptr);
    const Src* src = reinterpret_cast<const Src*>(src_ptr);
    for (size_t i = 0; i < count; ++i) {
      dst[i] = UnaryFunctor<DeviceType::kCPU, unary_op, Dst, Src>()(src[i]);
    }
  }
};

#ifdef WITH_ONEDNN
class ElementwiseUnaryneDnnImpl : public ElementwiseUnary {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnaryneDnnImpl);
  ElementwiseUnaryneDnnImpl(dnnl::algorithm algorithm, dnnl::memory::data_type type)
      : type_onednn_(type), algorithm_(algorithm) {};
  ~ElementwiseUnaryneDnnImpl() override = default;

  void Launch(Stream* stream, const void* src_ptr, void* dst_ptr, size_t count) override {
    dnnl::engine* onednn_engine = stream->As<CpuStream>()->onednn_engine();
    dnnl::stream* onednn_stream = stream->As<CpuStream>()->onednn_stream();

    dnnl::memory::dims src_dims = {static_cast<dnnl::memory::dim>(count)};

    auto src_md = dnnl::memory::desc(src_dims, type_onednn_, dnnl::memory::format_tag::x);
    auto dst_md = dnnl::memory::desc(src_dims, type_onednn_, dnnl::memory::format_tag::x);

    auto src_mem = dnnl::memory(src_md, *onednn_engine, const_cast<void*>(src_ptr));
    auto dst_mem = dnnl::memory(dst_md, *onednn_engine, dst_ptr);

    auto eltwise_d = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference, algorithm_,
                                                 src_md, 1.0, 0.0);
    auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(eltwise_d, *onednn_engine);
    auto eltwise_prim = dnnl::eltwise_forward(eltwise_pd);

    std::unordered_map<int, dnnl::memory> tmp_eltwise_args{{DNNL_ARG_SRC, src_mem},
                                                           {DNNL_ARG_DST, dst_mem}};
    eltwise_prim.execute(*onednn_stream, tmp_eltwise_args);

    onednn_stream->wait();
  }

 private:
  dnnl::memory::data_type type_onednn_;
  dnnl::algorithm algorithm_;
};

#define CPU_PRIMITIVE_ADD_ONEDNN_TYPE_SEQ \
  CPU_PRIMITIVE_ONEDNN_INT8_TYPE_SEQ      \
  CPU_PRIMITIVE_ONEDNN_UINT8_TYPE_SEQ     \
  CPU_PRIMITIVE_ONEDNN_INT32_TYPE_SEQ     \
  CPU_PRIMITIVE_ONEDNN_FLOAT_TYPE_SEQ     \
  CPU_PRIMITIVE_ONEDNN_FLOAT16_TYPE_SEQ   \
  CPU_PRIMITIVE_ONEDNN_BFLOAT16_TYPE_SEQ

#define CPU_PRIMITIVE_ADD_DEFAULT_TYPE_SEQ \
  CPU_PRIMITIVE_CHAR_TYPE_SEQ              \
  CPU_PRIMITIVE_DOUBLE_TYPE_SEQ            \
  CPU_PRIMITIVE_INT64_TYPE_SEQ

#define ELTWISE_ONEDNN_ABS_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kAbs, dnnl::algorithm::eltwise_abs)
#define ELTWISE_ONEDNN_EXP_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kExp, dnnl::algorithm::eltwise_exp)
#define ELTWISE_ONEDNN_GELU_ERF_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kGeluErf, dnnl::algorithm::eltwise_gelu_erf)
#define ELTWISE_ONEDNN_GELU_TANH_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kGeluTanh, dnnl::algorithm::eltwise_gelu_tanh)
#define ELTWISE_ONEDNN_HARDSWISH_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kHardSwish, dnnl::algorithm::eltwise_hardswish)
#define ELTWISE_ONEDNN_LOG_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLog, dnnl::algorithm::eltwise_log)
#define ELTWISE_ONEDNN_LOGISTIC_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLogistic, dnnl::algorithm::eltwise_logistic)
#define ELTWISE_ONEDNN_LOGSIGMOD_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kLogsigmoid, dnnl::algorithm::eltwise_logsigmoid)
#define ELTWISE_ONEDNN_MISH_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kMish, dnnl::algorithm::eltwise_mish)
#define ELTWISE_ONEDNN_RELU_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kRelu, dnnl::algorithm::eltwise_relu)
#define ELTWISE_ONEDNN_ROUND_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kRound, dnnl::algorithm::eltwise_round)
#define ELTWISE_ONEDNN_SOFT_RELU_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSoftRelu, dnnl::algorithm::eltwise_soft_relu)
#define ELTWISE_ONEDNN_SQRT_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSqrt, dnnl::algorithm::eltwise_sqrt)
#define ELTWISE_ONEDNN_SQUARE_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kSquare, dnnl::algorithm::eltwise_square)
#define ELTWISE_ONEDNN_TANH_SEQ OF_PP_MAKE_TUPLE_SEQ(UnaryOp::kTanh, dnnl::algorithm::eltwise_tanh)

#define ELTWISE_ONEDNN_SEQ        \
  ELTWISE_ONEDNN_ABS_SEQ          \
  ELTWISE_ONEDNN_EXP_SEQ          \
  ELTWISE_ONEDNN_GELU_ERF_SEQ     \
  ELTWISE_ONEDNN_GELU_TANH_SEQ    \
  ELTWISE_ONEDNN_HARDSWISH_SEQ    \
  ELTWISE_ONEDNN_LOG_SEQ          \
  ELTWISE_ONEDNN_LOGISTIC_SEQ     \
  ELTWISE_ONEDNN_LOGSIGMOD_SEQ    \
  ELTWISE_ONEDNN_MISH_SEQ         \
  ELTWISE_ONEDNN_RELU_SEQ         \
  ELTWISE_ONEDNN_ROUND_SEQ        \
  ELTWISE_ONEDNN_SOFT_RELU_SEQ    \
  ELTWISE_ONEDNN_SQRT_SEQ         \
  ELTWISE_ONEDNN_SQUARE_SEQ       \
  ELTWISE_ONEDNN_TANH_SEQ

template<dnnl::algorithm algorithm, dnnl::memory::data_type type_onednn>
std::unique_ptr<ElementwiseUnary> NewOneDnnElementwiseUnary() {
  return std::unique_ptr<ElementwiseUnary>(new ElementwiseUnaryneDnnImpl(algorithm, type_onednn));
}

#endif

template<UnaryOp unary_op, typename Src, typename Dst>
std::unique_ptr<ElementwiseUnary> NewElementwiseUnary() {
  return std::unique_ptr<ElementwiseUnary>(new ElementwiseUnaryImpl<unary_op, Src, Dst>());
}

class ElementwiseUnaryFactoryImpl : public ElementwiseUnaryFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseUnaryFactoryImpl);
  ElementwiseUnaryFactoryImpl() = default;
  ~ElementwiseUnaryFactoryImpl() override = default;

  std::unique_ptr<ElementwiseUnary> New(UnaryOp unary_op, DataType src_type,
                                        DataType dst_dtype) override {
#define MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY(unary_op, dtype_pair)                   \
  {std::make_tuple(unary_op, OF_PP_PAIR_SECOND(dtype_pair), OF_PP_PAIR_SECOND(dtype_pair)), \
   NewElementwiseUnary<unary_op, OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(dtype_pair)>},

#define MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY(unary_op, src_type_pair, dst_dtype_pair)  \
  {std::make_tuple(unary_op, OF_PP_PAIR_SECOND(src_type_pair), OF_PP_PAIR_SECOND(dst_dtype_pair)), \
   NewElementwiseUnary<unary_op, OF_PP_PAIR_FIRST(src_type_pair),                                  \
                       OF_PP_PAIR_FIRST(dst_dtype_pair)>},

#ifdef WITH_ONEDNN
#define MAKE_NEW_SAME_DTYPE_ONEDNN_ELEMENTWISE_UNARY_ENTRY(unary_op, dtype_pair) \
  {std::make_tuple(OF_PP_PAIR_FIRST(unary_op), OF_PP_PAIR_SECOND(dtype_pair),    \
                   OF_PP_PAIR_SECOND(dtype_pair)),                               \
   NewOneDnnElementwiseUnary<OF_PP_PAIR_SECOND(unary_op), OF_PP_PAIR_FIRST(dtype_pair)>},

    static const std::map<std::tuple<UnaryOp, DataType, DataType>,
                          std::function<std::unique_ptr<ElementwiseUnary>()>>
        new_elementwise_unary_handle{
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_SAME_DTYPE_ONEDNN_ELEMENTWISE_UNARY_ENTRY,
                                             ELTWISE_ONEDNN_SEQ, CPU_PRIMITIVE_ADD_ONEDNN_TYPE_SEQ)

                OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY,
                                                 UNARY_MATH_OP_SEQ,
                                                 CPU_PRIMITIVE_ADD_DEFAULT_TYPE_SEQ)

                    OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                        MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY, UNARY_LOGICAL_OP_SEQ,
                        CPU_PRIMITIVE_NATIVE_TYPE_SEQ, CPU_PRIMITIVE_INT8_TYPE_SEQ)};

#else
    static const std::map<std::tuple<UnaryOp, DataType, DataType>,
                          std::function<std::unique_ptr<ElementwiseUnary>()>>
        new_elementwise_unary_handle{
            OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY,
                                             UNARY_MATH_OP_SEQ, CPU_PRIMITIVE_NATIVE_TYPE_SEQ)

                OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
                    MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY, UNARY_LOGICAL_OP_SEQ,
                    CPU_PRIMITIVE_NATIVE_TYPE_SEQ, CPU_PRIMITIVE_INT8_TYPE_SEQ)};
#endif

#undef MAKE_NEW_DIFFERENT_DTYPE_ELEMENTWISE_UNARY_ENTRY

#undef MAKE_NEW_SAME_DTYPE_ELEMENTWISE_UNARY_ENTRY

    const auto it =
        new_elementwise_unary_handle.find(std::make_tuple(unary_op, src_type, dst_dtype));
    if (it != new_elementwise_unary_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, ElementwiseUnaryFactory, ElementwiseUnaryFactoryImpl);

}  // namespace
}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
