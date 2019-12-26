#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"

#include "oneflow/xrt/xla/xla_helpers.h"

namespace oneflow {
namespace xrt {
namespace mola {

std::vector<long long> AsLLongVec(const Shape &shape) {
  std::vector<long long> llong_vec(shape.NumAxes());
  for (size_t i = 0; i < llong_vec.size(); ++i) { llong_vec[i] = shape.At(i); }
  return std::move(llong_vec);
}

xla::XlaOp One(xla::XlaBuilder *builder, DataType data_type) {
  xla::PrimitiveType type = DataTypeToPrimitiveType(data_type);
  return xla::ConstantLiteral(builder, xla::LiteralUtil::One(type));
}

xla::XlaOp Zero(xla::XlaBuilder *builder, DataType data_type) {
  xla::PrimitiveType type = DataTypeToPrimitiveType(data_type);
  return xla::ConstantLiteral(builder, xla::LiteralUtil::Zero(type));
}

xla::XlaOp Ones(xla::XlaBuilder *builder, const Shape &shape, DataType data_type) {
  return xla::Broadcast(One(builder, data_type), AsLLongVec(shape));
}

xla::XlaOp Zeros(xla::XlaBuilder *builder, const Shape &shape, DataType data_type) {
  return xla::Broadcast(Zero(builder, data_type), AsLLongVec(shape));
}

xla::XlaOp IntegerLiteral(xla::XlaBuilder *builder, DataType data_type, int32_t value) {
  xla::PrimitiveType type = DataTypeToPrimitiveType(data_type);
  return ::tensorflow::IntegerLiteral(builder, type, value);
}

xla::XlaOp FloatLiteral(xla::XlaBuilder *builder, DataType data_type, float value) {
  xla::PrimitiveType type = DataTypeToPrimitiveType(data_type);
  return ::tensorflow::FloatLiteral(builder, type, value);
}

xla::XlaOp Reshape(xla::XlaOp input, Shape dest_shape) {
  std::vector<long long> shape_dim(dest_shape.NumAxes());
  for (int i = 0; i < dest_shape.NumAxes(); ++i) { shape_dim[i] = dest_shape.At(i); }
  return xla::Reshape(input, shape_dim);
}

xla::XlaOp MinValue(xla::XlaBuilder *builder, DataType data_type) {
  xla::PrimitiveType type = DataTypeToPrimitiveType(data_type);
  return xla::MinValue(builder, type);
}

xla::XlaOp MaxValue(xla::XlaBuilder *builder, DataType data_type) {
  xla::PrimitiveType type = DataTypeToPrimitiveType(data_type);
  return xla::MaxValue(builder, type);
}

#define OFXLA_CREATE_BINARY_COMPUTATION_FUNC(func_type)                             \
  xla::XlaComputation Create##func_type##Func(DataType data_type) {                 \
    std::string func_name = absl::StrCat(#func_type ".template<", data_type, ">");  \
    xla::XlaBuilder builder(func_name);                                             \
    xla::PrimitiveType type = DataTypeToPrimitiveType(data_type);                   \
    auto x = xla::Parameter(&builder, 0, xla::ShapeUtil::MakeShape(type, {}), "x"); \
    auto y = xla::Parameter(&builder, 1, xla::ShapeUtil::MakeShape(type, {}), "y"); \
    xla::func_type(x, y);                                                           \
    return builder.Build().ConsumeValueOrDie();                                     \
  }

OFXLA_CREATE_BINARY_COMPUTATION_FUNC(Max);
OFXLA_CREATE_BINARY_COMPUTATION_FUNC(Min);
OFXLA_CREATE_BINARY_COMPUTATION_FUNC(Add);
OFXLA_CREATE_BINARY_COMPUTATION_FUNC(Sub);
OFXLA_CREATE_BINARY_COMPUTATION_FUNC(Mul);
OFXLA_CREATE_BINARY_COMPUTATION_FUNC(Div);

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
