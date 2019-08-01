#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "oneflow/xla/of2xla/xla_helpers.h"

namespace oneflow {
namespace mola {

xla::XlaOp One(xla::XlaBuilder *builder, DataType data_type) {
  xla::PrimitiveType type = DataTypeToPrimitiveType(data_type);
  return xla::ConstantLiteral(builder, xla::LiteralUtil::One(type));
}

xla::XlaOp Zero(xla::XlaBuilder *builder, DataType data_type) {
  xla::PrimitiveType type = DataTypeToPrimitiveType(data_type);
  return xla::ConstantLiteral(builder, xla::LiteralUtil::Zero(type));
}

xla::XlaOp IntegerLiteral(xla::XlaBuilder *builder, DataType data_type,
                          int32_t value) {
  xla::PrimitiveType type = DataTypeToPrimitiveType(data_type);
  return ::tensorflow::IntegerLiteral(builder, type, value);
}

xla::XlaOp FloatLiteral(xla::XlaBuilder *builder, DataType data_type,
                        float value) {
  xla::PrimitiveType type = DataTypeToPrimitiveType(data_type);
  return ::tensorflow::FloatLiteral(builder, type, value);
}

xla::XlaOp Reshape(xla::XlaOp input, Shape dest_shape) {
  std::vector<long long> shape_dim(dest_shape.NumAxes());
  for (int i = 0; i < dest_shape.NumAxes(); ++i) {
    shape_dim[i] = dest_shape.At(i);
  }
  return xla::Reshape(input, shape_dim);
}

}  // namespace mola
}  // namespace oneflow
