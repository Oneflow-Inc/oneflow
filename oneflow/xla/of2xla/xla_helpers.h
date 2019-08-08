#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_HELPERS_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_HELPERS_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/xla/of2xla/xla_data_type.h"
#include "oneflow/xla/of2xla/xla_shape.h"

namespace oneflow {
namespace mola {

xla::XlaOp One(xla::XlaBuilder *builder, DataType data_type);

xla::XlaOp Zero(xla::XlaBuilder *builder, DataType data_type);

xla::XlaOp IntegerLiteral(xla::XlaBuilder *builder, DataType data_type,
                          int32_t value);

xla::XlaOp FloatLiteral(xla::XlaBuilder *builder, DataType data_type,
                        float value);

xla::XlaOp Reshape(xla::XlaOp input, Shape dest_shape);

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_HELPERS_H_
