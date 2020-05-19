#ifndef ONEFLOW_XRT_XLA_XLA_HELPERS_H_
#define ONEFLOW_XRT_XLA_XLA_HELPERS_H_

#include "oneflow/xrt/xla/xla_data_type.h"
#include "oneflow/xrt/xla/xla_shape.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace xrt {
namespace mola {

xla::XlaOp One(xla::XlaBuilder *builder, DataType data_type);

xla::XlaOp Zero(xla::XlaBuilder *builder, DataType data_type);

xla::XlaOp Ones(xla::XlaBuilder *builder, const Shape &shape, DataType data_type);

xla::XlaOp Zeros(xla::XlaBuilder *builder, const Shape &shape, DataType data_type);

xla::XlaOp IntegerLiteral(xla::XlaBuilder *builder, DataType data_type, int32_t value);

xla::XlaOp FloatLiteral(xla::XlaBuilder *builder, DataType data_type, float value);

xla::XlaOp Reshape(xla::XlaOp input, Shape dest_shape);

xla::XlaOp MinValue(xla::XlaBuilder *builder, DataType data_type);
xla::XlaOp MaxValue(xla::XlaBuilder *builder, DataType data_type);

// Create computation of max func with data_type
xla::XlaComputation CreateMaxFunc(DataType data_type);

// Create computation of min func with data_type
xla::XlaComputation CreateMinFunc(DataType data_type);

xla::XlaComputation CreateAddFunc(DataType data_type);

xla::XlaComputation CreateSubFunc(DataType data_type);

xla::XlaComputation CreateMulFunc(DataType data_type);

xla::XlaComputation CreateDivFunc(DataType data_type);

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_XLA_HELPERS_H_
