#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/core/compiler/of2xla/xla_op_context.h"

namespace oneflow {
namespace mola {

xla::Shape GetXlaOpShape(const xla::XlaOp &handle) {
  CHECK(handle.IsUninitialized()) << "XlaOp has not been initialized.";
  xla::StatusOr<xla::Shape> shape = handle.builder()->GetShape(handle);
  return shape.ValueOrDie();
}

XlaOprand XlaOprand::Constant(xla::Shape shape, DataType dtype) {
  XlaOprand op;
  op.shape_ = shape;
  op.dtype_ = dtype;
  op.initialized_ = true;
  return op;
}

XlaOprand XlaOprand::XlaOp(xla::XlaOp handle, DataType dtype) {
  XlaOprand op;
  op.handle_ = handle;
  op.dtype_ = dtype;
  op.shape_ = GetXlaOpShape(handle);
  op.initialized_ = true;
  return op;
}

xla::XlaOp XlaOprand::AsXlaOp(xla::XlaBuilder *builder) {
  CHECK(initialized_) << "XlaOprand has not been initialized.";
  if (handle_.IsUninitialized()) {
//    LiteralSlice literal;
//    HostTensorToBorrowingLiteral(constant_value_, &literal);
//    return xla::ConstantLiteral(builder, literal);
  }
  CHECK_EQ(builder, handle_.builder()) << "Mismatched builders in XlaOprand::AsXlaOp";
  return handle_;
}

xla::XlaBuilder *XlaOpContext::Builder() { return param_.builder; }

xla::XlaOp XlaOpContext::Input(const int index) {
  DCHECK_LT(index, param_.inputs.size());
  return param_.inputs[index].AsXlaOp(Builder());
}

xla::XlaOp XlaOpContext::Output(const int index) {
  DCHECK_LT(index, outputs_.size());
  return outputs_[index].AsXlaOp(Builder());
}

const XlaOprands &InputOprands() { return param_.inputs; }

const XlaOprands &OutputOprands() { return outputs_; }

int XlaOpContext::num_inputs() const { return param_.inputs.size(); }

int XlaOpContext::num_outputs() const { return param_.num_outputs; }

void XlaOpContext::SetOutput(const int index, const xla::XlaOp &handle) {
  SetOutput(index, XlaOprand::XlaOp(handle, OutputType(index)));
}

void XlaOpContext::SetOutput(const int index, const XlaOprand &handle) {
  DCHECK_LT(index, outputs_.size());
  outputs_[index] = handle;
}

const DataType XlaOpContext::InputType(const int index) {
  DCHECK_LT(index, param_.input_types.size());
  return param_.input_types[index];
}

const DataType XlaOpContext::OutputType(const int index) {
  DCHECK_LT(index, param_.output_types.size());
  return param_.output_types[index];
}

const xla::Shape XlaOpContext::InputShape(const int index) {
  xla::StatusOr<xla::Shape> shape_status = Builder()->GetShape(Input(index));
  return shape_status.ValueOrDie();
}

const xla::Shape XlaOpContext::OutputShape(const int index) {
  xla::StatusOr<xla::Shape> shape_status = Builder()->GetShape(Output(index));
  return shape_status.ValueOrDie();
}

}  // namespace mola
}  // namespace oneflow
