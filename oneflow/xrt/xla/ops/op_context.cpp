#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/argument.h"
#include "oneflow/xrt/xla/xla_shape.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"

namespace oneflow {
namespace xrt {
namespace mola {

xla::Shape GetXlaOpShape(const xla::XlaOp &handle) {
  CHECK(!handle.IsUninitialized()) << "XlaOp has not been initialized.";
  xla::StatusOr<xla::Shape> shape = handle.builder()->GetShape(handle);
  return shape.ValueOrDie();
}

Operand Operand::Constant(xla::Shape shape) {
  Operand op;
  op.shape_ = shape;
  op.initialized_ = true;
  return op;
}

Operand Operand::XlaOp(xla::XlaOp handle) {
  Operand op;
  op.handle_ = handle;
  op.shape_ = GetXlaOpShape(handle);
  op.initialized_ = true;
  return op;
}

xla::XlaOp Operand::AsXlaOp(xla::XlaBuilder *builder) const {
  CHECK(initialized_) << "Operand has not been initialized.";
  if (handle_.IsUninitialized()) {
    // LiteralSlice literal;
    // HostTensorToBorrowingLiteral(constant_value_, &literal);
    // return xla::ConstantLiteral(builder, literal);
  }
  CHECK_EQ(builder, handle_.builder())
      << "Mismatched builders in Operand::AsXlaOp";
  return handle_;
}

xla::XlaBuilder *OpContext::builder() const { return param_.builder; }

xla::XlaOp OpContext::Input(const std::string &name) {
  return Input(ArgumentFromKey(name));
}

xla::XlaOp OpContext::Output(const std::string &name) {
  return Output(ArgumentFromKey(name));
}

xla::XlaOp OpContext::Input(const Argument &arg) {
  // DCHECK_GT(param_.inputs.count(arg), 0);
  return param_.inputs.at(arg).AsXlaOp(builder());
}

xla::XlaOp OpContext::Output(const Argument &arg) {
  // DCHECK_GT(outputs_.count(arg), 0);
  return outputs_.at(arg).AsXlaOp(builder());
}

void OpContext::SetOutput(const std::string &name, const xla::XlaOp &handle) {
  SetOutput(name, Operand::XlaOp(handle));
}

void OpContext::SetOutput(const std::string &name, const Operand &handle) {
  Argument arg = ArgumentFromKey(name);
  CHECK_EQ(arg.shape(), XlaShapeToOfShape(handle.shape_));
  outputs_[arg] = handle;
}

DataType OpContext::InputType(const std::string &name) const {
  return ArgumentFromKey(name).data_type();
}

DataType OpContext::OutputType(const std::string &name) const {
  return ArgumentFromKey(name).data_type();
}

Shape OpContext::InputShape(const std::string &name) const {
  return ArgumentFromKey(name).shape();
}

Shape OpContext::OutputShape(const std::string &name) const {
  return ArgumentFromKey(name).shape();
}

Argument OpContext::ArgumentFromKey(const std::string &key) const {
  DCHECK_GT(param_.arguments.count(key), 0);
  return param_.arguments.at(key);
}

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
