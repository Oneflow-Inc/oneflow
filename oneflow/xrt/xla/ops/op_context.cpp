#include "oneflow/xrt/xla/ops/op_context.h"

#include "oneflow/xrt/argument.h"
#include "oneflow/xrt/xla/xla_data_type.h"
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

XlaValue XlaValue::Constant(xla::Shape shape) {
  XlaValue op;
  op.shape_ = shape;
  op.initialized_ = true;
  return op;
}

XlaValue XlaValue::XlaOp(xla::XlaOp handle) {
  XlaValue op;
  op.handle_ = handle;
  op.shape_ = GetXlaOpShape(handle);
  op.initialized_ = true;
  return op;
}

xla::XlaOp XlaValue::AsXlaOp(xla::XlaBuilder *builder) const {
  CHECK(initialized_) << "XlaValue has not been initialized.";
  if (handle_.IsUninitialized()) {
    // LiteralSlice literal;
    // HostTensorToBorrowingLiteral(constant_value_, &literal);
    // return xla::ConstantLiteral(builder, literal);
  }
  CHECK_EQ(builder, handle_.builder()) << "Mismatched builders in XlaValue::AsXlaOp";
  return handle_;
}

xla::XlaBuilder *XlaOpContext::builder() const { return param_.builder; }

bool XlaOpContext::HasInput(const std::string &name) const {
  return param_.arguments.count(name) > 0 && param_.inputs.count(ArgumentFromKey(name)) > 0;
}

bool XlaOpContext::HasOutput(const std::string &name) const {
  return param_.arguments.count(name) > 0;
}

xla::XlaOp XlaOpContext::Input(const std::string &name) { return Input(ArgumentFromKey(name)); }

xla::XlaOp XlaOpContext::Output(const std::string &name) { return Output(ArgumentFromKey(name)); }

xla::XlaOp XlaOpContext::Input(const Argument &arg) {
  CHECK_GT(param_.inputs.count(arg), 0);
  return param_.inputs.at(arg).AsXlaOp(builder());
}

xla::XlaOp XlaOpContext::Output(const Argument &arg) {
  CHECK_GT(outputs_.count(arg), 0);
  return outputs_.at(arg).AsXlaOp(builder());
}

void XlaOpContext::SetOutput(const std::string &name, const xla::XlaOp &handle) {
  SetOutput(name, XlaValue::XlaOp(handle));
}

void XlaOpContext::SetOutput(const std::string &name, const XlaValue &handle) {
  Argument arg = ArgumentFromKey(name);
  CHECK_EQ(arg.shape(), XlaShapeToOfShape(handle.shape_));
  CHECK_EQ(DataTypeToPrimitiveType(arg.data_type()), handle.shape_.element_type());
  // CHECK_EQ(OfShapeToXlaShape(arg.shape(), arg.data_type()), handle.shape_);
  outputs_[arg] = handle;
}

DataType XlaOpContext::InputType(const std::string &name) const {
  return ArgumentFromKey(name).data_type();
}

DataType XlaOpContext::OutputType(const std::string &name) const {
  return ArgumentFromKey(name).data_type();
}

Shape XlaOpContext::InputShape(const std::string &name) const {
  return ArgumentFromKey(name).shape();
}

Shape XlaOpContext::OutputShape(const std::string &name) const {
  return ArgumentFromKey(name).shape();
}

Argument XlaOpContext::ArgumentFromKey(const std::string &key) const {
  CHECK_GT(param_.arguments.count(key), 0);
  return param_.arguments.at(key);
}

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
