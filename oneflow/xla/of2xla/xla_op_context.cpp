#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/xla/of2xla/xla_shape.h"
#include "oneflow/xla/of2xla/xla_utility.h"
#include "oneflow/xla/of2xla/xla_argument.h"
#include "oneflow/xla/of2xla/xla_op_context.h"

namespace oneflow {
namespace mola {

xla::Shape GetXlaOpShape(const xla::XlaOp &handle) {
  CHECK(!handle.IsUninitialized()) << "XlaOp has not been initialized.";
  xla::StatusOr<xla::Shape> shape = handle.builder()->GetShape(handle);
  return shape.ValueOrDie();
}

XlaOprand XlaOprand::Constant(xla::Shape shape) {
  XlaOprand op;
  op.shape_ = shape;
  op.initialized_ = true;
  return op;
}

XlaOprand XlaOprand::XlaOp(xla::XlaOp handle) {
  XlaOprand op;
  op.handle_ = handle;
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
  CHECK_EQ(builder, handle_.builder())
      << "Mismatched builders in XlaOprand::AsXlaOp";
  return handle_;
}

xla::XlaBuilder *XlaOpContext::builder() const { return param_.builder; }

xla::XlaOp XlaOpContext::Input(const std::string &name) {
  return Input(ArgumentFromString(name));
}

xla::XlaOp XlaOpContext::Output(const std::string &name) {
  return Output(ArgumentFromString(name));
}

xla::XlaOp XlaOpContext::Input(const Argument &arg) {
  DCHECK_GT(param_.inputs.count(arg), 0);
  return param_.inputs[arg].AsXlaOp(builder());
}

xla::XlaOp XlaOpContext::Output(const Argument &arg) {
  DCHECK_GT(outputs_.count(arg), 0);
  return outputs_[arg].AsXlaOp(builder());
}

void XlaOpContext::SetOutput(const std::string &name,
                             const xla::XlaOp &handle) {
  Argument arg = ArgumentFromString(name);
  SetOutput(name, XlaOprand::XlaOp(handle));
}

void XlaOpContext::SetOutput(const std::string &name, const XlaOprand &handle) {
  Argument arg = ArgumentFromString(name);
  CHECK_EQ(arg.shape(), XlaShapeToOfShape(handle.shape_));
  outputs_[arg] = handle;
}

DataType XlaOpContext::InputType(const std::string &name) const {
  return ArgumentFromString(name).data_type();
}

DataType XlaOpContext::OutputType(const std::string &name) const {
  return ArgumentFromString(name).data_type();
}

Shape XlaOpContext::InputShape(const std::string &name) const {
  return ArgumentFromString(name).shape();
}

Shape XlaOpContext::OutputShape(const std::string &name) const {
  return ArgumentFromString(name).shape();
}

Argument XlaOpContext::ArgumentFromString(const std::string &name) const {
  return param_.arguments.at(name);
}

template <>
Shape XlaOpContext::GetAttr<Shape>(const std::string &attr_name) const {
  DCHECK(HasFieldInPbMessage(*param_.op_conf, attr_name));
  return Shape(GetValFromPbMessage<ShapeProto>(*param_.op_conf, attr_name));
}

template <>
void XlaOpContext::SetAttr<Shape>(const std::string &attr_name,
                                  const Shape &value) {
  ShapeProto shape;
  value.ToProto(&shape);
  SetValInPbMessage<ShapeProto>(const_cast<PbMessage *>(param_.op_conf),
                                attr_name, shape);
}

bool XlaOpContext::HasAttr(const std::string &attr_name) const {
  return HasFieldInPbMessage(*param_.op_conf, attr_name);
}

}  // namespace mola
}  // namespace oneflow
