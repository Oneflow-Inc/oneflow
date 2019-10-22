#include "oneflow/xrt/xla/op_context.h"
#include "oneflow/xrt/graph/argument.h"
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
    //    LiteralSlice literal;
    //    HostTensorToBorrowingLiteral(constant_value_, &literal);
    //    return xla::ConstantLiteral(builder, literal);
  }
  CHECK_EQ(builder, handle_.builder())
      << "Mismatched builders in Operand::AsXlaOp";
  return handle_;
}

xla::XlaBuilder *OpContext::builder() const { return param_.builder; }

xla::XlaOp OpContext::Input(const std::string &name) {
  return Input(ArgumentFromString(name));
}

xla::XlaOp OpContext::Output(const std::string &name) {
  return Output(ArgumentFromString(name));
}

xla::XlaOp OpContext::Input(const Argument &arg) {
  DCHECK_GT(param_.inputs.count(arg), 0);
  return param_.inputs[arg].AsXlaOp(builder());
}

xla::XlaOp OpContext::Output(const Argument &arg) {
  DCHECK_GT(outputs_.count(arg), 0);
  return outputs_[arg].AsXlaOp(builder());
}

void OpContext::SetOutput(const std::string &name, const xla::XlaOp &handle) {
  Argument arg = ArgumentFromString(name);
  SetOutput(name, Operand::XlaOp(handle));
}

void OpContext::SetOutput(const std::string &name, const Operand &handle) {
  Argument arg = ArgumentFromString(name);
  CHECK_EQ(arg.shape().shape(), XlaShapeToOfShape(handle.shape_));
  outputs_[arg] = handle;
}

DataType OpContext::InputType(const std::string &name) const {
  return ArgumentFromString(name).shape().data_type();
}

DataType OpContext::OutputType(const std::string &name) const {
  return ArgumentFromString(name).shape().data_type();
}

Shape OpContext::InputShape(const std::string &name) const {
  return ArgumentFromString(name).shape().shape();
}

Shape OpContext::OutputShape(const std::string &name) const {
  return ArgumentFromString(name).shape().shape();
}

Argument OpContext::ArgumentFromString(const std::string &name) const {
  DCHECK_GT(param_.arguments.count(name), 0);
  return param_.arguments.at(name);
}

bool OpContext::HasAttr(const std::string &attr_name) const {
  using namespace google::protobuf;
  const Descriptor *d = param_.op_conf->GetDescriptor();
  const FieldDescriptor *fd = d->FindFieldByName(attr_name);
  if (fd && fd->is_optional()) {
    const Reflection *r = param_.op_conf->GetReflection();
    return r->HasField(*param_.op_conf, fd);
  }
  return fd != nullptr;
}

template <>
Shape OpContext::GetAttr<Shape>(const std::string &attr_name) const {
  DCHECK(HasFieldInPbMessage(*param_.op_conf, attr_name));
  return Shape(GetValFromPbMessage<ShapeProto>(*param_.op_conf, attr_name));
}

template <>
void OpContext::SetAttr<Shape>(const std::string &attr_name,
                               const Shape &value) {
  ShapeProto shape;
  value.ToProto(&shape);
  SetValInPbMessage<ShapeProto>(const_cast<PbMessage *>(param_.op_conf),
                                attr_name, shape);
}

std::string OpContext::AttrTypeInOneof(const std::string &oneof_name) const {
  using namespace google::protobuf;
  const Descriptor *d = param_.op_conf->GetDescriptor();
  const OneofDescriptor *ofd = d->FindOneofByName(oneof_name);
  const Reflection *r = param_.op_conf->GetReflection();

  CHECK(ofd) << "Message has no oneof field named " << oneof_name;
  const google::protobuf::FieldDescriptor *fd =
      r->GetOneofFieldDescriptor(*(param_.op_conf), ofd);
  return fd->name();
}

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow
