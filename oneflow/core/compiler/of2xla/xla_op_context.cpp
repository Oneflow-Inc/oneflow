#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/core/compiler/of2xla/xla_utility.h"
#include "oneflow/core/compiler/of2xla/xla_op_context.h"

namespace oneflow {
namespace mola {

Shape GetXlaOpShape(const xla::XlaOp &handle) {
  CHECK(handle.IsUninitialized()) << "XlaOp has not been initialized.";
  xla::StatusOr<xla::Shape> shape = handle.builder()->GetShape(handle);
  return ShapeFromXlaShape(shape.ValueOrDie());
}

XlaOprand XlaOprand::Constant(Shape shape, DataType dtype) {
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

xla::XlaOp XlaOpContext::Input(const std::string &name) {
  return Input(BlobIdFromString(name));
}

xla::XlaOp XlaOpContext::Output(const std::string &name) {
  return Output(BlobIdFromString(name));
}

xla::XlaOp XlaOpContext::Input(const LogicalBlobId &blob_id) {
  DCHECK_GT(param_.inputs.count(blob_id), 0);
  return param_.inputs[blob_id].AsXlaOp(Builder());
}

xla::XlaOp XlaOpContext::Output(const LogicalBlobId &blob_id) {
  DCHECK_GT(outputs_.count(blob_id), 0);
  return outputs_[blob_id].AsXlaOp(Builder());
}

int XlaOpContext::num_inputs() const { return param_.inputs.size(); }

int XlaOpContext::num_outputs() const { return param_.num_outputs; }

void XlaOpContext::SetOutput(const std::string &name, const xla::XlaOp &handle) {
  SetOutput(name, XlaOprand::XlaOp(handle, OutputType(name)));
}

void XlaOpContext::SetOutput(const std::string &name, const XlaOprand &handle) {
  LogicalBlobId blob_id = BlobIdFromString(name);
  DCHECK_EQ(outputs_.count(blob_id), 0);
  outputs_[blob_id] = handle;
}

const DataType XlaOpContext::InputType(const std::string &name) {
  return InputType(BlobIdFromString(name));
}

const DataType XlaOpContext::OutputType(const std::string &name) {
  return OutputType(BlobIdFromString(name));
}

const DataType XlaOpContext::InputType(const LogicalBlobId &blob_id) {
  DCHECK_GT(param_.inputs.count(blob_id), 0);
  return param_.inputs[blob_id].dtype_;
}

const DataType XlaOpContext::OutputType(const LogicalBlobId &blob_id) {
  DCHECK_GT(outputs_.count(blob_id), 0);
  return outputs_[blob_id].dtype_;
}

const Shape XlaOpContext::InputShape(const std::string &name) {
  return InputShape(BlobIdFromString(name));
}

const Shape XlaOpContext::InputShape(const LogicalBlobId &blob_id) {
  return GetXlaOpShape(Input(blob_id));
}

const Shape XlaOpContext::OutputShape(const std::string &name) {
  return OutputShape(BlobIdFromString(name));
}

const Shape XlaOpContext::OutputShape(const LogicalBlobId &blob_id) {
  return GetXlaOpShape(Output(blob_id));
}

const LogicalBlobId XlaOpContext::BlobIdFromString(const std::string &name) {
  return param_.blob_id_from_string_fn(name);
}

}  // namespace mola
}  // namespace oneflow
