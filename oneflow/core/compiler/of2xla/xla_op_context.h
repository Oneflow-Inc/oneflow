#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_CONTEXT_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_CONTEXT_H_

#include <unordered_map>
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace mola {

class XlaOprand {
 public:
  XlaOprand() : dtype_(DataType::kInvalidDataType), initialized_(false) {}
  // Construct from Constant shape
  static XlaOprand Constant(const Shape shape, const DataType dtype);
  // Construct from XlaOp handle
  static XlaOprand XlaOp(const xla::XlaOp handle, const DataType dtype);

  // Return the XlaOp handle if the builder is matched with the handle
  xla::XlaOp AsXlaOp(xla::XlaBuilder *builder);

 private:
  friend class XlaOpContext;
  // XlaOp handle should be initialized if the oprand is constructed
  // from another XlaOp, otherwise uninitialized
  xla::XlaOp handle_;
  // Shape of the oprand
  Shape shape_;
  // Data type of the oprand
  DataType dtype_;

  bool initialized_;
};

class XlaOpContext {
 public:
  typedef std::unordered_map<LogicalBlobId, XlaOprand> XlaOprands;
  typedef std::unordered_map<LogicalBlobId, DataType> DataTypes;

  struct Param {
    // Input oprands
    XlaOprands inputs;
    // Output data types, it's size must equal to num_outputs
    DataTypes output_types;
    // The number of the operator's outputs
    int num_outputs;
    // Config proto related to the operator
    const PbMessage *op_conf;
    // XlaBuilder to compile the XlaComputation
    xla::XlaBuilder *builder;
    // Function that convert blob name to blob id
    std::function<LogicalBlobId(const std::string &)> blob_id_from_string_fn;
  };

  explicit XlaOpContext(const XlaOpContext::Param &param) : param_(param) {}

  ~XlaOpContext() = default;

  // Return input named `name` as XlaOp
  xla::XlaOp Input(const std::string &name);
  // Return input which blob id is `blob_id` as XlaOp
  xla::XlaOp Input(const LogicalBlobId &blob_id);
  // Return output named `name` as XlaOp
  xla::XlaOp Output(const std::string &name);
  // Return output which blob id is `blob_id` as XlaOp
  xla::XlaOp Output(const LogicalBlobId &blob_id);

  // Return inputs as Oprands
  const XlaOprands &InputOprands() { return param_.inputs; }
  // Return output as Oprands
  const XlaOprands &OutputOprands() { return outputs_; }

  int num_inputs() const;
  int num_outputs() const;

  // Setup the output `output_name` with XlaOp
  void SetOutput(const std::string &name, const xla::XlaOp &handle);
  // Setup the output `output_name` with XlaOprand
  void SetOutput(const std::string &name, const XlaOprand &handle);

  // Return input `name` shape as Shape
  const Shape InputShape(const std::string &name);
  // Return input which blob id is `blob_id` shape as Shape
  const Shape InputShape(const LogicalBlobId &blob_id);
  // Return output `name` shape as Shape
  const Shape OutputShape(const std::string &name);
  // Return output shape as Shape which blob id is `blob_id`
  const Shape OutputShape(const LogicalBlobId &blob_id);
 
  // Input data type
  const DataType InputType(const std::string &name);
  const DataType InputType(const LogicalBlobId &blob_id);
  // Output data type
  const DataType OutputType(const std::string &name);
  const DataType OutputType(const LogicalBlobId &blob_id);

  template <typename T>
  T GetAttr(const std::string &attr_name);

  template <typename T>
  void SetAttr(const std::string &attr_name, const T &value);

  // Return XlaBuilder
  xla::XlaBuilder *Builder();

 private:
  XlaOpContext() = delete;

  const LogicalBlobId BlobIdFromString(const std::string &name);

  // Output oprands
  XlaOprands outputs_;
  Param param_;
};

template <typename T>
T XlaOpContext::GetAttr(const std::string &attr_name) {
  DCHECK(HasFieldInPbMessage(*param_.op_conf, attr_name));
  return GetValFromPbMessage<T>(*param_.op_conf, attr_name);
}

template <typename T>
void XlaOpContext::SetAttr(const std::string &attr_name, const T &value) {
  SetValInPbMessage(const_cast<PbMessage *>(param_.op_conf), attr_name, value);
}

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_CONTEXT_H_
