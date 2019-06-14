#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_CONTEXT_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_CONTEXT_H_

#include "oneflow/core/common/protobuf.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace oneflow {
namespace mola {

enum class DataType {
  DT_INVALID,
  DT_FLOAT,
  DT_INT32,
  DT_STRING,
};

class XlaOprand {
 public:
  XlaOprand() : dtype_(DataType::DT_INVALID), initialized_(false) {}
  // Construct from Constant shape
  static XlaOprand Constant(const xla::Shape shape, const DataType dtype);
  // Construct from XlaOp handle
  static XlaOprand XlaOp(const xla::XlaOp handle, const DataType dtype);

  // Return the XlaOp handle if the builder is matched with the handle
  xla::XlaOp AsXlaOp(xla::XlaBuilder *builder);

 private:
  // XlaOp handle should be initialized if the oprand is constructed
  // from another XlaOp, otherwise uninitialized
  xla::XlaOp handle_;
  // Shape of the oprand
  xla::Shape shape_;
  // Data type of the oprand
  DataType dtype_;

  bool initialized_;
};

class XlaOpContext {
 public:
  typedef std::vector<XlaOprand> XlaOprands;
  typedef std::vector<DataType> DataTypes;

  struct Param {
    // Input oprands
    XlaOprands inputs;
    // Input data types
    DataTypes input_types;
    // Output data types, it's size must equal to num_outputs
    DataTypes output_types;
    // The number of the operator's outputs
    int num_outputs;
    // Config proto related to the operator
    PbMessage *op_conf;
    // XlaBuilder to compile the XlaComputation
    xla::XlaBuilder *builder;
  };

  explicit XlaOpContext(const XlaOpContext::Param &param) : param_(param) {
    outputs_.resize(param.num_outputs);
  }

  ~XlaOpContext() = default;

  // Return input as XlaOp
  xla::XlaOp Input(const int index);
  // Return output as XlaOp
  xla::XlaOp Output(const int index);

  // Return inputs as Oprands
  const XlaOprands &InputOprands();
  // Return output as Oprands
  const XlaOprands &OutputOprands();

  int num_inputs() const;
  int num_outputs() const;

  // Setup the output `index` with XlaOp
  void SetOutput(const int index, const xla::XlaOp &handle);
  // Setup the output `index` with XlaOprand
  void SetOutput(const int index, const XlaOprand &handle);

  // Return input `index` shape as Shape
  const xla::Shape InputShape(const int index);
  // Return output `index` shape as Shape
  const xla::Shape OutputShape(const int index);

  // Data type
  const DataType InputType(const int index);
  const DataType OutputType(const int index);

  template <typename T>
  T GetAttr(const std::string &attr_name);

  template <typename T>
  void SetAttr(const std::string &attr_name, const T &value);

  // Return XlaBuilder
  xla::XlaBuilder *Builder();

 private:
  XlaOpContext() = delete;

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
  SetValInPbMessage(param_.op_conf, attr_name, value);
}

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_CONTEXT_H_
