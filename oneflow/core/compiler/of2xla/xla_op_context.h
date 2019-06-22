#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_CONTEXT_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_CONTEXT_H_

#include <unordered_map>
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/compiler/of2xla/xla_argument.h"

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
  typedef std::unordered_map<Argument, XlaOprand> XlaOprands;

  struct Param {
    // Input oprands
    XlaOprands inputs;
    // Config proto related to the operator
    const PbMessage *op_conf;
    // XlaBuilder to compile the XlaComputation
    xla::XlaBuilder *builder;
    // Function that convert argument name to Argument
    std::function<Argument(const std::string &)> argument_from_string_fn;
  };

  explicit XlaOpContext(const XlaOpContext::Param &param) : param_(param) {}

  ~XlaOpContext() = default;

  // Return input named `name` as XlaOp
  xla::XlaOp Input(const std::string &name);
  xla::XlaOp Input(const Argument &arg);
  // Return output named `name` as XlaOp
  xla::XlaOp Output(const std::string &name);
  xla::XlaOp Output(const Argument &arg);

  // Return inputs as Oprands
  const XlaOprands &InputOprands() { return param_.inputs; }
  // Return output as Oprands
  const XlaOprands &OutputOprands() { return outputs_; }

  // Setup the output `output_name` with XlaOp
  void SetOutput(const std::string &name, const xla::XlaOp &handle);
  // Setup the output `output_name` with XlaOprand
  void SetOutput(const std::string &name, const XlaOprand &handle);

  // Return input `name` shape as Shape
  const Shape InputShape(const std::string &name);
  // Return output `name` shape as Shape
  const Shape OutputShape(const std::string &name);
 
  // Input data type
  const DataType InputType(const std::string &name);
  // Output data type
  const DataType OutputType(const std::string &name);

  template <typename T>
  T GetAttr(const std::string &attr_name);

  template <typename T>
  void SetAttr(const std::string &attr_name, const T &value);

  // Return XlaBuilder
  xla::XlaBuilder *Builder();

 private:
  XlaOpContext() = delete;

  const Argument ArgumentFromString(const std::string &name);

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
