#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_CONTEXT_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_CONTEXT_H_

#include <unordered_map>
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/xla/of2xla/xla_argument.h"

namespace oneflow {
namespace mola {

class XlaOprand {
 public:
  XlaOprand() : initialized_(false) {}
  // Construct from Constant shape
  static XlaOprand Constant(const xla::Shape shape);
  // Construct from XlaOp handle
  static XlaOprand XlaOp(const xla::XlaOp handle);

  // Return the XlaOp handle if the builder is matched with the handle
  xla::XlaOp AsXlaOp(xla::XlaBuilder *builder) const;

 private:
  friend class XlaOpContext;
  // XlaOp handle should be initialized if the oprand is constructed
  // from another XlaOp, otherwise uninitialized
  xla::XlaOp handle_;
  // Shape of the oprand
  xla::Shape shape_;

  bool initialized_;
};

class XlaOpContext {
 public:
  typedef std::unordered_map<Argument, XlaOprand> XlaOprands;

  struct Param {
    std::string backend;
    // Input oprands
    XlaOprands inputs;
    int num_outputs;
    // Config proto related to the operator
    const PbMessage *op_conf;
    // XlaBuilder to compile the XlaComputation
    xla::XlaBuilder *builder;
    std::unordered_map<std::string, Argument> arguments;
  };

  explicit XlaOpContext(const XlaOpContext::Param &param) : param_(param) {}

  ~XlaOpContext() = default;

  const std::string &backend() const { return param_.backend; }

  // Return input named `name` as XlaOp
  xla::XlaOp Input(const std::string &name);
  xla::XlaOp Input(const Argument &arg);
  // Return output named `name` as XlaOp
  xla::XlaOp Output(const std::string &name);
  xla::XlaOp Output(const Argument &arg);

  int num_inputs() const { return param_.inputs.size(); }
  int num_outputs() const { return param_.num_outputs; }
  // Return inputs as Oprands
  const XlaOprands &inputs() const { return param_.inputs; }
  // Return output as Oprands
  const XlaOprands &outputs() const { return outputs_; }

  // Setup the output `output_name` with XlaOp
  void SetOutput(const std::string &name, const xla::XlaOp &handle);
  // Setup the output `output_name` with XlaOprand
  void SetOutput(const std::string &name, const XlaOprand &handle);

  // Return input `name` shape as Shape
  Shape InputShape(const std::string &name) const;
  // Return output `name` shape as Shape
  Shape OutputShape(const std::string &name) const;
 
  // Input data type
  DataType InputType(const std::string &name) const;
  // Output data type
  DataType OutputType(const std::string &name) const;

  const Param &param() const { return param_; }

  template <typename T>
  T GetAttr(const std::string &attr_name) const;

  template <typename T>
  void SetAttr(const std::string &attr_name, const T &value);

  bool HasAttr(const std::string &attr_name) const;

  std::string AttrTypeInOneof(const std::string &oneof_name) const;

  // Return XlaBuilder
  xla::XlaBuilder *builder() const;

 private:
  XlaOpContext() = delete;

  Argument ArgumentFromString(const std::string &name) const;

  // Output oprands
  XlaOprands outputs_;
  Param param_;
};

template <typename T>
T XlaOpContext::GetAttr(const std::string &attr_name) const {
  DCHECK(HasFieldInPbMessage(*param_.op_conf, attr_name));
  return GetValFromPbMessage<T>(*param_.op_conf, attr_name);
}

template <typename T>
void XlaOpContext::SetAttr(const std::string &attr_name, const T &value) {
  SetValInPbMessage(const_cast<PbMessage *>(param_.op_conf), attr_name, value);
}

template <>
Shape XlaOpContext::GetAttr<Shape>(const std::string &attr_name) const;

template <>
void XlaOpContext::SetAttr<Shape>(const std::string &attr_name,
                                  const Shape &value);
}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_CONTEXT_H_
