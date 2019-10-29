#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_CONTEXT_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_CONTEXT_H_

#include <unordered_map>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/xrt/argument.h"
#include "oneflow/xrt/platform.h"
#include "oneflow/xrt/xrt.pb.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape.h"

namespace oneflow {
namespace xrt {
namespace mola {

using Argument = Argument;

class Operand {
 public:
  Operand() : initialized_(false) {}
  // Construct from Constant shape
  static Operand Constant(const xla::Shape shape);
  // Construct from XlaOp handle
  static Operand XlaOp(const xla::XlaOp handle);

  // Return the XlaOp handle if the builder is matched with the handle
  xla::XlaOp AsXlaOp(xla::XlaBuilder *builder) const;

 private:
  friend class OpContext;
  // XlaOp handle should be initialized if the oprand is constructed
  // from another XlaOp, otherwise uninitialized
  xla::XlaOp handle_;
  // Shape of the oprand
  xla::Shape shape_;

  bool initialized_;
};

class OpContext {
 public:
  typedef std::unordered_map<Argument, Operand> Operands;

  struct Param {
    XrtDevice backend;
    // Input oprands
    Operands inputs;
    int num_outputs;
    // Config proto related to the operator
    const PbMessage *op_conf;
    // XlaBuilder to compile the XlaComputation
    xla::XlaBuilder *builder;
    std::unordered_map<std::string, Argument> arguments;
  };

  explicit OpContext(const OpContext::Param &param) : param_(param) {}

  ~OpContext() = default;

  const XrtDevice &backend() const { return param_.backend; }

  // Return input named `name` as XlaOp
  xla::XlaOp Input(const std::string &name);
  xla::XlaOp Input(const Argument &arg);
  // Return output named `name` as XlaOp
  xla::XlaOp Output(const std::string &name);
  xla::XlaOp Output(const Argument &arg);

  int num_inputs() const { return param_.inputs.size(); }
  int num_outputs() const { return param_.num_outputs; }
  // Return inputs as Oprands
  const Operands &inputs() const { return param_.inputs; }
  // Return output as Oprands
  const Operands &outputs() const { return outputs_; }

  // Setup the output `output_name` with XlaOp
  void SetOutput(const std::string &name, const xla::XlaOp &handle);
  // Setup the output `output_name` with Operand
  void SetOutput(const std::string &name, const Operand &handle);

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
  OpContext() = delete;

  Argument ArgumentFromString(const std::string &name) const;

  // Output oprands
  Operands outputs_;
  Param param_;
};

template <typename T>
T OpContext::GetAttr(const std::string &attr_name) const {
  DCHECK(HasFieldInPbMessage(*param_.op_conf, attr_name));
  return GetValFromPbMessage<T>(*param_.op_conf, attr_name);
}

template <typename T>
void OpContext::SetAttr(const std::string &attr_name, const T &value) {
  SetValInPbMessage(const_cast<PbMessage *>(param_.op_conf), attr_name, value);
}

template <>
Shape OpContext::GetAttr<Shape>(const std::string &attr_name) const;

template <>
void OpContext::SetAttr<Shape>(const std::string &attr_name,
                               const Shape &value);
}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_CONTEXT_H_
