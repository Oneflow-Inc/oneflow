#ifndef ONEFLOW_XRT_XLA_OPS_OP_CONTEXT_H_
#define ONEFLOW_XRT_XLA_OPS_OP_CONTEXT_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/xrt/argument.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/message_attr.h"
#include "oneflow/xrt/utility/stl.h"
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
  // Construct from Constant shape.
  static Operand Constant(const xla::Shape shape);
  // Construct from XlaOp handle.
  static Operand XlaOp(const xla::XlaOp handle);

  // Return the XlaOp handle if the builder is matched with the handle.
  xla::XlaOp AsXlaOp(xla::XlaBuilder *builder) const;

  friend class OpContext;

 private:
  bool initialized_;
  // XlaOp handle should be initialized if the operand is
  // constructed from another XlaOp.
  xla::XlaOp handle_;
  // Shape of the operand.
  xla::Shape shape_;
};

class OpContext {
 public:
  struct Param {
    // XlaBuilder to compile the XlaComputation
    xla::XlaBuilder *builder;

    XrtDevice backend;
    // Config proto related to the operator
    const PbMessage *message;
    // Input oprands
    util::Map<Argument, Operand> inputs;
    int num_outputs;

    util::Map<std::string, Argument> arguments;
  };

  explicit OpContext(const Param &param) : param_(param) {}

  virtual ~OpContext() = default;

  const XrtDevice &backend() const { return param_.backend; }
  // Return XlaBuilder
  xla::XlaBuilder *builder() const;

  // Return input named `name` as XlaOp
  xla::XlaOp Input(const std::string &name);
  xla::XlaOp Input(const Argument &arg);
  // Return output named `name` as XlaOp
  xla::XlaOp Output(const std::string &name);
  xla::XlaOp Output(const Argument &arg);

  int num_inputs() const { return param_.inputs.size(); }
  int num_outputs() const { return param_.num_outputs; }
  // Return inputs as Operands
  const util::Map<Argument, Operand> &inputs() const { return param_.inputs; }
  // Return output as Operands
  const util::Map<Argument, Operand> &outputs() const { return outputs_; }

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
  T GetAttr(const std::string &attr_name) const {
    T value;
    util::GetAttr<T>(*param_.message, attr_name, &value);
    return std::move(value);
  }

  template <typename T>
  void SetAttr(const std::string &attr_name, const T &value) {
    util::SetAttr<T>(const_cast<PbMessage *>(param_.message), attr_name, value);
  }

  bool HasAttr(const std::string &attr_name) const {
    return util::HasAttr(*param_.message, attr_name);
  }

  std::string GetOneofType(const std::string &oneof_name) const {
    std::string oneof_type;
    util::GetOneofType(*param_.message, oneof_name, &oneof_type);
    return std::move(oneof_type);
  }

 private:
  OpContext() = delete;
  Argument ArgumentFromKey(const std::string &key) const;

  Param param_;
  // Output oprands
  util::Map<Argument, Operand> outputs_;
};

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_OPS_OP_CONTEXT_H_
