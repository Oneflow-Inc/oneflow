R"OP_SCHEMA_INC(
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/framework/op_definition.h"

#include <string>
#include <vector>
#include <functional>
#include <complex>

class OperatorConf;
class NdSbpSignature;

namespace oneflow {

class Device;
class Stream;
class InputBlobModifier;
class OutputBlobModifier;

namespace user_op {
class UserOpDefWrapper;
class UserOpConfWrapper;
class InferContext;
class SbpContext;
class InferSbpSignatureFnContext;
class InferOutputBlobTimeShapeFnContext;
class InferNdSbpFnContext;
class DeviceAndStreamInferContext;
class ComputeComplexityFnContext;
class GetNdSbpSignatureListContext;
}  // namespace user_op

using GetInputArgModifier =
    std::function<InputBlobModifier*(const std::string& in_arg_name, int32_t in_arg_index)>;
using GetOutputArgModifier =
    std::function<OutputBlobModifier*(const std::string& out_arg_name, int32_t out_arg_index)>;

{% for opname, op in ops %}
class {{opname}} : public OpDefinition<{{opname}}> {
 public:
  virtual ~{{opname}}() = default;
  {% if op.has_nd_sbp_infer_fn -%}
  static Maybe<void> InferNdSbp(user_op::InferNdSbpFnContext* ctx);
  {% endif -%}
  {% if op.has_get_sbp_fn -%}
  static Maybe<void> GetSbp(user_op::SbpContext* ctx);
  {% endif -%}
  {% if op.has_get_nd_sbp_fn -%}
  static Maybe<void> GetNdSbpSignatureList(user_op::GetNdSbpSignatureListContext* ctx);
  {% endif -%}
  {% if op.has_enumerate_nd_sbp_signatures_fn -%}
  static Maybe<void> EnumerateNdSbpSignatures(user_op::GetNdSbpSignatureListContext* ctx);
  {% endif -%}
  {% if op.has_dump_nd_sbp_signature_for_op_conf_fn -%}
  static Maybe<void> DumpNdSbpSignatureForOpConfFn(const NdSbpSignature& nd_sbp_sig, OperatorConf* op_conf);
  {% endif -%}
  {% if op.has_logical_tensor_desc_infer_fn -%}
  static Maybe<void> InferLogicalTensorDesc(user_op::InferContext* ctx);
  {% endif -%}
  {% if op.has_physical_tensor_desc_infer_fn -%}
  static Maybe<void> InferPhysicalTensorDesc(user_op::InferContext* ctx);
  {% endif -%}
  {% if op.has_data_type_infer_fn -%}
  static Maybe<void> InferDataType(user_op::InferContext* ctx);
  {% endif -%}
  {% if op.has_device_and_stream_infer_fn -%}
  static Maybe<Symbol<Stream>> InferDeviceAndStream(user_op::DeviceAndStreamInferContext* ctx);
  {% endif -%}
  {% if op.has_sbp_signature_infer_fn -%}
  static Maybe<void> InferSbpSignature(user_op::InferSbpSignatureFnContext* ctx);
  {% endif -%}
  {% if op.has_compute_complexity_fn -%}
  static Maybe<double> GetComputeComplexity(user_op::ComputeComplexityFnContext* ctx);
  {% endif -%}
  {% if op.has_input_arg_modify_fn -%}
  static Maybe<void> ModifyInputArg(const GetInputArgModifier&, const user_op::UserOpConfWrapper&);
  {% endif -%}
  {% if op.has_output_arg_modify_fn -%}
  static Maybe<void> ModifyOutputArg(const GetOutputArgModifier&, const user_op::UserOpConfWrapper&);
  {% endif -%}
  {% if op.has_output_blob_time_shape_infer_fn -%}
  static Maybe<void> InferOutputBlobTimeShape(user_op::InferOutputBlobTimeShapeFnContext* ctx);
  {% endif -%}
  {% if op.has_check_fn -%}
  static Maybe<void> CheckAttr(const user_op::UserOpDefWrapper&, const user_op::UserOpConfWrapper&);
  {% endif -%}

  {% for attr in op.attrs -%}
  virtual const {{attr.type}}& {{attr.name}}() const = 0;
  virtual {{attr.type}}* mutable_{{attr.name}}() = 0;
  virtual void set_{{attr.name}}(const {{attr.type}}& {{attr.name}}) = 0;

  {% endfor -%}
  static const HashSet<std::string>& AttrNames();
};

namespace schema {
class {{opname}} : public oneflow::{{opname}} {
 public:
  {% for attr in op.attrs -%}
  const {{attr.type}}& {{attr.name}}() const override { return {{attr.name}}_; }
  {{attr.type}}* mutable_{{attr.name}}() override { return &{{attr.name}}_; }
  void set_{{attr.name}}(const {{attr.type}}& {{attr.name}}) override { {{attr.name}}_ = {{attr.name}}; }

  {% endfor -%}

  Maybe<AttrVal> Attr(const std::string& attr_name) const override;

 private:
  {% for attr in op.attrs -%}
  {{attr.type}} {{attr.name}}_{% if existsIn(attr, "default") %} = {{attr.default}}{% endif %};
  {% endfor %}
};
}  // namespace schema
{% endfor %}
} // namespace oneflow
)OP_SCHEMA_INC"
