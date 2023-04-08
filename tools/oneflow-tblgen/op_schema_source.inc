R"OP_SCHEMA_INC(
{% if include != "" %}#include "{{ include }}"
{% else if filename != "" %}#include "{{ to_header(filename) }}"
{% endif %}
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/infer_nd_sbp_fn_context.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include <complex>

namespace oneflow {

#define REGISTER_OP_SCHEMA(op_type, schema) \
  REGISTER_CLASS_CREATOR(std::string, op_type, OpDefinitionBase, ([]() { return new schema; }))

{% for opname, op in ops %}
/*static*/ const HashSet<std::string>& {{opname}}::AttrNames() {
  static const HashSet<std::string> attr_names = { {%- for attr in op.attrs -%}"{{attr.name}}", {%- endfor -%} };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> {{opname}}::Attr(const std::string& attr_name) const {
  {% for attr in op.attrs %}if(attr_name == "{{attr.name}}") {
    return CastAttrValue(&{{attr.name}}_);
  }
  {% endfor -%}
  return Error::RuntimeError() << "{{op.name}} op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_SCHEMA("user.{{op.name}}", schema::{{opname}});

REGISTER_USER_OP("{{op.name}}")
{%- if op.input -%}
{%- for input in op.input -%}
{%- if input.is_optional -%}
    .OptionalInput("{{input.name}}")
{%- else -%}
    .Input("{{input.name}}")
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{%- if op.output -%}
{%- for output in op.output -%}
{%- if output.is_optional -%}
    .OptionalOutput("{{output.name}}")
{%- else -%}
    .Output("{{output.name}}")
{%- endif -%}
{%- endfor -%}
{%- endif -%}

{%- for attr in op.attrs -%}
{%- if existsIn(attr, "default") -%}
    .Attr<{{attr.type}}>("{{attr.name}}", {{attr.default}})
{%- else -%}
    .Attr<{{attr.type}}>("{{attr.name}}")
{%- endif -%}
{%- endfor -%}
{%- if op.cpu_only -%}
    .SupportCpuOnly()
{%- endif -%}
{%- if op.no_grad -%}
    .NoGrad()
{%- endif -%}
{%- if op.support_non_contiguous -%}
    .SupportNonContiguous()
{%- endif -%}
{%- if op.same_output_regst_num != -1 -%}
    .SetOutputBufferNum({{op.same_output_regst_num}})
{%- endif -%}
{%- if op.has_nd_sbp_infer_fn -%}
    .SetNdSbpInferFn(&{{opname}}::InferNdSbp)
{%- endif -%}
{%- if op.has_get_sbp_fn -%}
    .SetGetSbpFn(&{{opname}}::GetSbp)
{%- endif -%}
{%- if op.has_get_nd_sbp_fn -%}
    .SetGetNdSbpSignatureListFn(&{{opname}}::GetNdSbpSignatureList)
{%- endif -%}
{%- if op.has_enumerate_nd_sbp_signatures_fn -%}
    .SetEnumerateNdSbpSignaturesFn(&{{opname}}::EnumerateNdSbpSignatures)
{%- endif -%}
{%- if op.has_dump_nd_sbp_signature_for_op_conf_fn -%}
    .SetDumpNdSbpSignatureForOpConfFn(&{{opname}}::DumpNdSbpSignatureForOpConfFn)
{%- endif -%}
{%- if op.has_compute_complexity_fn -%}
    .SetComputeComplexityFn(&{{opname}}::GetComputeComplexity)
{%- endif -%}
{%- if op.has_logical_tensor_desc_infer_fn -%}
    .SetLogicalTensorDescInferFn(&{{opname}}::InferLogicalTensorDesc)
{%- endif -%}
{%- if op.has_physical_tensor_desc_infer_fn -%}
    .SetPhysicalTensorDescInferFn(&{{opname}}::InferPhysicalTensorDesc)
{%- endif -%}
{%- if op.has_data_type_infer_fn -%}
    .SetDataTypeInferFn(&{{opname}}::InferDataType)
{%- endif -%}
{%- if op.has_device_and_stream_infer_fn -%}
    .SetDeviceAndStreamInferFn(&{{opname}}::InferDeviceAndStream)
{%- endif -%}
{%- if op.has_sbp_signature_infer_fn -%}
    .SetSbpSignatureInferFn(&{{opname}}::InferSbpSignature)
{% endif -%}
{%- if op.has_input_arg_modify_fn -%}
    .SetInputArgModifyFn(&{{opname}}::ModifyInputArg)
{%- endif -%}
{%- if op.has_output_arg_modify_fn -%}
    .SetOutputArgModifyFn(&{{opname}}::ModifyOutputArg)
{%- endif -%}
{%- if op.has_output_blob_time_shape_infer_fn -%}
    .SetOutputBlobTimeShapeInferFn(&{{opname}}::InferOutputBlobTimeShape)
{%- endif -%}
{%- if op.has_check_fn -%}
    .SetCheckAttrFn(&{{opname}}::CheckAttr)
{%- endif -%}
;
{%- endfor %}
} // namespace oneflow
)OP_SCHEMA_INC"
