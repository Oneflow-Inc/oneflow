R"OP_SCHEMA_INC(
{% if include != "" %}#include "{{ include }}"
{% else if filename != "" %}#include "{{ to_header(filename) }}"
{% endif %}
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/infer_nd_sbp_fn_context.h"
#include "oneflow/core/framework/user_op_registry_manager.h"

namespace oneflow {

#define REGISTER_OP_SCHEMA(op_type, schema) \
  REGISTER_CLASS_CREATOR(std::string, op_type, OpBase, ([]() { return new schema; }))

{% for opname, op in ops %}
const HashSet<std::string>& {{opname}}::AttrNames() const {
  static const HashSet<std::string> attr_names = { {%- for name, attr in op.attrs -%}"{{name}}", {%- endfor -%} };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> {{opname}}::GetAttr(const std::string& attr_name) const {
  {% for name, attr in op.attrs %}if(attr_name == "{{name}}") {
    return CastAttrValue(&{{name}}_);
  }
  {% endfor -%}
  return Error::RuntimeError() << "{{op.name}} op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_SCHEMA("user.{{op.name}}", schema::{{opname}});

// REGISTER_USER_OP("{{op.name}}")
{%- if op.input -%}
{%- for name, input in op.input -%}
{%- if input.is_optional -%}
//     .OptionalInput("{{name}}")
{%- else -%}
//     .Input("{{name}}")
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{%- if op.output -%}
{%- for name, output in op.output -%}
{%- if output.is_optional -%}
//     .OptionalOutput("{{name}}")
{%- else -%}
//     .Output("{{name}}")
{%- endif -%}
{%- endfor -%}
{%- endif -%}

{%- for name, attr in op.attrs -%}
{%- if existsIn(attr, "default") -%}
//     .Attr<{{attr.type}}>("{{name}}", {{attr.default}})
{%- else -%}
//     .Attr<{{attr.type}}>("{{name}}")
{%- endif -%}
{%- endfor -%}

{%- if op.infer_nd_sbp -%}
//     .SetNdSbpInferFn([](user_op::InferNdSbpFnContext* ctx) { {{op.infer_nd_sbp.code}} })
{%- endif -%}

{%- endfor %}
} // namespace oneflow
)OP_SCHEMA_INC"
