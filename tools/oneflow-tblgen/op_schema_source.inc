R"OP_SCHEMA_INC(
{% if include != "" %}#include "{{ include }}"
{% else if filename != "" %}#include "{{ to_header(filename) }}"
{% endif %}
#include "oneflow/core/common/auto_registration_factory.h"

namespace oneflow {

#define REGISTER_OP_INTERP_CTX(op_type, ctx) \
  REGISTER_CLASS_CREATOR(std::string, op_type, OpInterpCtx, ([]() { return new ctx; }))

{% for opname, op in ops %}
const HashSet<std::string>& {{opname}}InterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { {%- for name, attr in op.attrs -%}"{{name}}", {%- endfor -%} };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> {{opname}}::GetAttr(const std::string& attr_name) const {
  {% for name, attr in op.attrs %}if(attr_name == "{{name}}") {
    return CastAttr(&{{name}});
  }
  {% endfor -%}
  return Error::RuntimeError() << "{{op.name}} op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.{{op.name}}", {{opname}}InterpCtxImpl<schema::{{opname}}>);
{%- endfor %}
} // namespace oneflow
)OP_SCHEMA_INC"
