R"OP_SCHEMA_INC(
#include "oneflow/core/framework/op_interp_ctx.h"

namespace oneflow {

namespace user_op {
{% for name, op in ops %}
struct {{name}}InterpCtx : OpInterpCtx {
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override{% if header_only %} {
    {% for n, attr in op.attrs %}if(attr_name == "{{ n }}") {
      return CastAttr(&{{ n }});
    }
    {% endfor %}
    return Error::RuntimeError() << "{{ op.name }} op has no attribute named " << attr_name;
  }{% else %};{% endif %}

  const HashSet<std::string>& AttrNamesSet() const override {
    {% if header_only %}static const HashSet<std::string> attr_names{
      {% for n, attr in op.attrs %}"{{ n }}",
      {% endfor %}
    };
    {% endif %}return attr_names;
  }

  {% if not header_only %}static const HashSet<std::string> attr_names;
  {% endif %}
  {% for name, attr in op.attrs %}{{attr.type}} {{name}}{% if existsIn(attr, "default") %} = {{attr.default}}{% endif %};
  {% endfor %}
};{% endfor %}

} // namespace user_op

} // namespace oneflow
)OP_SCHEMA_INC"
