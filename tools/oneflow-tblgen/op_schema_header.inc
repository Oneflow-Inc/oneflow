R"OP_SCHEMA_INC(
#include "oneflow/core/framework/op_interp_ctx.h"

namespace oneflow {

namespace user_op {
{% for name, op in ops %}
struct {{name}}InterpCtx : public OpInterpCtx {
 public:
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    {% for name, attr in op.attrs %}if(attr_name == "{{name}}") {
      return CastAttr(&{{name}});
    }
    {% endfor %}
    return Error::RuntimeError() << "{{op.name}} op has no attribute named " << attr_name;
  }

  const HashSet<std::string>& AttrNamesSet() const override {
    {% if header_only %}static const HashSet<std::string> attr_names{
      {% for name, attr in op.attrs %}"{{name}}",
      {% endfor %}
    };
    {% endif %}
    return attr_names;
  }

  {% if not header_only %}static const HashSet<std::string> attr_names;
  {% endif %}

 public:
  {% for name, attr in op.attrs %}{{attr.type}} {{name}}{% if existsIn(attr, "default") %} = {{attr.default}}{% endif %};
  {% endfor %}
};{% endfor %}

} // namespace user_op

} // namespace oneflow
)OP_SCHEMA_INC"
