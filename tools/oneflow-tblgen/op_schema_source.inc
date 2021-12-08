R"OP_SCHEMA_INC(
{% if include != "" %}#include "{{ include }}"
{% else if filename != "" %}#include "{{ to_header(filename) }}"
{% endif %}{% if not header_only %}
namespace oneflow {

namespace user_op {

{% for name, op in ops %}
Maybe<AttrVal> {{ name }}InterpCtx::GetAttr(const std::string& attr_name) const {
  {% for n, attr in op.attrs %}if(attr_name == "{{ n }}") {
    return CastAttr(&{{ n }});
  }
  {% endfor %}
  return Error::RuntimeError() << "{{ op.name }} op has no attribute named " << attr_name;
}
const HashSet<std::string> {{ name }}InterpCtx::attr_names {
  {% for n, attr in op.attrs %}"{{ n }}",
  {% endfor %}
};{% endfor %}

} // namespace user_op

} // namespace oneflow
{% endif %}
)OP_SCHEMA_INC"
