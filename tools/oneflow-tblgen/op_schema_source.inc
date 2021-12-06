R"OP_SCHEMA_INC(
{% if include != "" %}#include "{{ include }}"
{% else if filename != "" %}#include "{{ to_header(filename) }}"
{% endif %}{% if not header_only %}
namespace oneflow {

namespace user_op {

{% for name, op in ops %}
const HashSet<std::string> {{ name }}InterpCtx::attr_names {
    {% for name, attr in op.attrs %}"{{ name }}",
    {% endfor %}
};{% endfor %}

} // namespace user_op

} // namespace oneflow
{% endif %}
)OP_SCHEMA_INC"
