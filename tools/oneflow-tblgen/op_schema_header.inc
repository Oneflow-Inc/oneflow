R"OP_SCHEMA_INC(
#include <string>
#include <vector>

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/op_base.h"

namespace oneflow {
namespace user_op {
class InferNdSbpFnContext;
}  // namespace user_op

{% for opname, op in ops %}
class {{opname}} : public OpBase {
 public:
  virtual ~{{opname}}() = default;
  {% if op.infer_nd_sbp %}
  static Maybe<void> InferNdSbp(user_op::InferNdSbpFnContext* ctx);

  {% endif -%}

  {% for name, attr in op.attrs -%}
  virtual const {{attr.type}}& {{name}}() const = 0;
  virtual {{attr.type}}* mutable_{{name}}() = 0;
  virtual void set_{{name}}(const {{attr.type}}& {{name}}) = 0;

  {% endfor -%}
  const HashSet<std::string>& AttrNames() const;
};

namespace schema {
class {{opname}} : public oneflow::{{opname}} {
 public:
  {% for name, attr in op.attrs -%}
  const {{attr.type}}& {{name}}() const override { return {{name}}_; }
  {{attr.type}}* mutable_{{name}}() override { return &{{name}}_; }
  void set_{{name}}(const {{attr.type}}& {{name}}) override { {{name}}_ = {{name}}; }

  {% endfor -%}

  Maybe<AttrVal> GetAttr(const std::string& attr_name) const;

 private:
  {% for name, attr in op.attrs -%}
  {{attr.type}} {{name}}_{% if existsIn(attr, "default") %} = {{attr.default}}{% endif %};
  {% endfor %}
};
}  // namespace schema
{% endfor %}
} // namespace oneflow
)OP_SCHEMA_INC"
