R"OP_SCHEMA_INC(
#include "oneflow/core/framework/op_interp_ctx.h"

namespace oneflow {
{% for opname, op in ops %}
class {{opname}}InterpCtx : public OpInterpCtx {
 public:
  virtual ~{{opname}}InterpCtx() = default;

  {% if length(op.attrs) > 0 -%}
  {% for name, attr in op.attrs -%}
  virtual const {{attr.type}}& {{name}}() const = 0;
  virtual {{attr.type}}* mutable_{{name}}() = 0;
  virtual void set_{{name}}(const {{attr.type}}& {{name}}) = 0;

  {% endfor %}{%- endif -%}
  const HashSet<std::string>& AttrNames() const override;

  static Maybe<void> InferNdSbp(user_op::InferNdSbpFnContext* ctx);
  static Maybe<void> GetSbp(user_op::SbpContext* ctx);
  static Maybe<void> InferLogicalTensorDesc(user_op::InferContext* ctx);
  static Maybe<void> InferTensorDesc(user_op::InferContext* ctx);
  static Maybe<void> InferDataType(user_op::InferContext* ctx);
};

template <typename Provider>
class {{opname}}InterpCtxImpl : public {{opname}}InterpCtx {
 public:
  {{opname}}InterpCtxImpl() = default;
  {{opname}}InterpCtxImpl(const Provider& impl) : impl_(impl) {}
  {% for name, attr in op.attrs %}
  const {{attr.type}}& {{name}}() const { return impl_.{{name}}(); }
  {{attr.type}}* mutable_{{name}}() { return impl_.mutable_{{name}}(); }
  void set_{{name}}(const {{attr.type}}& {{name}}) { impl_.set_{{name}}({{name}}); }
  {% endfor %}
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const override {
    return impl_.GetAttr(attr_name);
  }

 private:
  Provider impl_;
};

namespace schema {
class {{opname}} {
 public:
  {{opname}}() : internal_(std::make_shared<Internal>()) {}
  {% for name, attr in op.attrs %}
  const {{attr.type}}& {{name}}() const { return internal_->{{name}}; }
  {{attr.type}}* mutable_{{name}}() { return &internal_->{{name}}; }
  void set_{{name}}(const {{attr.type}}& {{name}}) { internal_->{{name}} = {{name}}; }
  {% endfor %}
  Maybe<AttrVal> GetAttr(const std::string& attr_name) const;

 private:
  {% if length(op.attrs) > 0 -%}
  struct Internal { {% for name, attr in op.attrs %}
    {{attr.type}} {{name}}{% if existsIn(attr, "default") %} = {{attr.default}}{% endif %};{% endfor %}
  };
  {%- else -%}
  struct Internal {};
  {%- endif %}
  std::shared_ptr<Internal> internal_;
};
}  // namespace schema
{% endfor %}
} // namespace oneflow
)OP_SCHEMA_INC"
