/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* oneflow op schema                                                          *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


#include "oneflow/core/framework/op_interp_ctx_generated.h"

#include "oneflow/core/common/auto_registration_factory.h"

namespace oneflow {

#define REGISTER_OP_INTERP_CTX(op_type, ctx) \
  REGISTER_CLASS_CREATOR(std::string, op_type, OpInterpCtx, ([]() { return new ctx; }))


const HashSet<std::string>& AbsGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AbsGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "abs_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.abs_grad", AbsGradOpInterpCtxImpl<schema::AbsGradOp>);

const HashSet<std::string>& AbsOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AbsOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "abs op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.abs", AbsOpInterpCtxImpl<schema::AbsOp>);

const HashSet<std::string>& AccOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "max_acc_num", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AccOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "max_acc_num") {
    return CastAttr(&internal_->max_acc_num);
  }
  return Error::RuntimeError() << "acc op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.acc", AccOpInterpCtxImpl<schema::AccOp>);

const HashSet<std::string>& AcosGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AcosGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "acos_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.acos_grad", AcosGradOpInterpCtxImpl<schema::AcosGradOp>);

const HashSet<std::string>& AcosOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AcosOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "acos op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.acos", AcosOpInterpCtxImpl<schema::AcosOp>);

const HashSet<std::string>& AcoshGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AcoshGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "acosh_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.acosh_grad", AcoshGradOpInterpCtxImpl<schema::AcoshGradOp>);

const HashSet<std::string>& AcoshOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AcoshOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "acosh op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.acosh", AcoshOpInterpCtxImpl<schema::AcoshOp>);

const HashSet<std::string>& AdagradUpdateOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "epsilon", "l1", "l2", "learning_rate_val", "lr_decay", "scale", "train_step_val", "weight_decay", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AdagradUpdateOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "epsilon") {
    return CastAttr(&internal_->epsilon);
  }
  if(attr_name == "l1") {
    return CastAttr(&internal_->l1);
  }
  if(attr_name == "l2") {
    return CastAttr(&internal_->l2);
  }
  if(attr_name == "learning_rate_val") {
    return CastAttr(&internal_->learning_rate_val);
  }
  if(attr_name == "lr_decay") {
    return CastAttr(&internal_->lr_decay);
  }
  if(attr_name == "scale") {
    return CastAttr(&internal_->scale);
  }
  if(attr_name == "train_step_val") {
    return CastAttr(&internal_->train_step_val);
  }
  if(attr_name == "weight_decay") {
    return CastAttr(&internal_->weight_decay);
  }
  return Error::RuntimeError() << "adagrad_update op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.adagrad_update", AdagradUpdateOpInterpCtxImpl<schema::AdagradUpdateOp>);

const HashSet<std::string>& AdamBiasCorrectionFactorOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "beta", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AdamBiasCorrectionFactorOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "beta") {
    return CastAttr(&internal_->beta);
  }
  return Error::RuntimeError() << "adam_bias_correction_factor op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.adam_bias_correction_factor", AdamBiasCorrectionFactorOpInterpCtxImpl<schema::AdamBiasCorrectionFactorOp>);

const HashSet<std::string>& AdamUpdateOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "amsgrad", "beta1", "beta2", "bias_correction1_val", "bias_correction2_val", "do_bias_correction", "epsilon", "l1", "l2", "learning_rate_val", "scale", "weight_decay", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AdamUpdateOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "amsgrad") {
    return CastAttr(&internal_->amsgrad);
  }
  if(attr_name == "beta1") {
    return CastAttr(&internal_->beta1);
  }
  if(attr_name == "beta2") {
    return CastAttr(&internal_->beta2);
  }
  if(attr_name == "bias_correction1_val") {
    return CastAttr(&internal_->bias_correction1_val);
  }
  if(attr_name == "bias_correction2_val") {
    return CastAttr(&internal_->bias_correction2_val);
  }
  if(attr_name == "do_bias_correction") {
    return CastAttr(&internal_->do_bias_correction);
  }
  if(attr_name == "epsilon") {
    return CastAttr(&internal_->epsilon);
  }
  if(attr_name == "l1") {
    return CastAttr(&internal_->l1);
  }
  if(attr_name == "l2") {
    return CastAttr(&internal_->l2);
  }
  if(attr_name == "learning_rate_val") {
    return CastAttr(&internal_->learning_rate_val);
  }
  if(attr_name == "scale") {
    return CastAttr(&internal_->scale);
  }
  if(attr_name == "weight_decay") {
    return CastAttr(&internal_->weight_decay);
  }
  return Error::RuntimeError() << "adam_update op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.adam_update", AdamUpdateOpInterpCtxImpl<schema::AdamUpdateOp>);

const HashSet<std::string>& AdaptiveAvgPool1DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "output_size", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AdaptiveAvgPool1DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "output_size") {
    return CastAttr(&internal_->output_size);
  }
  return Error::RuntimeError() << "adaptive_avg_pool1d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.adaptive_avg_pool1d_grad", AdaptiveAvgPool1DGradOpInterpCtxImpl<schema::AdaptiveAvgPool1DGradOp>);

const HashSet<std::string>& AdaptiveAvgPool1DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "output_size", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AdaptiveAvgPool1DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "output_size") {
    return CastAttr(&internal_->output_size);
  }
  return Error::RuntimeError() << "adaptive_avg_pool1d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.adaptive_avg_pool1d", AdaptiveAvgPool1DOpInterpCtxImpl<schema::AdaptiveAvgPool1DOp>);

const HashSet<std::string>& AdaptiveAvgPool2DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "output_size", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AdaptiveAvgPool2DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "output_size") {
    return CastAttr(&internal_->output_size);
  }
  return Error::RuntimeError() << "adaptive_avg_pool2d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.adaptive_avg_pool2d_grad", AdaptiveAvgPool2DGradOpInterpCtxImpl<schema::AdaptiveAvgPool2DGradOp>);

const HashSet<std::string>& AdaptiveAvgPool2DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "output_size", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AdaptiveAvgPool2DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "output_size") {
    return CastAttr(&internal_->output_size);
  }
  return Error::RuntimeError() << "adaptive_avg_pool2d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.adaptive_avg_pool2d", AdaptiveAvgPool2DOpInterpCtxImpl<schema::AdaptiveAvgPool2DOp>);

const HashSet<std::string>& AdaptiveAvgPool3DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "output_size", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AdaptiveAvgPool3DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "output_size") {
    return CastAttr(&internal_->output_size);
  }
  return Error::RuntimeError() << "adaptive_avg_pool3d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.adaptive_avg_pool3d_grad", AdaptiveAvgPool3DGradOpInterpCtxImpl<schema::AdaptiveAvgPool3DGradOp>);

const HashSet<std::string>& AdaptiveAvgPool3DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "output_size", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AdaptiveAvgPool3DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "output_size") {
    return CastAttr(&internal_->output_size);
  }
  return Error::RuntimeError() << "adaptive_avg_pool3d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.adaptive_avg_pool3d", AdaptiveAvgPool3DOpInterpCtxImpl<schema::AdaptiveAvgPool3DOp>);

const HashSet<std::string>& AddNOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AddNOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "add_n op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.add_n", AddNOpInterpCtxImpl<schema::AddNOp>);

const HashSet<std::string>& AffineGridGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "align_corners", "size", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AffineGridGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "align_corners") {
    return CastAttr(&internal_->align_corners);
  }
  if(attr_name == "size") {
    return CastAttr(&internal_->size);
  }
  return Error::RuntimeError() << "affine_grid_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.affine_grid_grad", AffineGridGradOpInterpCtxImpl<schema::AffineGridGradOp>);

const HashSet<std::string>& AffineGridOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "align_corners", "size", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AffineGridOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "align_corners") {
    return CastAttr(&internal_->align_corners);
  }
  if(attr_name == "size") {
    return CastAttr(&internal_->size);
  }
  return Error::RuntimeError() << "affine_grid op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.affine_grid", AffineGridOpInterpCtxImpl<schema::AffineGridOp>);

const HashSet<std::string>& AmpWhiteIdentityOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AmpWhiteIdentityOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "amp_white_identity op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.amp_white_identity", AmpWhiteIdentityOpInterpCtxImpl<schema::AmpWhiteIdentityOp>);

const HashSet<std::string>& ArangeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dtype", "float_delta", "float_limit", "float_start", "integer_delta", "integer_limit", "integer_start", "nd_sbp", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ArangeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dtype") {
    return CastAttr(&internal_->dtype);
  }
  if(attr_name == "float_delta") {
    return CastAttr(&internal_->float_delta);
  }
  if(attr_name == "float_limit") {
    return CastAttr(&internal_->float_limit);
  }
  if(attr_name == "float_start") {
    return CastAttr(&internal_->float_start);
  }
  if(attr_name == "integer_delta") {
    return CastAttr(&internal_->integer_delta);
  }
  if(attr_name == "integer_limit") {
    return CastAttr(&internal_->integer_limit);
  }
  if(attr_name == "integer_start") {
    return CastAttr(&internal_->integer_start);
  }
  if(attr_name == "nd_sbp") {
    return CastAttr(&internal_->nd_sbp);
  }
  return Error::RuntimeError() << "arange op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.arange", ArangeOpInterpCtxImpl<schema::ArangeOp>);

const HashSet<std::string>& ArgSortOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "direction", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ArgSortOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "direction") {
    return CastAttr(&internal_->direction);
  }
  return Error::RuntimeError() << "arg_sort op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.arg_sort", ArgSortOpInterpCtxImpl<schema::ArgSortOp>);

const HashSet<std::string>& ArgmaxOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ArgmaxOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "argmax op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.argmax", ArgmaxOpInterpCtxImpl<schema::ArgmaxOp>);

const HashSet<std::string>& ArgwhereOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dtype", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ArgwhereOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dtype") {
    return CastAttr(&internal_->dtype);
  }
  return Error::RuntimeError() << "argwhere op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.argwhere", ArgwhereOpInterpCtxImpl<schema::ArgwhereOp>);

const HashSet<std::string>& AsinGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AsinGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "asin_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.asin_grad", AsinGradOpInterpCtxImpl<schema::AsinGradOp>);

const HashSet<std::string>& AsinOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AsinOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "asin op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.asin", AsinOpInterpCtxImpl<schema::AsinOp>);

const HashSet<std::string>& AsinhGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AsinhGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "asinh_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.asinh_grad", AsinhGradOpInterpCtxImpl<schema::AsinhGradOp>);

const HashSet<std::string>& AsinhOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AsinhOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "asinh op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.asinh", AsinhOpInterpCtxImpl<schema::AsinhOp>);

const HashSet<std::string>& AssignIfNotOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AssignIfNotOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "assign_if_not op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.assign_if_not", AssignIfNotOpInterpCtxImpl<schema::AssignIfNotOp>);

const HashSet<std::string>& AssignIfOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AssignIfOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "assign_if op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.assign_if", AssignIfOpInterpCtxImpl<schema::AssignIfOp>);

const HashSet<std::string>& AssignOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AssignOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "assign op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.assign", AssignOpInterpCtxImpl<schema::AssignOp>);

const HashSet<std::string>& Atan2OpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> Atan2Op::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "atan2 op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.atan2", Atan2OpInterpCtxImpl<schema::Atan2Op>);

const HashSet<std::string>& Atan2XGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> Atan2XGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "atan2_x_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.atan2_x_grad", Atan2XGradOpInterpCtxImpl<schema::Atan2XGradOp>);

const HashSet<std::string>& Atan2YGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> Atan2YGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "atan2_y_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.atan2_y_grad", Atan2YGradOpInterpCtxImpl<schema::Atan2YGradOp>);

const HashSet<std::string>& AtanGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AtanGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "atan_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.atan_grad", AtanGradOpInterpCtxImpl<schema::AtanGradOp>);

const HashSet<std::string>& AtanOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AtanOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "atan op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.atan", AtanOpInterpCtxImpl<schema::AtanOp>);

const HashSet<std::string>& AtanhGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AtanhGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "atanh_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.atanh_grad", AtanhGradOpInterpCtxImpl<schema::AtanhGradOp>);

const HashSet<std::string>& AtanhOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AtanhOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "atanh op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.atanh", AtanhOpInterpCtxImpl<schema::AtanhOp>);

const HashSet<std::string>& AvgPool1DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "count_include_pad", "data_format", "divisor_override", "kernel_size", "padding", "stride", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AvgPool1DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "count_include_pad") {
    return CastAttr(&internal_->count_include_pad);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "divisor_override") {
    return CastAttr(&internal_->divisor_override);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "stride") {
    return CastAttr(&internal_->stride);
  }
  return Error::RuntimeError() << "avgpool_1d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.avgpool_1d_grad", AvgPool1DGradOpInterpCtxImpl<schema::AvgPool1DGradOp>);

const HashSet<std::string>& AvgPool1DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "count_include_pad", "data_format", "divisor_override", "kernel_size", "padding", "stride", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AvgPool1DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "count_include_pad") {
    return CastAttr(&internal_->count_include_pad);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "divisor_override") {
    return CastAttr(&internal_->divisor_override);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "stride") {
    return CastAttr(&internal_->stride);
  }
  return Error::RuntimeError() << "avgpool_1d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.avgpool_1d", AvgPool1DOpInterpCtxImpl<schema::AvgPool1DOp>);

const HashSet<std::string>& AvgPool2DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "count_include_pad", "data_format", "divisor_override", "kernel_size", "padding", "stride", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AvgPool2DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "count_include_pad") {
    return CastAttr(&internal_->count_include_pad);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "divisor_override") {
    return CastAttr(&internal_->divisor_override);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "stride") {
    return CastAttr(&internal_->stride);
  }
  return Error::RuntimeError() << "avgpool_2d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.avgpool_2d_grad", AvgPool2DGradOpInterpCtxImpl<schema::AvgPool2DGradOp>);

const HashSet<std::string>& AvgPool2DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "count_include_pad", "data_format", "divisor_override", "kernel_size", "padding", "stride", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AvgPool2DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "count_include_pad") {
    return CastAttr(&internal_->count_include_pad);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "divisor_override") {
    return CastAttr(&internal_->divisor_override);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "stride") {
    return CastAttr(&internal_->stride);
  }
  return Error::RuntimeError() << "avgpool_2d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.avgpool_2d", AvgPool2DOpInterpCtxImpl<schema::AvgPool2DOp>);

const HashSet<std::string>& AvgPool3DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "count_include_pad", "data_format", "divisor_override", "kernel_size", "padding", "stride", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AvgPool3DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "count_include_pad") {
    return CastAttr(&internal_->count_include_pad);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "divisor_override") {
    return CastAttr(&internal_->divisor_override);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "stride") {
    return CastAttr(&internal_->stride);
  }
  return Error::RuntimeError() << "avgpool_3d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.avgpool_3d_grad", AvgPool3DGradOpInterpCtxImpl<schema::AvgPool3DGradOp>);

const HashSet<std::string>& AvgPool3DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "count_include_pad", "data_format", "divisor_override", "kernel_size", "padding", "stride", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> AvgPool3DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "count_include_pad") {
    return CastAttr(&internal_->count_include_pad);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "divisor_override") {
    return CastAttr(&internal_->divisor_override);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "stride") {
    return CastAttr(&internal_->stride);
  }
  return Error::RuntimeError() << "avgpool_3d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.avgpool_3d", AvgPool3DOpInterpCtxImpl<schema::AvgPool3DOp>);

const HashSet<std::string>& BatchGatherOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BatchGatherOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "batch_gather op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.batch_gather", BatchGatherOpInterpCtxImpl<schema::BatchGatherOp>);

const HashSet<std::string>& BatchMatmulOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "alpha", "transpose_a", "transpose_b", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BatchMatmulOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "alpha") {
    return CastAttr(&internal_->alpha);
  }
  if(attr_name == "transpose_a") {
    return CastAttr(&internal_->transpose_a);
  }
  if(attr_name == "transpose_b") {
    return CastAttr(&internal_->transpose_b);
  }
  return Error::RuntimeError() << "batch_matmul op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.batch_matmul", BatchMatmulOpInterpCtxImpl<schema::BatchMatmulOp>);

const HashSet<std::string>& BernoulliOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dtype", "has_seed", "seed", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BernoulliOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dtype") {
    return CastAttr(&internal_->dtype);
  }
  if(attr_name == "has_seed") {
    return CastAttr(&internal_->has_seed);
  }
  if(attr_name == "seed") {
    return CastAttr(&internal_->seed);
  }
  return Error::RuntimeError() << "bernoulli op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.bernoulli", BernoulliOpInterpCtxImpl<schema::BernoulliOp>);

const HashSet<std::string>& BiasAddOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BiasAddOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  return Error::RuntimeError() << "bias_add op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.bias_add", BiasAddOpInterpCtxImpl<schema::BiasAddOp>);

const HashSet<std::string>& BinaryCrossEntropyGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BinaryCrossEntropyGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "binary_cross_entropy_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.binary_cross_entropy_grad", BinaryCrossEntropyGradOpInterpCtxImpl<schema::BinaryCrossEntropyGradOp>);

const HashSet<std::string>& BinaryCrossEntropyOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BinaryCrossEntropyOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "binary_cross_entropy op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.binary_cross_entropy", BinaryCrossEntropyOpInterpCtxImpl<schema::BinaryCrossEntropyOp>);

const HashSet<std::string>& BinaryCrossEntropyWithLogitsGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "has_pos_weight", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BinaryCrossEntropyWithLogitsGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "has_pos_weight") {
    return CastAttr(&internal_->has_pos_weight);
  }
  return Error::RuntimeError() << "binary_cross_entropy_with_logits_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.binary_cross_entropy_with_logits_grad", BinaryCrossEntropyWithLogitsGradOpInterpCtxImpl<schema::BinaryCrossEntropyWithLogitsGradOp>);

const HashSet<std::string>& BinaryCrossEntropyWithLogitsOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "has_pos_weight", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BinaryCrossEntropyWithLogitsOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "has_pos_weight") {
    return CastAttr(&internal_->has_pos_weight);
  }
  return Error::RuntimeError() << "binary_cross_entropy_with_logits op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.binary_cross_entropy_with_logits", BinaryCrossEntropyWithLogitsOpInterpCtxImpl<schema::BinaryCrossEntropyWithLogitsOp>);

const HashSet<std::string>& BroadcastAddOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastAddOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_add op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_add", BroadcastAddOpInterpCtxImpl<schema::BroadcastAddOp>);

const HashSet<std::string>& BroadcastDivGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastDivGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_div_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_div_grad", BroadcastDivGradOpInterpCtxImpl<schema::BroadcastDivGradOp>);

const HashSet<std::string>& BroadcastDivOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastDivOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_div op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_div", BroadcastDivOpInterpCtxImpl<schema::BroadcastDivOp>);

const HashSet<std::string>& BroadcastEqualOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastEqualOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_equal op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_equal", BroadcastEqualOpInterpCtxImpl<schema::BroadcastEqualOp>);

const HashSet<std::string>& BroadcastFloorModOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastFloorModOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_floor_mod op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_floor_mod", BroadcastFloorModOpInterpCtxImpl<schema::BroadcastFloorModOp>);

const HashSet<std::string>& BroadcastFmodOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastFmodOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_fmod op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_fmod", BroadcastFmodOpInterpCtxImpl<schema::BroadcastFmodOp>);

const HashSet<std::string>& BroadcastGreaterEqualOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastGreaterEqualOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_greater_equal op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_greater_equal", BroadcastGreaterEqualOpInterpCtxImpl<schema::BroadcastGreaterEqualOp>);

const HashSet<std::string>& BroadcastGreaterOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastGreaterOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_greater op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_greater", BroadcastGreaterOpInterpCtxImpl<schema::BroadcastGreaterOp>);

const HashSet<std::string>& BroadcastLessEqualOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastLessEqualOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_less_equal op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_less_equal", BroadcastLessEqualOpInterpCtxImpl<schema::BroadcastLessEqualOp>);

const HashSet<std::string>& BroadcastLessOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastLessOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_less op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_less", BroadcastLessOpInterpCtxImpl<schema::BroadcastLessOp>);

const HashSet<std::string>& BroadcastLikeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "broadcast_axes", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastLikeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "broadcast_axes") {
    return CastAttr(&internal_->broadcast_axes);
  }
  return Error::RuntimeError() << "broadcast_like op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_like", BroadcastLikeOpInterpCtxImpl<schema::BroadcastLikeOp>);

const HashSet<std::string>& BroadcastLogicalAndOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastLogicalAndOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_logical_and op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_logical_and", BroadcastLogicalAndOpInterpCtxImpl<schema::BroadcastLogicalAndOp>);

const HashSet<std::string>& BroadcastLogicalOrOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastLogicalOrOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_logical_or op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_logical_or", BroadcastLogicalOrOpInterpCtxImpl<schema::BroadcastLogicalOrOp>);

const HashSet<std::string>& BroadcastLogicalXorOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastLogicalXorOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_logical_xor op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_logical_xor", BroadcastLogicalXorOpInterpCtxImpl<schema::BroadcastLogicalXorOp>);

const HashSet<std::string>& BroadcastMatmulGradBOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "alpha", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastMatmulGradBOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "alpha") {
    return CastAttr(&internal_->alpha);
  }
  return Error::RuntimeError() << "broadcast_matmul_grad_b op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_matmul_grad_b", BroadcastMatmulGradBOpInterpCtxImpl<schema::BroadcastMatmulGradBOp>);

const HashSet<std::string>& BroadcastMatmulOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "alpha", "transpose_a", "transpose_b", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastMatmulOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "alpha") {
    return CastAttr(&internal_->alpha);
  }
  if(attr_name == "transpose_a") {
    return CastAttr(&internal_->transpose_a);
  }
  if(attr_name == "transpose_b") {
    return CastAttr(&internal_->transpose_b);
  }
  return Error::RuntimeError() << "broadcast_matmul op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_matmul", BroadcastMatmulOpInterpCtxImpl<schema::BroadcastMatmulOp>);

const HashSet<std::string>& BroadcastMaximumOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastMaximumOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_maximum op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_maximum", BroadcastMaximumOpInterpCtxImpl<schema::BroadcastMaximumOp>);

const HashSet<std::string>& BroadcastMinimumOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastMinimumOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_minimum op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_minimum", BroadcastMinimumOpInterpCtxImpl<schema::BroadcastMinimumOp>);

const HashSet<std::string>& BroadcastMulOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastMulOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_mul op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_mul", BroadcastMulOpInterpCtxImpl<schema::BroadcastMulOp>);

const HashSet<std::string>& BroadcastNotEqualOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastNotEqualOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_not_equal op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_not_equal", BroadcastNotEqualOpInterpCtxImpl<schema::BroadcastNotEqualOp>);

const HashSet<std::string>& BroadcastPowOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastPowOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_pow op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_pow", BroadcastPowOpInterpCtxImpl<schema::BroadcastPowOp>);

const HashSet<std::string>& BroadcastPowXGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastPowXGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_pow_x_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_pow_x_grad", BroadcastPowXGradOpInterpCtxImpl<schema::BroadcastPowXGradOp>);

const HashSet<std::string>& BroadcastPowYGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastPowYGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_pow_y_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_pow_y_grad", BroadcastPowYGradOpInterpCtxImpl<schema::BroadcastPowYGradOp>);

const HashSet<std::string>& BroadcastSubOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> BroadcastSubOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "broadcast_sub op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.broadcast_sub", BroadcastSubOpInterpCtxImpl<schema::BroadcastSubOp>);

const HashSet<std::string>& COCOReaderOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "annotation_file", "batch_size", "group_by_ratio", "image_dir", "nd_sbp", "random_seed", "remove_images_without_annotations", "session_id", "shuffle_after_epoch", "stride_partition", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> COCOReaderOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "annotation_file") {
    return CastAttr(&internal_->annotation_file);
  }
  if(attr_name == "batch_size") {
    return CastAttr(&internal_->batch_size);
  }
  if(attr_name == "group_by_ratio") {
    return CastAttr(&internal_->group_by_ratio);
  }
  if(attr_name == "image_dir") {
    return CastAttr(&internal_->image_dir);
  }
  if(attr_name == "nd_sbp") {
    return CastAttr(&internal_->nd_sbp);
  }
  if(attr_name == "random_seed") {
    return CastAttr(&internal_->random_seed);
  }
  if(attr_name == "remove_images_without_annotations") {
    return CastAttr(&internal_->remove_images_without_annotations);
  }
  if(attr_name == "session_id") {
    return CastAttr(&internal_->session_id);
  }
  if(attr_name == "shuffle_after_epoch") {
    return CastAttr(&internal_->shuffle_after_epoch);
  }
  if(attr_name == "stride_partition") {
    return CastAttr(&internal_->stride_partition);
  }
  return Error::RuntimeError() << "COCOReader op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.COCOReader", COCOReaderOpInterpCtxImpl<schema::COCOReaderOp>);

const HashSet<std::string>& CastLikeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CastLikeOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "cast_like op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.cast_like", CastLikeOpInterpCtxImpl<schema::CastLikeOp>);

const HashSet<std::string>& CastOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dtype", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CastOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dtype") {
    return CastAttr(&internal_->dtype);
  }
  return Error::RuntimeError() << "cast op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.cast", CastOpInterpCtxImpl<schema::CastOp>);

const HashSet<std::string>& CastToStaticShapeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CastToStaticShapeOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "cast_to_static_shape op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.cast_to_static_shape", CastToStaticShapeOpInterpCtxImpl<schema::CastToStaticShapeOp>);

const HashSet<std::string>& CastToTickOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CastToTickOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "cast_to_tick op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.cast_to_tick", CastToTickOpInterpCtxImpl<schema::CastToTickOp>);

const HashSet<std::string>& CategoricalOrdinalEncodeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "hash_precomputed", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CategoricalOrdinalEncodeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "hash_precomputed") {
    return CastAttr(&internal_->hash_precomputed);
  }
  return Error::RuntimeError() << "CategoricalOrdinalEncode op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.CategoricalOrdinalEncode", CategoricalOrdinalEncodeOpInterpCtxImpl<schema::CategoricalOrdinalEncodeOp>);

const HashSet<std::string>& CcreluGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CcreluGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "ccrelu_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.ccrelu_grad", CcreluGradOpInterpCtxImpl<schema::CcreluGradOp>);

const HashSet<std::string>& CcreluOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CcreluOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "ccrelu op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.ccrelu", CcreluOpInterpCtxImpl<schema::CcreluOp>);

const HashSet<std::string>& CeilGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CeilGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "ceil_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.ceil_grad", CeilGradOpInterpCtxImpl<schema::CeilGradOp>);

const HashSet<std::string>& CeilOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CeilOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "ceil op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.ceil", CeilOpInterpCtxImpl<schema::CeilOp>);

const HashSet<std::string>& CeluGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "alpha", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CeluGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "alpha") {
    return CastAttr(&internal_->alpha);
  }
  return Error::RuntimeError() << "celu_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.celu_grad", CeluGradOpInterpCtxImpl<schema::CeluGradOp>);

const HashSet<std::string>& CeluOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "alpha", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CeluOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "alpha") {
    return CastAttr(&internal_->alpha);
  }
  return Error::RuntimeError() << "celu op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.celu", CeluOpInterpCtxImpl<schema::CeluOp>);

const HashSet<std::string>& ClipByScalarGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "floating_max", "floating_min", "integral_max", "integral_min", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ClipByScalarGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "floating_max") {
    return CastAttr(&internal_->floating_max);
  }
  if(attr_name == "floating_min") {
    return CastAttr(&internal_->floating_min);
  }
  if(attr_name == "integral_max") {
    return CastAttr(&internal_->integral_max);
  }
  if(attr_name == "integral_min") {
    return CastAttr(&internal_->integral_min);
  }
  return Error::RuntimeError() << "clip_by_scalar_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.clip_by_scalar_grad", ClipByScalarGradOpInterpCtxImpl<schema::ClipByScalarGradOp>);

const HashSet<std::string>& ClipByScalarMaxGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "floating_max", "integral_max", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ClipByScalarMaxGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "floating_max") {
    return CastAttr(&internal_->floating_max);
  }
  if(attr_name == "integral_max") {
    return CastAttr(&internal_->integral_max);
  }
  return Error::RuntimeError() << "clip_by_scalar_max_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.clip_by_scalar_max_grad", ClipByScalarMaxGradOpInterpCtxImpl<schema::ClipByScalarMaxGradOp>);

const HashSet<std::string>& ClipByScalarMaxOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "floating_max", "integral_max", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ClipByScalarMaxOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "floating_max") {
    return CastAttr(&internal_->floating_max);
  }
  if(attr_name == "integral_max") {
    return CastAttr(&internal_->integral_max);
  }
  return Error::RuntimeError() << "clip_by_scalar_max op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.clip_by_scalar_max", ClipByScalarMaxOpInterpCtxImpl<schema::ClipByScalarMaxOp>);

const HashSet<std::string>& ClipByScalarMinGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "floating_min", "integral_min", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ClipByScalarMinGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "floating_min") {
    return CastAttr(&internal_->floating_min);
  }
  if(attr_name == "integral_min") {
    return CastAttr(&internal_->integral_min);
  }
  return Error::RuntimeError() << "clip_by_scalar_min_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.clip_by_scalar_min_grad", ClipByScalarMinGradOpInterpCtxImpl<schema::ClipByScalarMinGradOp>);

const HashSet<std::string>& ClipByScalarMinOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "floating_min", "integral_min", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ClipByScalarMinOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "floating_min") {
    return CastAttr(&internal_->floating_min);
  }
  if(attr_name == "integral_min") {
    return CastAttr(&internal_->integral_min);
  }
  return Error::RuntimeError() << "clip_by_scalar_min op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.clip_by_scalar_min", ClipByScalarMinOpInterpCtxImpl<schema::ClipByScalarMinOp>);

const HashSet<std::string>& ClipByScalarOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "floating_max", "floating_min", "integral_max", "integral_min", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ClipByScalarOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "floating_max") {
    return CastAttr(&internal_->floating_max);
  }
  if(attr_name == "floating_min") {
    return CastAttr(&internal_->floating_min);
  }
  if(attr_name == "integral_max") {
    return CastAttr(&internal_->integral_max);
  }
  if(attr_name == "integral_min") {
    return CastAttr(&internal_->integral_min);
  }
  return Error::RuntimeError() << "clip_by_scalar op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.clip_by_scalar", ClipByScalarOpInterpCtxImpl<schema::ClipByScalarOp>);

const HashSet<std::string>& CoinFlipOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "batch_size", "has_seed", "nd_sbp", "probability", "seed", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CoinFlipOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "batch_size") {
    return CastAttr(&internal_->batch_size);
  }
  if(attr_name == "has_seed") {
    return CastAttr(&internal_->has_seed);
  }
  if(attr_name == "nd_sbp") {
    return CastAttr(&internal_->nd_sbp);
  }
  if(attr_name == "probability") {
    return CastAttr(&internal_->probability);
  }
  if(attr_name == "seed") {
    return CastAttr(&internal_->seed);
  }
  return Error::RuntimeError() << "coin_flip op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.coin_flip", CoinFlipOpInterpCtxImpl<schema::CoinFlipOp>);

const HashSet<std::string>& CombinedMarginLossGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "depth", "m1", "m2", "m3", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CombinedMarginLossGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "depth") {
    return CastAttr(&internal_->depth);
  }
  if(attr_name == "m1") {
    return CastAttr(&internal_->m1);
  }
  if(attr_name == "m2") {
    return CastAttr(&internal_->m2);
  }
  if(attr_name == "m3") {
    return CastAttr(&internal_->m3);
  }
  return Error::RuntimeError() << "combined_margin_loss_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.combined_margin_loss_grad", CombinedMarginLossGradOpInterpCtxImpl<schema::CombinedMarginLossGradOp>);

const HashSet<std::string>& CombinedMarginLossOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "depth", "m1", "m2", "m3", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CombinedMarginLossOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "depth") {
    return CastAttr(&internal_->depth);
  }
  if(attr_name == "m1") {
    return CastAttr(&internal_->m1);
  }
  if(attr_name == "m2") {
    return CastAttr(&internal_->m2);
  }
  if(attr_name == "m3") {
    return CastAttr(&internal_->m3);
  }
  return Error::RuntimeError() << "combined_margin_loss op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.combined_margin_loss", CombinedMarginLossOpInterpCtxImpl<schema::CombinedMarginLossOp>);

const HashSet<std::string>& ConcatOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "max_dim_size", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ConcatOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "max_dim_size") {
    return CastAttr(&internal_->max_dim_size);
  }
  return Error::RuntimeError() << "concat op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.concat", ConcatOpInterpCtxImpl<schema::ConcatOp>);

const HashSet<std::string>& ConstantOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dtype", "floating_value", "integer_value", "is_floating_value", "nd_sbp", "shape", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ConstantOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dtype") {
    return CastAttr(&internal_->dtype);
  }
  if(attr_name == "floating_value") {
    return CastAttr(&internal_->floating_value);
  }
  if(attr_name == "integer_value") {
    return CastAttr(&internal_->integer_value);
  }
  if(attr_name == "is_floating_value") {
    return CastAttr(&internal_->is_floating_value);
  }
  if(attr_name == "nd_sbp") {
    return CastAttr(&internal_->nd_sbp);
  }
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  return Error::RuntimeError() << "constant op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.constant", ConstantOpInterpCtxImpl<schema::ConstantOp>);

const HashSet<std::string>& ConstantPad1DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "floating_value", "integral_value", "padding", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ConstantPad1DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "floating_value") {
    return CastAttr(&internal_->floating_value);
  }
  if(attr_name == "integral_value") {
    return CastAttr(&internal_->integral_value);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  return Error::RuntimeError() << "constant_pad1d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.constant_pad1d_grad", ConstantPad1DGradOpInterpCtxImpl<schema::ConstantPad1DGradOp>);

const HashSet<std::string>& ConstantPad1DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "floating_value", "integral_value", "padding", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ConstantPad1DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "floating_value") {
    return CastAttr(&internal_->floating_value);
  }
  if(attr_name == "integral_value") {
    return CastAttr(&internal_->integral_value);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  return Error::RuntimeError() << "constant_pad1d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.constant_pad1d", ConstantPad1DOpInterpCtxImpl<schema::ConstantPad1DOp>);

const HashSet<std::string>& ConstantPad2DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "floating_value", "integral_value", "padding", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ConstantPad2DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "floating_value") {
    return CastAttr(&internal_->floating_value);
  }
  if(attr_name == "integral_value") {
    return CastAttr(&internal_->integral_value);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  return Error::RuntimeError() << "constant_pad2d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.constant_pad2d_grad", ConstantPad2DGradOpInterpCtxImpl<schema::ConstantPad2DGradOp>);

const HashSet<std::string>& ConstantPad2DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "floating_value", "integral_value", "padding", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ConstantPad2DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "floating_value") {
    return CastAttr(&internal_->floating_value);
  }
  if(attr_name == "integral_value") {
    return CastAttr(&internal_->integral_value);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  return Error::RuntimeError() << "constant_pad2d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.constant_pad2d", ConstantPad2DOpInterpCtxImpl<schema::ConstantPad2DOp>);

const HashSet<std::string>& ConstantPad3DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "floating_value", "integral_value", "padding", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ConstantPad3DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "floating_value") {
    return CastAttr(&internal_->floating_value);
  }
  if(attr_name == "integral_value") {
    return CastAttr(&internal_->integral_value);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  return Error::RuntimeError() << "constant_pad3d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.constant_pad3d_grad", ConstantPad3DGradOpInterpCtxImpl<schema::ConstantPad3DGradOp>);

const HashSet<std::string>& ConstantPad3DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "floating_value", "integral_value", "padding", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ConstantPad3DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "floating_value") {
    return CastAttr(&internal_->floating_value);
  }
  if(attr_name == "integral_value") {
    return CastAttr(&internal_->integral_value);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  return Error::RuntimeError() << "constant_pad3d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.constant_pad3d", ConstantPad3DOpInterpCtxImpl<schema::ConstantPad3DOp>);

const HashSet<std::string>& Conv1DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "dilation_rate", "filters", "group", "kernel_size", "padding_before", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> Conv1DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "dilation_rate") {
    return CastAttr(&internal_->dilation_rate);
  }
  if(attr_name == "filters") {
    return CastAttr(&internal_->filters);
  }
  if(attr_name == "group") {
    return CastAttr(&internal_->group);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "conv1d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.conv1d", Conv1DOpInterpCtxImpl<schema::Conv1DOp>);

const HashSet<std::string>& Conv2DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "dilation_rate", "filters", "group", "kernel_size", "padding_before", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> Conv2DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "dilation_rate") {
    return CastAttr(&internal_->dilation_rate);
  }
  if(attr_name == "filters") {
    return CastAttr(&internal_->filters);
  }
  if(attr_name == "group") {
    return CastAttr(&internal_->group);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "conv2d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.conv2d", Conv2DOpInterpCtxImpl<schema::Conv2DOp>);

const HashSet<std::string>& Conv3DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "dilation_rate", "filters", "group", "kernel_size", "padding_before", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> Conv3DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "dilation_rate") {
    return CastAttr(&internal_->dilation_rate);
  }
  if(attr_name == "filters") {
    return CastAttr(&internal_->filters);
  }
  if(attr_name == "group") {
    return CastAttr(&internal_->group);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "conv3d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.conv3d", Conv3DOpInterpCtxImpl<schema::Conv3DOp>);

const HashSet<std::string>& ConvBiasGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "num_spatial_dims", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ConvBiasGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "num_spatial_dims") {
    return CastAttr(&internal_->num_spatial_dims);
  }
  return Error::RuntimeError() << "conv_bias_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.conv_bias_grad", ConvBiasGradOpInterpCtxImpl<schema::ConvBiasGradOp>);

const HashSet<std::string>& ConvDataGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "dilation_rate", "groups", "kernel_size", "num_spatial_dims", "padding_before", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ConvDataGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "dilation_rate") {
    return CastAttr(&internal_->dilation_rate);
  }
  if(attr_name == "groups") {
    return CastAttr(&internal_->groups);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "num_spatial_dims") {
    return CastAttr(&internal_->num_spatial_dims);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "conv_data_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.conv_data_grad", ConvDataGradOpInterpCtxImpl<schema::ConvDataGradOp>);

const HashSet<std::string>& ConvFilterGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "dilation_rate", "groups", "kernel_size", "num_spatial_dims", "padding_before", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ConvFilterGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "dilation_rate") {
    return CastAttr(&internal_->dilation_rate);
  }
  if(attr_name == "groups") {
    return CastAttr(&internal_->groups);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "num_spatial_dims") {
    return CastAttr(&internal_->num_spatial_dims);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "conv_filter_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.conv_filter_grad", ConvFilterGradOpInterpCtxImpl<schema::ConvFilterGradOp>);

const HashSet<std::string>& CopyOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "device_id", "device_type", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CopyOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "device_id") {
    return CastAttr(&internal_->device_id);
  }
  if(attr_name == "device_type") {
    return CastAttr(&internal_->device_type);
  }
  return Error::RuntimeError() << "copy op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.copy", CopyOpInterpCtxImpl<schema::CopyOp>);

const HashSet<std::string>& CosGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CosGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "cos_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.cos_grad", CosGradOpInterpCtxImpl<schema::CosGradOp>);

const HashSet<std::string>& CosOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CosOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "cos op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.cos", CosOpInterpCtxImpl<schema::CosOp>);

const HashSet<std::string>& CoshGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CoshGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "cosh_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.cosh_grad", CoshGradOpInterpCtxImpl<schema::CoshGradOp>);

const HashSet<std::string>& CoshOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CoshOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "cosh op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.cosh", CoshOpInterpCtxImpl<schema::CoshOp>);

const HashSet<std::string>& CountNotFiniteOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CountNotFiniteOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "count_not_finite op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.count_not_finite", CountNotFiniteOpInterpCtxImpl<schema::CountNotFiniteOp>);

const HashSet<std::string>& CpuOnlyReluTestOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CpuOnlyReluTestOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "cpu_only_relu_test op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.cpu_only_relu_test", CpuOnlyReluTestOpInterpCtxImpl<schema::CpuOnlyReluTestOp>);

const HashSet<std::string>& CreateSummaryWriterOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "logdir", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CreateSummaryWriterOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "logdir") {
    return CastAttr(&internal_->logdir);
  }
  return Error::RuntimeError() << "create_summary_writer op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.create_summary_writer", CreateSummaryWriterOpInterpCtxImpl<schema::CreateSummaryWriterOp>);

const HashSet<std::string>& CropMirrorNormalizeFromTensorbufferOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "color_space", "crop_h", "crop_pos_x", "crop_pos_y", "crop_w", "mean", "output_dtype", "output_layout", "std", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CropMirrorNormalizeFromTensorbufferOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "color_space") {
    return CastAttr(&internal_->color_space);
  }
  if(attr_name == "crop_h") {
    return CastAttr(&internal_->crop_h);
  }
  if(attr_name == "crop_pos_x") {
    return CastAttr(&internal_->crop_pos_x);
  }
  if(attr_name == "crop_pos_y") {
    return CastAttr(&internal_->crop_pos_y);
  }
  if(attr_name == "crop_w") {
    return CastAttr(&internal_->crop_w);
  }
  if(attr_name == "mean") {
    return CastAttr(&internal_->mean);
  }
  if(attr_name == "output_dtype") {
    return CastAttr(&internal_->output_dtype);
  }
  if(attr_name == "output_layout") {
    return CastAttr(&internal_->output_layout);
  }
  if(attr_name == "std") {
    return CastAttr(&internal_->std);
  }
  return Error::RuntimeError() << "crop_mirror_normalize_from_tensorbuffer op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.crop_mirror_normalize_from_tensorbuffer", CropMirrorNormalizeFromTensorbufferOpInterpCtxImpl<schema::CropMirrorNormalizeFromTensorbufferOp>);

const HashSet<std::string>& CropMirrorNormalizeFromUint8OpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "color_space", "crop_h", "crop_pos_x", "crop_pos_y", "crop_w", "mean", "output_dtype", "output_layout", "std", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CropMirrorNormalizeFromUint8Op::GetAttr(const std::string& attr_name) const {
  if(attr_name == "color_space") {
    return CastAttr(&internal_->color_space);
  }
  if(attr_name == "crop_h") {
    return CastAttr(&internal_->crop_h);
  }
  if(attr_name == "crop_pos_x") {
    return CastAttr(&internal_->crop_pos_x);
  }
  if(attr_name == "crop_pos_y") {
    return CastAttr(&internal_->crop_pos_y);
  }
  if(attr_name == "crop_w") {
    return CastAttr(&internal_->crop_w);
  }
  if(attr_name == "mean") {
    return CastAttr(&internal_->mean);
  }
  if(attr_name == "output_dtype") {
    return CastAttr(&internal_->output_dtype);
  }
  if(attr_name == "output_layout") {
    return CastAttr(&internal_->output_layout);
  }
  if(attr_name == "std") {
    return CastAttr(&internal_->std);
  }
  return Error::RuntimeError() << "crop_mirror_normalize_from_uint8 op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.crop_mirror_normalize_from_uint8", CropMirrorNormalizeFromUint8OpInterpCtxImpl<schema::CropMirrorNormalizeFromUint8Op>);

const HashSet<std::string>& CtcGreedyDecoderOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "merge_repeated", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CtcGreedyDecoderOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "merge_repeated") {
    return CastAttr(&internal_->merge_repeated);
  }
  return Error::RuntimeError() << "ctc_greedy_decoder op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.ctc_greedy_decoder", CtcGreedyDecoderOpInterpCtxImpl<schema::CtcGreedyDecoderOp>);

const HashSet<std::string>& CtcLossGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "blank", "max_target_length", "zero_infinity", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CtcLossGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "blank") {
    return CastAttr(&internal_->blank);
  }
  if(attr_name == "max_target_length") {
    return CastAttr(&internal_->max_target_length);
  }
  if(attr_name == "zero_infinity") {
    return CastAttr(&internal_->zero_infinity);
  }
  return Error::RuntimeError() << "ctc_loss_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.ctc_loss_grad", CtcLossGradOpInterpCtxImpl<schema::CtcLossGradOp>);

const HashSet<std::string>& CtcLossOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "blank", "max_target_length", "zero_infinity", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CtcLossOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "blank") {
    return CastAttr(&internal_->blank);
  }
  if(attr_name == "max_target_length") {
    return CastAttr(&internal_->max_target_length);
  }
  if(attr_name == "zero_infinity") {
    return CastAttr(&internal_->zero_infinity);
  }
  return Error::RuntimeError() << "ctc_loss op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.ctc_loss", CtcLossOpInterpCtxImpl<schema::CtcLossOp>);

const HashSet<std::string>& CudnnFusedNormalizationAddReluGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "epsilon", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CudnnFusedNormalizationAddReluGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "epsilon") {
    return CastAttr(&internal_->epsilon);
  }
  return Error::RuntimeError() << "cudnn_fused_normalization_add_relu_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.cudnn_fused_normalization_add_relu_grad", CudnnFusedNormalizationAddReluGradOpInterpCtxImpl<schema::CudnnFusedNormalizationAddReluGradOp>);

const HashSet<std::string>& CudnnFusedNormalizationAddReluOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "epsilon", "momentum", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> CudnnFusedNormalizationAddReluOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "epsilon") {
    return CastAttr(&internal_->epsilon);
  }
  if(attr_name == "momentum") {
    return CastAttr(&internal_->momentum);
  }
  return Error::RuntimeError() << "cudnn_fused_normalization_add_relu op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.cudnn_fused_normalization_add_relu", CudnnFusedNormalizationAddReluOpInterpCtxImpl<schema::CudnnFusedNormalizationAddReluOp>);

const HashSet<std::string>& Deconv1DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "dilation_rate", "filters", "groups", "kernel_size", "output_padding", "padding_before", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> Deconv1DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "dilation_rate") {
    return CastAttr(&internal_->dilation_rate);
  }
  if(attr_name == "filters") {
    return CastAttr(&internal_->filters);
  }
  if(attr_name == "groups") {
    return CastAttr(&internal_->groups);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "output_padding") {
    return CastAttr(&internal_->output_padding);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "deconv1d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.deconv1d", Deconv1DOpInterpCtxImpl<schema::Deconv1DOp>);

const HashSet<std::string>& Deconv2DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "dilation_rate", "filters", "groups", "kernel_size", "output_padding", "padding_before", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> Deconv2DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "dilation_rate") {
    return CastAttr(&internal_->dilation_rate);
  }
  if(attr_name == "filters") {
    return CastAttr(&internal_->filters);
  }
  if(attr_name == "groups") {
    return CastAttr(&internal_->groups);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "output_padding") {
    return CastAttr(&internal_->output_padding);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "deconv2d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.deconv2d", Deconv2DOpInterpCtxImpl<schema::Deconv2DOp>);

const HashSet<std::string>& Deconv3DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "dilation_rate", "filters", "groups", "kernel_size", "output_padding", "padding_before", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> Deconv3DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "dilation_rate") {
    return CastAttr(&internal_->dilation_rate);
  }
  if(attr_name == "filters") {
    return CastAttr(&internal_->filters);
  }
  if(attr_name == "groups") {
    return CastAttr(&internal_->groups);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "output_padding") {
    return CastAttr(&internal_->output_padding);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "deconv3d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.deconv3d", Deconv3DOpInterpCtxImpl<schema::Deconv3DOp>);

const HashSet<std::string>& DiagGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "diagonal", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> DiagGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "diagonal") {
    return CastAttr(&internal_->diagonal);
  }
  return Error::RuntimeError() << "diag_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.diag_grad", DiagGradOpInterpCtxImpl<schema::DiagGradOp>);

const HashSet<std::string>& DiagOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "diagonal", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> DiagOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "diagonal") {
    return CastAttr(&internal_->diagonal);
  }
  return Error::RuntimeError() << "diag op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.diag", DiagOpInterpCtxImpl<schema::DiagOp>);

const HashSet<std::string>& DimGatherOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dim", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> DimGatherOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dim") {
    return CastAttr(&internal_->dim);
  }
  return Error::RuntimeError() << "dim_gather op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.dim_gather", DimGatherOpInterpCtxImpl<schema::DimGatherOp>);

const HashSet<std::string>& DimScatterAddLikeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dim", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> DimScatterAddLikeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dim") {
    return CastAttr(&internal_->dim);
  }
  return Error::RuntimeError() << "dim_scatter_add_like op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.dim_scatter_add_like", DimScatterAddLikeOpInterpCtxImpl<schema::DimScatterAddLikeOp>);

const HashSet<std::string>& DimScatterAddOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dim", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> DimScatterAddOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dim") {
    return CastAttr(&internal_->dim);
  }
  return Error::RuntimeError() << "dim_scatter_add op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.dim_scatter_add", DimScatterAddOpInterpCtxImpl<schema::DimScatterAddOp>);

const HashSet<std::string>& DimScatterAddScalarOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dim", "src_scalar", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> DimScatterAddScalarOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dim") {
    return CastAttr(&internal_->dim);
  }
  if(attr_name == "src_scalar") {
    return CastAttr(&internal_->src_scalar);
  }
  return Error::RuntimeError() << "dim_scatter_add_scalar op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.dim_scatter_add_scalar", DimScatterAddScalarOpInterpCtxImpl<schema::DimScatterAddScalarOp>);

const HashSet<std::string>& DimScatterMulOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dim", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> DimScatterMulOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dim") {
    return CastAttr(&internal_->dim);
  }
  return Error::RuntimeError() << "dim_scatter_mul op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.dim_scatter_mul", DimScatterMulOpInterpCtxImpl<schema::DimScatterMulOp>);

const HashSet<std::string>& DimScatterMulScalarOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dim", "src_scalar", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> DimScatterMulScalarOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dim") {
    return CastAttr(&internal_->dim);
  }
  if(attr_name == "src_scalar") {
    return CastAttr(&internal_->src_scalar);
  }
  return Error::RuntimeError() << "dim_scatter_mul_scalar op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.dim_scatter_mul_scalar", DimScatterMulScalarOpInterpCtxImpl<schema::DimScatterMulScalarOp>);

const HashSet<std::string>& DimScatterUpdateOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dim", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> DimScatterUpdateOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dim") {
    return CastAttr(&internal_->dim);
  }
  return Error::RuntimeError() << "dim_scatter_update op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.dim_scatter_update", DimScatterUpdateOpInterpCtxImpl<schema::DimScatterUpdateOp>);

const HashSet<std::string>& DimScatterUpdateScalarOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dim", "src_scalar", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> DimScatterUpdateScalarOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dim") {
    return CastAttr(&internal_->dim);
  }
  if(attr_name == "src_scalar") {
    return CastAttr(&internal_->src_scalar);
  }
  return Error::RuntimeError() << "dim_scatter_update_scalar op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.dim_scatter_update_scalar", DimScatterUpdateScalarOpInterpCtxImpl<schema::DimScatterUpdateScalarOp>);

const HashSet<std::string>& DistributedPartialFcSampleDisableBoxingOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> DistributedPartialFcSampleDisableBoxingOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "distributed_partial_fc_sample_disable_boxing op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.distributed_partial_fc_sample_disable_boxing", DistributedPartialFcSampleDisableBoxingOpInterpCtxImpl<schema::DistributedPartialFcSampleDisableBoxingOp>);

const HashSet<std::string>& DistributedPartialFcSampleOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "num_sample", "seed", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> DistributedPartialFcSampleOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "num_sample") {
    return CastAttr(&internal_->num_sample);
  }
  if(attr_name == "seed") {
    return CastAttr(&internal_->seed);
  }
  return Error::RuntimeError() << "distributed_partial_fc_sample op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.distributed_partial_fc_sample", DistributedPartialFcSampleOpInterpCtxImpl<schema::DistributedPartialFcSampleOp>);

const HashSet<std::string>& DotOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> DotOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "dot op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.dot", DotOpInterpCtxImpl<schema::DotOp>);

const HashSet<std::string>& DropoutGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> DropoutGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "scale") {
    return CastAttr(&internal_->scale);
  }
  return Error::RuntimeError() << "dropout_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.dropout_grad", DropoutGradOpInterpCtxImpl<schema::DropoutGradOp>);

const HashSet<std::string>& DropoutOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "rate", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> DropoutOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "rate") {
    return CastAttr(&internal_->rate);
  }
  return Error::RuntimeError() << "dropout op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.dropout", DropoutOpInterpCtxImpl<schema::DropoutOp>);

const HashSet<std::string>& DynamicLossScaleScheduleOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "increment_period", "multiplier", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> DynamicLossScaleScheduleOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "increment_period") {
    return CastAttr(&internal_->increment_period);
  }
  if(attr_name == "multiplier") {
    return CastAttr(&internal_->multiplier);
  }
  return Error::RuntimeError() << "dynamic_loss_scale_schedule op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.dynamic_loss_scale_schedule", DynamicLossScaleScheduleOpInterpCtxImpl<schema::DynamicLossScaleScheduleOp>);

const HashSet<std::string>& EagerBToSOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "in_parallel_conf", "out_parallel_conf", "out_split_axis", "shape", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> EagerBToSOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "in_parallel_conf") {
    return CastAttr(&internal_->in_parallel_conf);
  }
  if(attr_name == "out_parallel_conf") {
    return CastAttr(&internal_->out_parallel_conf);
  }
  if(attr_name == "out_split_axis") {
    return CastAttr(&internal_->out_split_axis);
  }
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  return Error::RuntimeError() << "eager_b_to_s op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.eager_b_to_s", EagerBToSOpInterpCtxImpl<schema::EagerBToSOp>);

const HashSet<std::string>& EagerNaiveSToSOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "in_parallel_conf", "in_split_axis", "out_parallel_conf", "out_split_axis", "shape", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> EagerNaiveSToSOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "in_parallel_conf") {
    return CastAttr(&internal_->in_parallel_conf);
  }
  if(attr_name == "in_split_axis") {
    return CastAttr(&internal_->in_split_axis);
  }
  if(attr_name == "out_parallel_conf") {
    return CastAttr(&internal_->out_parallel_conf);
  }
  if(attr_name == "out_split_axis") {
    return CastAttr(&internal_->out_split_axis);
  }
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  return Error::RuntimeError() << "eager_naive_s_to_s op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.eager_naive_s_to_s", EagerNaiveSToSOpInterpCtxImpl<schema::EagerNaiveSToSOp>);

const HashSet<std::string>& EagerNcclAllGatherOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "parallel_conf", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> EagerNcclAllGatherOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "parallel_conf") {
    return CastAttr(&internal_->parallel_conf);
  }
  return Error::RuntimeError() << "eager_nccl_all_gather op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.eager_nccl_all_gather", EagerNcclAllGatherOpInterpCtxImpl<schema::EagerNcclAllGatherOp>);

const HashSet<std::string>& EagerNcclAllReduceOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "async_launch", "parallel_conf", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> EagerNcclAllReduceOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "async_launch") {
    return CastAttr(&internal_->async_launch);
  }
  if(attr_name == "parallel_conf") {
    return CastAttr(&internal_->parallel_conf);
  }
  return Error::RuntimeError() << "eager_nccl_all_reduce op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.eager_nccl_all_reduce", EagerNcclAllReduceOpInterpCtxImpl<schema::EagerNcclAllReduceOp>);

const HashSet<std::string>& EagerNcclBroadcastOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "parallel_conf", "root", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> EagerNcclBroadcastOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "parallel_conf") {
    return CastAttr(&internal_->parallel_conf);
  }
  if(attr_name == "root") {
    return CastAttr(&internal_->root);
  }
  return Error::RuntimeError() << "eager_nccl_broadcast op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.eager_nccl_broadcast", EagerNcclBroadcastOpInterpCtxImpl<schema::EagerNcclBroadcastOp>);

const HashSet<std::string>& EagerNcclReduceOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "parallel_conf", "root", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> EagerNcclReduceOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "parallel_conf") {
    return CastAttr(&internal_->parallel_conf);
  }
  if(attr_name == "root") {
    return CastAttr(&internal_->root);
  }
  return Error::RuntimeError() << "eager_nccl_reduce op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.eager_nccl_reduce", EagerNcclReduceOpInterpCtxImpl<schema::EagerNcclReduceOp>);

const HashSet<std::string>& EagerNcclReduceScatterOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "op_type", "parallel_conf", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> EagerNcclReduceScatterOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "op_type") {
    return CastAttr(&internal_->op_type);
  }
  if(attr_name == "parallel_conf") {
    return CastAttr(&internal_->parallel_conf);
  }
  return Error::RuntimeError() << "eager_nccl_reduce_scatter op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.eager_nccl_reduce_scatter", EagerNcclReduceScatterOpInterpCtxImpl<schema::EagerNcclReduceScatterOp>);

const HashSet<std::string>& EagerNcclS2sOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "in_split_axis", "out_split_axis", "parallel_conf", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> EagerNcclS2sOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "in_split_axis") {
    return CastAttr(&internal_->in_split_axis);
  }
  if(attr_name == "out_split_axis") {
    return CastAttr(&internal_->out_split_axis);
  }
  if(attr_name == "parallel_conf") {
    return CastAttr(&internal_->parallel_conf);
  }
  return Error::RuntimeError() << "eager_nccl_s2s op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.eager_nccl_s2s", EagerNcclS2sOpInterpCtxImpl<schema::EagerNcclS2sOp>);

const HashSet<std::string>& EagerPToBOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "in_parallel_conf", "out_parallel_conf", "shape", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> EagerPToBOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "in_parallel_conf") {
    return CastAttr(&internal_->in_parallel_conf);
  }
  if(attr_name == "out_parallel_conf") {
    return CastAttr(&internal_->out_parallel_conf);
  }
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  return Error::RuntimeError() << "eager_p_to_b op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.eager_p_to_b", EagerPToBOpInterpCtxImpl<schema::EagerPToBOp>);

const HashSet<std::string>& EagerPToSOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "in_parallel_conf", "out_parallel_conf", "out_split_axis", "shape", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> EagerPToSOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "in_parallel_conf") {
    return CastAttr(&internal_->in_parallel_conf);
  }
  if(attr_name == "out_parallel_conf") {
    return CastAttr(&internal_->out_parallel_conf);
  }
  if(attr_name == "out_split_axis") {
    return CastAttr(&internal_->out_split_axis);
  }
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  return Error::RuntimeError() << "eager_p_to_s op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.eager_p_to_s", EagerPToSOpInterpCtxImpl<schema::EagerPToSOp>);

const HashSet<std::string>& EagerSToBOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "in_parallel_conf", "in_split_axis", "out_parallel_conf", "shape", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> EagerSToBOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "in_parallel_conf") {
    return CastAttr(&internal_->in_parallel_conf);
  }
  if(attr_name == "in_split_axis") {
    return CastAttr(&internal_->in_split_axis);
  }
  if(attr_name == "out_parallel_conf") {
    return CastAttr(&internal_->out_parallel_conf);
  }
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  return Error::RuntimeError() << "eager_s_to_b op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.eager_s_to_b", EagerSToBOpInterpCtxImpl<schema::EagerSToBOp>);

const HashSet<std::string>& EagerSymmetricSToPOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "in_split_axis", "parallel_conf", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> EagerSymmetricSToPOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "in_split_axis") {
    return CastAttr(&internal_->in_split_axis);
  }
  if(attr_name == "parallel_conf") {
    return CastAttr(&internal_->parallel_conf);
  }
  return Error::RuntimeError() << "eager_symmetric_s_to_p op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.eager_symmetric_s_to_p", EagerSymmetricSToPOpInterpCtxImpl<schema::EagerSymmetricSToPOp>);

const HashSet<std::string>& ElementwiseMaximumBackwardOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ElementwiseMaximumBackwardOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "elementwise_maximum_backward op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.elementwise_maximum_backward", ElementwiseMaximumBackwardOpInterpCtxImpl<schema::ElementwiseMaximumBackwardOp>);

const HashSet<std::string>& ElementwiseMaximumOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ElementwiseMaximumOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "elementwise_maximum op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.elementwise_maximum", ElementwiseMaximumOpInterpCtxImpl<schema::ElementwiseMaximumOp>);

const HashSet<std::string>& ElementwiseMinimumBackwardOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ElementwiseMinimumBackwardOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "elementwise_minimum_backward op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.elementwise_minimum_backward", ElementwiseMinimumBackwardOpInterpCtxImpl<schema::ElementwiseMinimumBackwardOp>);

const HashSet<std::string>& ElementwiseMinimumOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ElementwiseMinimumOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "elementwise_minimum op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.elementwise_minimum", ElementwiseMinimumOpInterpCtxImpl<schema::ElementwiseMinimumOp>);

const HashSet<std::string>& EluGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "alpha", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> EluGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "alpha") {
    return CastAttr(&internal_->alpha);
  }
  return Error::RuntimeError() << "elu_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.elu_grad", EluGradOpInterpCtxImpl<schema::EluGradOp>);

const HashSet<std::string>& EluOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "alpha", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> EluOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "alpha") {
    return CastAttr(&internal_->alpha);
  }
  return Error::RuntimeError() << "elu op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.elu", EluOpInterpCtxImpl<schema::EluOp>);

const HashSet<std::string>& EmptyOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dtype", "nd_sbp", "shape", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> EmptyOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dtype") {
    return CastAttr(&internal_->dtype);
  }
  if(attr_name == "nd_sbp") {
    return CastAttr(&internal_->nd_sbp);
  }
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  return Error::RuntimeError() << "empty op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.empty", EmptyOpInterpCtxImpl<schema::EmptyOp>);

const HashSet<std::string>& ErfGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ErfGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "erf_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.erf_grad", ErfGradOpInterpCtxImpl<schema::ErfGradOp>);

const HashSet<std::string>& ErfOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ErfOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "erf op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.erf", ErfOpInterpCtxImpl<schema::ErfOp>);

const HashSet<std::string>& ErfcGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ErfcGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "erfc_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.erfc_grad", ErfcGradOpInterpCtxImpl<schema::ErfcGradOp>);

const HashSet<std::string>& ErfcOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ErfcOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "erfc op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.erfc", ErfcOpInterpCtxImpl<schema::ErfcOp>);

const HashSet<std::string>& ExpGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ExpGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "exp_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.exp_grad", ExpGradOpInterpCtxImpl<schema::ExpGradOp>);

const HashSet<std::string>& ExpOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ExpOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "exp op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.exp", ExpOpInterpCtxImpl<schema::ExpOp>);

const HashSet<std::string>& ExpandDimsOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ExpandDimsOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  return Error::RuntimeError() << "expand_dims op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.expand_dims", ExpandDimsOpInterpCtxImpl<schema::ExpandDimsOp>);

const HashSet<std::string>& ExpandGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "logical_expand_shape", "logical_out_shape", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ExpandGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "logical_expand_shape") {
    return CastAttr(&internal_->logical_expand_shape);
  }
  if(attr_name == "logical_out_shape") {
    return CastAttr(&internal_->logical_out_shape);
  }
  return Error::RuntimeError() << "expand_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.expand_grad", ExpandGradOpInterpCtxImpl<schema::ExpandGradOp>);

const HashSet<std::string>& ExpandOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "logical_expand_shape", "logical_in_shape", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ExpandOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "logical_expand_shape") {
    return CastAttr(&internal_->logical_expand_shape);
  }
  if(attr_name == "logical_in_shape") {
    return CastAttr(&internal_->logical_in_shape);
  }
  return Error::RuntimeError() << "expand op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.expand", ExpandOpInterpCtxImpl<schema::ExpandOp>);

const HashSet<std::string>& Expm1GradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> Expm1GradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "expm1_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.expm1_grad", Expm1GradOpInterpCtxImpl<schema::Expm1GradOp>);

const HashSet<std::string>& Expm1OpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> Expm1Op::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "expm1 op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.expm1", Expm1OpInterpCtxImpl<schema::Expm1Op>);

const HashSet<std::string>& EyeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> EyeOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "eye op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.eye", EyeOpInterpCtxImpl<schema::EyeOp>);

const HashSet<std::string>& FakeQuantizationOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "quantization_bit", "quantization_formula", "quantization_scheme", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FakeQuantizationOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "quantization_bit") {
    return CastAttr(&internal_->quantization_bit);
  }
  if(attr_name == "quantization_formula") {
    return CastAttr(&internal_->quantization_formula);
  }
  if(attr_name == "quantization_scheme") {
    return CastAttr(&internal_->quantization_scheme);
  }
  return Error::RuntimeError() << "fake_quantization op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.fake_quantization", FakeQuantizationOpInterpCtxImpl<schema::FakeQuantizationOp>);

const HashSet<std::string>& FlattenOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "end_dim", "start_dim", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FlattenOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "end_dim") {
    return CastAttr(&internal_->end_dim);
  }
  if(attr_name == "start_dim") {
    return CastAttr(&internal_->start_dim);
  }
  return Error::RuntimeError() << "flatten op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.flatten", FlattenOpInterpCtxImpl<schema::FlattenOp>);

const HashSet<std::string>& FlipGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dims", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FlipGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dims") {
    return CastAttr(&internal_->dims);
  }
  return Error::RuntimeError() << "flip_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.flip_grad", FlipGradOpInterpCtxImpl<schema::FlipGradOp>);

const HashSet<std::string>& FlipOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dims", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FlipOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dims") {
    return CastAttr(&internal_->dims);
  }
  return Error::RuntimeError() << "flip op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.flip", FlipOpInterpCtxImpl<schema::FlipOp>);

const HashSet<std::string>& FloorGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FloorGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "floor_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.floor_grad", FloorGradOpInterpCtxImpl<schema::FloorGradOp>);

const HashSet<std::string>& FloorOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FloorOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "floor op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.floor", FloorOpInterpCtxImpl<schema::FloorOp>);

const HashSet<std::string>& FloordivOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FloordivOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "floordiv op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.floordiv", FloordivOpInterpCtxImpl<schema::FloordivOp>);

const HashSet<std::string>& FloordivXGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FloordivXGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "floordiv_x_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.floordiv_x_grad", FloordivXGradOpInterpCtxImpl<schema::FloordivXGradOp>);

const HashSet<std::string>& FloordivYGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FloordivYGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "floordiv_y_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.floordiv_y_grad", FloordivYGradOpInterpCtxImpl<schema::FloordivYGradOp>);

const HashSet<std::string>& FlushSummaryWriterOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FlushSummaryWriterOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "flush_summary_writer op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.flush_summary_writer", FlushSummaryWriterOpInterpCtxImpl<schema::FlushSummaryWriterOp>);

const HashSet<std::string>& FoldOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dilation_rate", "kernel_size", "output_size", "padding", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FoldOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dilation_rate") {
    return CastAttr(&internal_->dilation_rate);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "output_size") {
    return CastAttr(&internal_->output_size);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "fold op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.fold", FoldOpInterpCtxImpl<schema::FoldOp>);

const HashSet<std::string>& FusedBiasAddGeluGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FusedBiasAddGeluGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  return Error::RuntimeError() << "fused_bias_add_gelu_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.fused_bias_add_gelu_grad", FusedBiasAddGeluGradOpInterpCtxImpl<schema::FusedBiasAddGeluGradOp>);

const HashSet<std::string>& FusedBiasAddGeluOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FusedBiasAddGeluOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  return Error::RuntimeError() << "fused_bias_add_gelu op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.fused_bias_add_gelu", FusedBiasAddGeluOpInterpCtxImpl<schema::FusedBiasAddGeluOp>);

const HashSet<std::string>& FusedBiasAddMaskScaleOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FusedBiasAddMaskScaleOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "scale") {
    return CastAttr(&internal_->scale);
  }
  return Error::RuntimeError() << "fused_bias_add_mask_scale op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.fused_bias_add_mask_scale", FusedBiasAddMaskScaleOpInterpCtxImpl<schema::FusedBiasAddMaskScaleOp>);

const HashSet<std::string>& FusedCastScaleOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FusedCastScaleOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "scale") {
    return CastAttr(&internal_->scale);
  }
  return Error::RuntimeError() << "fused_cast_scale op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.fused_cast_scale", FusedCastScaleOpInterpCtxImpl<schema::FusedCastScaleOp>);

const HashSet<std::string>& FusedScaleMaskSoftmaxDropoutGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dropout_scale_value", "scale_value", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FusedScaleMaskSoftmaxDropoutGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dropout_scale_value") {
    return CastAttr(&internal_->dropout_scale_value);
  }
  if(attr_name == "scale_value") {
    return CastAttr(&internal_->scale_value);
  }
  return Error::RuntimeError() << "fused_scale_mask_softmax_dropout_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.fused_scale_mask_softmax_dropout_grad", FusedScaleMaskSoftmaxDropoutGradOpInterpCtxImpl<schema::FusedScaleMaskSoftmaxDropoutGradOp>);

const HashSet<std::string>& FusedScaleMaskSoftmaxDropoutOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dropout_scale_value", "mask_fill_value", "scale_value", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FusedScaleMaskSoftmaxDropoutOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dropout_scale_value") {
    return CastAttr(&internal_->dropout_scale_value);
  }
  if(attr_name == "mask_fill_value") {
    return CastAttr(&internal_->mask_fill_value);
  }
  if(attr_name == "scale_value") {
    return CastAttr(&internal_->scale_value);
  }
  return Error::RuntimeError() << "fused_scale_mask_softmax_dropout op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.fused_scale_mask_softmax_dropout", FusedScaleMaskSoftmaxDropoutOpInterpCtxImpl<schema::FusedScaleMaskSoftmaxDropoutOp>);

const HashSet<std::string>& FusedScaleMaskSoftmaxGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "scale_value", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FusedScaleMaskSoftmaxGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "scale_value") {
    return CastAttr(&internal_->scale_value);
  }
  return Error::RuntimeError() << "fused_scale_mask_softmax_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.fused_scale_mask_softmax_grad", FusedScaleMaskSoftmaxGradOpInterpCtxImpl<schema::FusedScaleMaskSoftmaxGradOp>);

const HashSet<std::string>& FusedScaleMaskSoftmaxOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "mask_fill_value", "scale_value", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FusedScaleMaskSoftmaxOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "mask_fill_value") {
    return CastAttr(&internal_->mask_fill_value);
  }
  if(attr_name == "scale_value") {
    return CastAttr(&internal_->scale_value);
  }
  return Error::RuntimeError() << "fused_scale_mask_softmax op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.fused_scale_mask_softmax", FusedScaleMaskSoftmaxOpInterpCtxImpl<schema::FusedScaleMaskSoftmaxOp>);

const HashSet<std::string>& FusedScaleTrilOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "diagonal", "floating_fill_value", "floating_scale_value", "integer_fill_value", "integer_scale_value", "is_floating_fill_value", "is_floating_scale_value", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FusedScaleTrilOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "diagonal") {
    return CastAttr(&internal_->diagonal);
  }
  if(attr_name == "floating_fill_value") {
    return CastAttr(&internal_->floating_fill_value);
  }
  if(attr_name == "floating_scale_value") {
    return CastAttr(&internal_->floating_scale_value);
  }
  if(attr_name == "integer_fill_value") {
    return CastAttr(&internal_->integer_fill_value);
  }
  if(attr_name == "integer_scale_value") {
    return CastAttr(&internal_->integer_scale_value);
  }
  if(attr_name == "is_floating_fill_value") {
    return CastAttr(&internal_->is_floating_fill_value);
  }
  if(attr_name == "is_floating_scale_value") {
    return CastAttr(&internal_->is_floating_scale_value);
  }
  return Error::RuntimeError() << "fused_scale_tril op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.fused_scale_tril", FusedScaleTrilOpInterpCtxImpl<schema::FusedScaleTrilOp>);

const HashSet<std::string>& FusedSelfAttentionQueryMulKeyAndValueGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "alpha", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FusedSelfAttentionQueryMulKeyAndValueGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "alpha") {
    return CastAttr(&internal_->alpha);
  }
  return Error::RuntimeError() << "fused_self_attention_query_mul_key_and_value_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.fused_self_attention_query_mul_key_and_value_grad", FusedSelfAttentionQueryMulKeyAndValueGradOpInterpCtxImpl<schema::FusedSelfAttentionQueryMulKeyAndValueGradOp>);

const HashSet<std::string>& FusedSelfAttentionQueryMulKeyAndValueOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "alpha", "head_size", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FusedSelfAttentionQueryMulKeyAndValueOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "alpha") {
    return CastAttr(&internal_->alpha);
  }
  if(attr_name == "head_size") {
    return CastAttr(&internal_->head_size);
  }
  return Error::RuntimeError() << "fused_self_attention_query_mul_key_and_value op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.fused_self_attention_query_mul_key_and_value", FusedSelfAttentionQueryMulKeyAndValueOpInterpCtxImpl<schema::FusedSelfAttentionQueryMulKeyAndValueOp>);

const HashSet<std::string>& FusedTrilScaleSoftmaxMaskScaleGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "diagonal", "mask_scale_value", "tril_scale_value", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FusedTrilScaleSoftmaxMaskScaleGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "diagonal") {
    return CastAttr(&internal_->diagonal);
  }
  if(attr_name == "mask_scale_value") {
    return CastAttr(&internal_->mask_scale_value);
  }
  if(attr_name == "tril_scale_value") {
    return CastAttr(&internal_->tril_scale_value);
  }
  return Error::RuntimeError() << "fused_tril_scale_softmax_mask_scale_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.fused_tril_scale_softmax_mask_scale_grad", FusedTrilScaleSoftmaxMaskScaleGradOpInterpCtxImpl<schema::FusedTrilScaleSoftmaxMaskScaleGradOp>);

const HashSet<std::string>& FusedTrilScaleSoftmaxMaskScaleOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "diagonal", "mask_scale_value", "tril_fill_value", "tril_scale_value", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> FusedTrilScaleSoftmaxMaskScaleOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "diagonal") {
    return CastAttr(&internal_->diagonal);
  }
  if(attr_name == "mask_scale_value") {
    return CastAttr(&internal_->mask_scale_value);
  }
  if(attr_name == "tril_fill_value") {
    return CastAttr(&internal_->tril_fill_value);
  }
  if(attr_name == "tril_scale_value") {
    return CastAttr(&internal_->tril_scale_value);
  }
  return Error::RuntimeError() << "fused_tril_scale_softmax_mask_scale op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.fused_tril_scale_softmax_mask_scale", FusedTrilScaleSoftmaxMaskScaleOpInterpCtxImpl<schema::FusedTrilScaleSoftmaxMaskScaleOp>);

const HashSet<std::string>& GatherNdOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> GatherNdOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "gather_nd op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.gather_nd", GatherNdOpInterpCtxImpl<schema::GatherNdOp>);

const HashSet<std::string>& GatherOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> GatherOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  return Error::RuntimeError() << "gather op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.gather", GatherOpInterpCtxImpl<schema::GatherOp>);

const HashSet<std::string>& GeluGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> GeluGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "gelu_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.gelu_grad", GeluGradOpInterpCtxImpl<schema::GeluGradOp>);

const HashSet<std::string>& GeluOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> GeluOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "gelu op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.gelu", GeluOpInterpCtxImpl<schema::GeluOp>);

const HashSet<std::string>& GenTensorBufferOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_type", "dynamic_out", "shape", "shape_list", "value_list", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> GenTensorBufferOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_type") {
    return CastAttr(&internal_->data_type);
  }
  if(attr_name == "dynamic_out") {
    return CastAttr(&internal_->dynamic_out);
  }
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  if(attr_name == "shape_list") {
    return CastAttr(&internal_->shape_list);
  }
  if(attr_name == "value_list") {
    return CastAttr(&internal_->value_list);
  }
  return Error::RuntimeError() << "gen_tensor_buffer op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.gen_tensor_buffer", GenTensorBufferOpInterpCtxImpl<schema::GenTensorBufferOp>);

const HashSet<std::string>& GenerateRandomBatchPermutationIndicesOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "seed", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> GenerateRandomBatchPermutationIndicesOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "seed") {
    return CastAttr(&internal_->seed);
  }
  return Error::RuntimeError() << "generate_random_batch_permutation_indices op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.generate_random_batch_permutation_indices", GenerateRandomBatchPermutationIndicesOpInterpCtxImpl<schema::GenerateRandomBatchPermutationIndicesOp>);

const HashSet<std::string>& GridSampleGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "align_corners", "interpolation_mode", "padding_mode", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> GridSampleGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "align_corners") {
    return CastAttr(&internal_->align_corners);
  }
  if(attr_name == "interpolation_mode") {
    return CastAttr(&internal_->interpolation_mode);
  }
  if(attr_name == "padding_mode") {
    return CastAttr(&internal_->padding_mode);
  }
  return Error::RuntimeError() << "grid_sample_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.grid_sample_grad", GridSampleGradOpInterpCtxImpl<schema::GridSampleGradOp>);

const HashSet<std::string>& GridSampleOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "align_corners", "interpolation_mode", "padding_mode", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> GridSampleOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "align_corners") {
    return CastAttr(&internal_->align_corners);
  }
  if(attr_name == "interpolation_mode") {
    return CastAttr(&internal_->interpolation_mode);
  }
  if(attr_name == "padding_mode") {
    return CastAttr(&internal_->padding_mode);
  }
  return Error::RuntimeError() << "grid_sample op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.grid_sample", GridSampleOpInterpCtxImpl<schema::GridSampleOp>);

const HashSet<std::string>& HardsigmoidGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> HardsigmoidGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "hardsigmoid_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.hardsigmoid_grad", HardsigmoidGradOpInterpCtxImpl<schema::HardsigmoidGradOp>);

const HashSet<std::string>& HardsigmoidOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> HardsigmoidOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "hardsigmoid op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.hardsigmoid", HardsigmoidOpInterpCtxImpl<schema::HardsigmoidOp>);

const HashSet<std::string>& HardswishGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> HardswishGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "hardswish_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.hardswish_grad", HardswishGradOpInterpCtxImpl<schema::HardswishGradOp>);

const HashSet<std::string>& HardswishOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> HardswishOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "hardswish op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.hardswish", HardswishOpInterpCtxImpl<schema::HardswishOp>);

const HashSet<std::string>& HardtanhGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "max_val", "min_val", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> HardtanhGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "max_val") {
    return CastAttr(&internal_->max_val);
  }
  if(attr_name == "min_val") {
    return CastAttr(&internal_->min_val);
  }
  return Error::RuntimeError() << "hardtanh_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.hardtanh_grad", HardtanhGradOpInterpCtxImpl<schema::HardtanhGradOp>);

const HashSet<std::string>& HardtanhOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "max_val", "min_val", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> HardtanhOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "max_val") {
    return CastAttr(&internal_->max_val);
  }
  if(attr_name == "min_val") {
    return CastAttr(&internal_->min_val);
  }
  return Error::RuntimeError() << "hardtanh op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.hardtanh", HardtanhOpInterpCtxImpl<schema::HardtanhOp>);

const HashSet<std::string>& HierarchicalParallelCastLikeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> HierarchicalParallelCastLikeOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "hierarchical_parallel_cast_like op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.hierarchical_parallel_cast_like", HierarchicalParallelCastLikeOpInterpCtxImpl<schema::HierarchicalParallelCastLikeOp>);

const HashSet<std::string>& HierarchicalParallelCastOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "grad_mode", "grad_nd_sbp", "nd_sbp", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> HierarchicalParallelCastOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "grad_mode") {
    return CastAttr(&internal_->grad_mode);
  }
  if(attr_name == "grad_nd_sbp") {
    return CastAttr(&internal_->grad_nd_sbp);
  }
  if(attr_name == "nd_sbp") {
    return CastAttr(&internal_->nd_sbp);
  }
  return Error::RuntimeError() << "hierarchical_parallel_cast op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.hierarchical_parallel_cast", HierarchicalParallelCastOpInterpCtxImpl<schema::HierarchicalParallelCastOp>);

const HashSet<std::string>& IdentityBufferOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "buffer_size", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> IdentityBufferOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "buffer_size") {
    return CastAttr(&internal_->buffer_size);
  }
  return Error::RuntimeError() << "identity_buffer op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.identity_buffer", IdentityBufferOpInterpCtxImpl<schema::IdentityBufferOp>);

const HashSet<std::string>& IdentityOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> IdentityOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "identity op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.identity", IdentityOpInterpCtxImpl<schema::IdentityOp>);

const HashSet<std::string>& ImageBatchAlignOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "alignment", "data_type", "dynamic_out", "shape", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ImageBatchAlignOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "alignment") {
    return CastAttr(&internal_->alignment);
  }
  if(attr_name == "data_type") {
    return CastAttr(&internal_->data_type);
  }
  if(attr_name == "dynamic_out") {
    return CastAttr(&internal_->dynamic_out);
  }
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  return Error::RuntimeError() << "image_batch_align op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.image_batch_align", ImageBatchAlignOpInterpCtxImpl<schema::ImageBatchAlignOp>);

const HashSet<std::string>& ImageDecodeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "color_space", "data_type", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ImageDecodeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "color_space") {
    return CastAttr(&internal_->color_space);
  }
  if(attr_name == "data_type") {
    return CastAttr(&internal_->data_type);
  }
  return Error::RuntimeError() << "image_decode op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.image_decode", ImageDecodeOpInterpCtxImpl<schema::ImageDecodeOp>);

const HashSet<std::string>& ImageFlipOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ImageFlipOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "image_flip op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.image_flip", ImageFlipOpInterpCtxImpl<schema::ImageFlipOp>);

const HashSet<std::string>& ImageNormalizeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "mean", "std", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ImageNormalizeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "mean") {
    return CastAttr(&internal_->mean);
  }
  if(attr_name == "std") {
    return CastAttr(&internal_->std);
  }
  return Error::RuntimeError() << "image_normalize op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.image_normalize", ImageNormalizeOpInterpCtxImpl<schema::ImageNormalizeOp>);

const HashSet<std::string>& ImageRandomCropOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "has_seed", "num_attempts", "random_area", "random_aspect_ratio", "seed", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ImageRandomCropOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "has_seed") {
    return CastAttr(&internal_->has_seed);
  }
  if(attr_name == "num_attempts") {
    return CastAttr(&internal_->num_attempts);
  }
  if(attr_name == "random_area") {
    return CastAttr(&internal_->random_area);
  }
  if(attr_name == "random_aspect_ratio") {
    return CastAttr(&internal_->random_aspect_ratio);
  }
  if(attr_name == "seed") {
    return CastAttr(&internal_->seed);
  }
  return Error::RuntimeError() << "image_random_crop op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.image_random_crop", ImageRandomCropOpInterpCtxImpl<schema::ImageRandomCropOp>);

const HashSet<std::string>& ImageResizeKeepAspectRatioOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "interpolation_type", "max_size", "min_size", "resize_longer", "target_size", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ImageResizeKeepAspectRatioOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "interpolation_type") {
    return CastAttr(&internal_->interpolation_type);
  }
  if(attr_name == "max_size") {
    return CastAttr(&internal_->max_size);
  }
  if(attr_name == "min_size") {
    return CastAttr(&internal_->min_size);
  }
  if(attr_name == "resize_longer") {
    return CastAttr(&internal_->resize_longer);
  }
  if(attr_name == "target_size") {
    return CastAttr(&internal_->target_size);
  }
  return Error::RuntimeError() << "image_resize_keep_aspect_ratio op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.image_resize_keep_aspect_ratio", ImageResizeKeepAspectRatioOpInterpCtxImpl<schema::ImageResizeKeepAspectRatioOp>);

const HashSet<std::string>& ImageResizeToFixedOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "channels", "data_type", "interpolation_type", "target_height", "target_width", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ImageResizeToFixedOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "channels") {
    return CastAttr(&internal_->channels);
  }
  if(attr_name == "data_type") {
    return CastAttr(&internal_->data_type);
  }
  if(attr_name == "interpolation_type") {
    return CastAttr(&internal_->interpolation_type);
  }
  if(attr_name == "target_height") {
    return CastAttr(&internal_->target_height);
  }
  if(attr_name == "target_width") {
    return CastAttr(&internal_->target_width);
  }
  return Error::RuntimeError() << "image_resize_to_fixed op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.image_resize_to_fixed", ImageResizeToFixedOpInterpCtxImpl<schema::ImageResizeToFixedOp>);

const HashSet<std::string>& ImageTargetResizeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "max_size", "target_size", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ImageTargetResizeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "max_size") {
    return CastAttr(&internal_->max_size);
  }
  if(attr_name == "target_size") {
    return CastAttr(&internal_->target_size);
  }
  return Error::RuntimeError() << "image_target_resize op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.image_target_resize", ImageTargetResizeOpInterpCtxImpl<schema::ImageTargetResizeOp>);

const HashSet<std::string>& InTopKOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "k", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> InTopKOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "k") {
    return CastAttr(&internal_->k);
  }
  return Error::RuntimeError() << "in_top_k op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.in_top_k", InTopKOpInterpCtxImpl<schema::InTopKOp>);

const HashSet<std::string>& IndexedSlicesAdamUpdateOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "amsgrad", "beta1", "beta2", "do_bias_correction", "epsilon", "learning_rate_val", "weight_decay", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> IndexedSlicesAdamUpdateOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "amsgrad") {
    return CastAttr(&internal_->amsgrad);
  }
  if(attr_name == "beta1") {
    return CastAttr(&internal_->beta1);
  }
  if(attr_name == "beta2") {
    return CastAttr(&internal_->beta2);
  }
  if(attr_name == "do_bias_correction") {
    return CastAttr(&internal_->do_bias_correction);
  }
  if(attr_name == "epsilon") {
    return CastAttr(&internal_->epsilon);
  }
  if(attr_name == "learning_rate_val") {
    return CastAttr(&internal_->learning_rate_val);
  }
  if(attr_name == "weight_decay") {
    return CastAttr(&internal_->weight_decay);
  }
  return Error::RuntimeError() << "indexed_slices_adam_update op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.indexed_slices_adam_update", IndexedSlicesAdamUpdateOpInterpCtxImpl<schema::IndexedSlicesAdamUpdateOp>);

const HashSet<std::string>& IndexedSlicesMomentumUpdateOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "beta", "weight_decay", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> IndexedSlicesMomentumUpdateOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "beta") {
    return CastAttr(&internal_->beta);
  }
  if(attr_name == "weight_decay") {
    return CastAttr(&internal_->weight_decay);
  }
  return Error::RuntimeError() << "indexed_slices_momentum_update op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.indexed_slices_momentum_update", IndexedSlicesMomentumUpdateOpInterpCtxImpl<schema::IndexedSlicesMomentumUpdateOp>);

const HashSet<std::string>& IndexedSlicesReduceSumOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> IndexedSlicesReduceSumOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "indexed_slices_reduce_sum op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.indexed_slices_reduce_sum", IndexedSlicesReduceSumOpInterpCtxImpl<schema::IndexedSlicesReduceSumOp>);

const HashSet<std::string>& IndexedSlicesSgdUpdateOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "weight_decay", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> IndexedSlicesSgdUpdateOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "weight_decay") {
    return CastAttr(&internal_->weight_decay);
  }
  return Error::RuntimeError() << "indexed_slices_sgd_update op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.indexed_slices_sgd_update", IndexedSlicesSgdUpdateOpInterpCtxImpl<schema::IndexedSlicesSgdUpdateOp>);

const HashSet<std::string>& KlDivLossGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "log_target", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> KlDivLossGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "log_target") {
    return CastAttr(&internal_->log_target);
  }
  return Error::RuntimeError() << "kl_div_loss_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.kl_div_loss_grad", KlDivLossGradOpInterpCtxImpl<schema::KlDivLossGradOp>);

const HashSet<std::string>& KlDivLossOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "log_target", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> KlDivLossOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "log_target") {
    return CastAttr(&internal_->log_target);
  }
  return Error::RuntimeError() << "kl_div_loss op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.kl_div_loss", KlDivLossOpInterpCtxImpl<schema::KlDivLossOp>);

const HashSet<std::string>& L1L2RegularizeGradientOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "l1", "l2", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> L1L2RegularizeGradientOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "l1") {
    return CastAttr(&internal_->l1);
  }
  if(attr_name == "l2") {
    return CastAttr(&internal_->l2);
  }
  return Error::RuntimeError() << "l1_l2_regularize_gradient op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.l1_l2_regularize_gradient", L1L2RegularizeGradientOpInterpCtxImpl<schema::L1L2RegularizeGradientOp>);

const HashSet<std::string>& L2NormalizeGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "epsilon", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> L2NormalizeGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "epsilon") {
    return CastAttr(&internal_->epsilon);
  }
  return Error::RuntimeError() << "l2_normalize_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.l2_normalize_grad", L2NormalizeGradOpInterpCtxImpl<schema::L2NormalizeGradOp>);

const HashSet<std::string>& L2NormalizeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "epsilon", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> L2NormalizeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "epsilon") {
    return CastAttr(&internal_->epsilon);
  }
  return Error::RuntimeError() << "l2_normalize op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.l2_normalize", L2NormalizeOpInterpCtxImpl<schema::L2NormalizeOp>);

const HashSet<std::string>& LambUpdateOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "beta1", "beta2", "epsilon", "l1", "l2", "scale", "weight_decay", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LambUpdateOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "beta1") {
    return CastAttr(&internal_->beta1);
  }
  if(attr_name == "beta2") {
    return CastAttr(&internal_->beta2);
  }
  if(attr_name == "epsilon") {
    return CastAttr(&internal_->epsilon);
  }
  if(attr_name == "l1") {
    return CastAttr(&internal_->l1);
  }
  if(attr_name == "l2") {
    return CastAttr(&internal_->l2);
  }
  if(attr_name == "scale") {
    return CastAttr(&internal_->scale);
  }
  if(attr_name == "weight_decay") {
    return CastAttr(&internal_->weight_decay);
  }
  return Error::RuntimeError() << "lamb_update op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.lamb_update", LambUpdateOpInterpCtxImpl<schema::LambUpdateOp>);

const HashSet<std::string>& LarsUpdateOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "epsilon", "l1", "l2", "lars_coefficient", "momentum_beta", "scale", "weight_decay", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LarsUpdateOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "epsilon") {
    return CastAttr(&internal_->epsilon);
  }
  if(attr_name == "l1") {
    return CastAttr(&internal_->l1);
  }
  if(attr_name == "l2") {
    return CastAttr(&internal_->l2);
  }
  if(attr_name == "lars_coefficient") {
    return CastAttr(&internal_->lars_coefficient);
  }
  if(attr_name == "momentum_beta") {
    return CastAttr(&internal_->momentum_beta);
  }
  if(attr_name == "scale") {
    return CastAttr(&internal_->scale);
  }
  if(attr_name == "weight_decay") {
    return CastAttr(&internal_->weight_decay);
  }
  return Error::RuntimeError() << "lars_update op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.lars_update", LarsUpdateOpInterpCtxImpl<schema::LarsUpdateOp>);

const HashSet<std::string>& LayerNormGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "begin_norm_axis", "epsilon", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LayerNormGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "begin_norm_axis") {
    return CastAttr(&internal_->begin_norm_axis);
  }
  if(attr_name == "epsilon") {
    return CastAttr(&internal_->epsilon);
  }
  return Error::RuntimeError() << "layer_norm_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.layer_norm_grad", LayerNormGradOpInterpCtxImpl<schema::LayerNormGradOp>);

const HashSet<std::string>& LayerNormOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "begin_norm_axis", "begin_params_axis", "center", "epsilon", "scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LayerNormOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "begin_norm_axis") {
    return CastAttr(&internal_->begin_norm_axis);
  }
  if(attr_name == "begin_params_axis") {
    return CastAttr(&internal_->begin_params_axis);
  }
  if(attr_name == "center") {
    return CastAttr(&internal_->center);
  }
  if(attr_name == "epsilon") {
    return CastAttr(&internal_->epsilon);
  }
  if(attr_name == "scale") {
    return CastAttr(&internal_->scale);
  }
  return Error::RuntimeError() << "layer_norm op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.layer_norm", LayerNormOpInterpCtxImpl<schema::LayerNormOp>);

const HashSet<std::string>& LayerNormParamGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "begin_params_axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LayerNormParamGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "begin_params_axis") {
    return CastAttr(&internal_->begin_params_axis);
  }
  return Error::RuntimeError() << "layer_norm_param_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.layer_norm_param_grad", LayerNormParamGradOpInterpCtxImpl<schema::LayerNormParamGradOp>);

const HashSet<std::string>& LeakyReluGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "alpha", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LeakyReluGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "alpha") {
    return CastAttr(&internal_->alpha);
  }
  return Error::RuntimeError() << "leaky_relu_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.leaky_relu_grad", LeakyReluGradOpInterpCtxImpl<schema::LeakyReluGradOp>);

const HashSet<std::string>& LeakyReluOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "alpha", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LeakyReluOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "alpha") {
    return CastAttr(&internal_->alpha);
  }
  return Error::RuntimeError() << "leaky_relu op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.leaky_relu", LeakyReluOpInterpCtxImpl<schema::LeakyReluOp>);

const HashSet<std::string>& LgammaGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LgammaGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "lgamma_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.lgamma_grad", LgammaGradOpInterpCtxImpl<schema::LgammaGradOp>);

const HashSet<std::string>& LgammaOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LgammaOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "lgamma op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.lgamma", LgammaOpInterpCtxImpl<schema::LgammaOp>);

const HashSet<std::string>& Log1pGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> Log1pGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "log1p_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.log1p_grad", Log1pGradOpInterpCtxImpl<schema::Log1pGradOp>);

const HashSet<std::string>& Log1pOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> Log1pOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "log1p op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.log1p", Log1pOpInterpCtxImpl<schema::Log1pOp>);

const HashSet<std::string>& LogGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LogGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "log_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.log_grad", LogGradOpInterpCtxImpl<schema::LogGradOp>);

const HashSet<std::string>& LogOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LogOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "log op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.log", LogOpInterpCtxImpl<schema::LogOp>);

const HashSet<std::string>& LogSigmoidGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LogSigmoidGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "log_sigmoid_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.log_sigmoid_grad", LogSigmoidGradOpInterpCtxImpl<schema::LogSigmoidGradOp>);

const HashSet<std::string>& LogSigmoidOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LogSigmoidOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "log_sigmoid op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.log_sigmoid", LogSigmoidOpInterpCtxImpl<schema::LogSigmoidOp>);

const HashSet<std::string>& LogSoftmaxGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LogSoftmaxGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "log_softmax_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.log_softmax_grad", LogSoftmaxGradOpInterpCtxImpl<schema::LogSoftmaxGradOp>);

const HashSet<std::string>& LogSoftmaxOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LogSoftmaxOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "log_softmax op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.log_softmax", LogSoftmaxOpInterpCtxImpl<schema::LogSoftmaxOp>);

const HashSet<std::string>& LogicalNotOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LogicalNotOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "logical_not op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.logical_not", LogicalNotOpInterpCtxImpl<schema::LogicalNotOp>);

const HashSet<std::string>& LogicalSliceAssignOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "start", "step", "stop", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LogicalSliceAssignOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "start") {
    return CastAttr(&internal_->start);
  }
  if(attr_name == "step") {
    return CastAttr(&internal_->step);
  }
  if(attr_name == "stop") {
    return CastAttr(&internal_->stop);
  }
  return Error::RuntimeError() << "logical_slice_assign op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.logical_slice_assign", LogicalSliceAssignOpInterpCtxImpl<schema::LogicalSliceAssignOp>);

const HashSet<std::string>& LogicalSliceOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "start", "step", "stop", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> LogicalSliceOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "start") {
    return CastAttr(&internal_->start);
  }
  if(attr_name == "step") {
    return CastAttr(&internal_->step);
  }
  if(attr_name == "stop") {
    return CastAttr(&internal_->stop);
  }
  return Error::RuntimeError() << "logical_slice op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.logical_slice", LogicalSliceOpInterpCtxImpl<schema::LogicalSliceOp>);

const HashSet<std::string>& MaskedFillOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MaskedFillOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "masked_fill op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.masked_fill", MaskedFillOpInterpCtxImpl<schema::MaskedFillOp>);

const HashSet<std::string>& MatmulOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "alpha", "transpose_a", "transpose_b", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MatmulOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "alpha") {
    return CastAttr(&internal_->alpha);
  }
  if(attr_name == "transpose_a") {
    return CastAttr(&internal_->transpose_a);
  }
  if(attr_name == "transpose_b") {
    return CastAttr(&internal_->transpose_b);
  }
  return Error::RuntimeError() << "matmul op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.matmul", MatmulOpInterpCtxImpl<schema::MatmulOp>);

const HashSet<std::string>& MaxPool1DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "count_include_pad", "data_format", "divisor_override", "kernel_size", "padding", "stride", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MaxPool1DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "count_include_pad") {
    return CastAttr(&internal_->count_include_pad);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "divisor_override") {
    return CastAttr(&internal_->divisor_override);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "stride") {
    return CastAttr(&internal_->stride);
  }
  return Error::RuntimeError() << "maxpool_1d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.maxpool_1d_grad", MaxPool1DGradOpInterpCtxImpl<schema::MaxPool1DGradOp>);

const HashSet<std::string>& MaxPool1DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "count_include_pad", "data_format", "divisor_override", "kernel_size", "padding", "stride", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MaxPool1DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "count_include_pad") {
    return CastAttr(&internal_->count_include_pad);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "divisor_override") {
    return CastAttr(&internal_->divisor_override);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "stride") {
    return CastAttr(&internal_->stride);
  }
  return Error::RuntimeError() << "maxpool_1d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.maxpool_1d", MaxPool1DOpInterpCtxImpl<schema::MaxPool1DOp>);

const HashSet<std::string>& MaxPool2DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "count_include_pad", "data_format", "divisor_override", "kernel_size", "padding", "stride", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MaxPool2DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "count_include_pad") {
    return CastAttr(&internal_->count_include_pad);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "divisor_override") {
    return CastAttr(&internal_->divisor_override);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "stride") {
    return CastAttr(&internal_->stride);
  }
  return Error::RuntimeError() << "maxpool_2d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.maxpool_2d_grad", MaxPool2DGradOpInterpCtxImpl<schema::MaxPool2DGradOp>);

const HashSet<std::string>& MaxPool2DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "count_include_pad", "data_format", "divisor_override", "kernel_size", "padding", "stride", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MaxPool2DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "count_include_pad") {
    return CastAttr(&internal_->count_include_pad);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "divisor_override") {
    return CastAttr(&internal_->divisor_override);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "stride") {
    return CastAttr(&internal_->stride);
  }
  return Error::RuntimeError() << "maxpool_2d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.maxpool_2d", MaxPool2DOpInterpCtxImpl<schema::MaxPool2DOp>);

const HashSet<std::string>& MaxPool3DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "count_include_pad", "data_format", "divisor_override", "kernel_size", "padding", "stride", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MaxPool3DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "count_include_pad") {
    return CastAttr(&internal_->count_include_pad);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "divisor_override") {
    return CastAttr(&internal_->divisor_override);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "stride") {
    return CastAttr(&internal_->stride);
  }
  return Error::RuntimeError() << "maxpool_3d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.maxpool_3d_grad", MaxPool3DGradOpInterpCtxImpl<schema::MaxPool3DGradOp>);

const HashSet<std::string>& MaxPool3DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "count_include_pad", "data_format", "divisor_override", "kernel_size", "padding", "stride", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MaxPool3DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "count_include_pad") {
    return CastAttr(&internal_->count_include_pad);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "divisor_override") {
    return CastAttr(&internal_->divisor_override);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "stride") {
    return CastAttr(&internal_->stride);
  }
  return Error::RuntimeError() << "maxpool_3d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.maxpool_3d", MaxPool3DOpInterpCtxImpl<schema::MaxPool3DOp>);

const HashSet<std::string>& MegatronGptMmapDataLoaderOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "batch_size", "data_file_prefix", "dtype", "label_length", "nd_sbp", "num_samples", "random_seed", "seq_length", "shuffle", "split_index", "split_sizes", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MegatronGptMmapDataLoaderOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "batch_size") {
    return CastAttr(&internal_->batch_size);
  }
  if(attr_name == "data_file_prefix") {
    return CastAttr(&internal_->data_file_prefix);
  }
  if(attr_name == "dtype") {
    return CastAttr(&internal_->dtype);
  }
  if(attr_name == "label_length") {
    return CastAttr(&internal_->label_length);
  }
  if(attr_name == "nd_sbp") {
    return CastAttr(&internal_->nd_sbp);
  }
  if(attr_name == "num_samples") {
    return CastAttr(&internal_->num_samples);
  }
  if(attr_name == "random_seed") {
    return CastAttr(&internal_->random_seed);
  }
  if(attr_name == "seq_length") {
    return CastAttr(&internal_->seq_length);
  }
  if(attr_name == "shuffle") {
    return CastAttr(&internal_->shuffle);
  }
  if(attr_name == "split_index") {
    return CastAttr(&internal_->split_index);
  }
  if(attr_name == "split_sizes") {
    return CastAttr(&internal_->split_sizes);
  }
  return Error::RuntimeError() << "megatron_gpt_mmap_data_loader op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.megatron_gpt_mmap_data_loader", MegatronGptMmapDataLoaderOpInterpCtxImpl<schema::MegatronGptMmapDataLoaderOp>);

const HashSet<std::string>& MinMaxObserverOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "per_layer_quantization", "quantization_bit", "quantization_formula", "quantization_scheme", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MinMaxObserverOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "per_layer_quantization") {
    return CastAttr(&internal_->per_layer_quantization);
  }
  if(attr_name == "quantization_bit") {
    return CastAttr(&internal_->quantization_bit);
  }
  if(attr_name == "quantization_formula") {
    return CastAttr(&internal_->quantization_formula);
  }
  if(attr_name == "quantization_scheme") {
    return CastAttr(&internal_->quantization_scheme);
  }
  return Error::RuntimeError() << "min_max_observer op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.min_max_observer", MinMaxObserverOpInterpCtxImpl<schema::MinMaxObserverOp>);

const HashSet<std::string>& MishGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MishGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "mish_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.mish_grad", MishGradOpInterpCtxImpl<schema::MishGradOp>);

const HashSet<std::string>& MishOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MishOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "mish op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.mish", MishOpInterpCtxImpl<schema::MishOp>);

const HashSet<std::string>& MomentumUpdateOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "beta", "l1", "l2", "learning_rate_val", "scale", "weight_decay", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MomentumUpdateOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "beta") {
    return CastAttr(&internal_->beta);
  }
  if(attr_name == "l1") {
    return CastAttr(&internal_->l1);
  }
  if(attr_name == "l2") {
    return CastAttr(&internal_->l2);
  }
  if(attr_name == "learning_rate_val") {
    return CastAttr(&internal_->learning_rate_val);
  }
  if(attr_name == "scale") {
    return CastAttr(&internal_->scale);
  }
  if(attr_name == "weight_decay") {
    return CastAttr(&internal_->weight_decay);
  }
  return Error::RuntimeError() << "momentum_update op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.momentum_update", MomentumUpdateOpInterpCtxImpl<schema::MomentumUpdateOp>);

const HashSet<std::string>& MovingAverageMinMaxObserverOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "momentum", "quantization_bit", "quantization_formula", "quantization_scheme", "stop_update_after_iters", "training", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MovingAverageMinMaxObserverOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "momentum") {
    return CastAttr(&internal_->momentum);
  }
  if(attr_name == "quantization_bit") {
    return CastAttr(&internal_->quantization_bit);
  }
  if(attr_name == "quantization_formula") {
    return CastAttr(&internal_->quantization_formula);
  }
  if(attr_name == "quantization_scheme") {
    return CastAttr(&internal_->quantization_scheme);
  }
  if(attr_name == "stop_update_after_iters") {
    return CastAttr(&internal_->stop_update_after_iters);
  }
  if(attr_name == "training") {
    return CastAttr(&internal_->training);
  }
  return Error::RuntimeError() << "moving_average_min_max_observer op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.moving_average_min_max_observer", MovingAverageMinMaxObserverOpInterpCtxImpl<schema::MovingAverageMinMaxObserverOp>);

const HashSet<std::string>& MultiCountNotFiniteOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MultiCountNotFiniteOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "multi_count_not_finite op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.multi_count_not_finite", MultiCountNotFiniteOpInterpCtxImpl<schema::MultiCountNotFiniteOp>);

const HashSet<std::string>& MultiSquareSumOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MultiSquareSumOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "multi_square_sum op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.multi_square_sum", MultiSquareSumOpInterpCtxImpl<schema::MultiSquareSumOp>);

const HashSet<std::string>& MultiplyOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> MultiplyOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "multiply op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.multiply", MultiplyOpInterpCtxImpl<schema::MultiplyOp>);

const HashSet<std::string>& NarrowGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dim", "length", "start", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> NarrowGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dim") {
    return CastAttr(&internal_->dim);
  }
  if(attr_name == "length") {
    return CastAttr(&internal_->length);
  }
  if(attr_name == "start") {
    return CastAttr(&internal_->start);
  }
  return Error::RuntimeError() << "narrow_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.narrow_grad", NarrowGradOpInterpCtxImpl<schema::NarrowGradOp>);

const HashSet<std::string>& NarrowOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dim", "length", "start", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> NarrowOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dim") {
    return CastAttr(&internal_->dim);
  }
  if(attr_name == "length") {
    return CastAttr(&internal_->length);
  }
  if(attr_name == "start") {
    return CastAttr(&internal_->start);
  }
  return Error::RuntimeError() << "narrow op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.narrow", NarrowOpInterpCtxImpl<schema::NarrowOp>);

const HashSet<std::string>& NegativeGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> NegativeGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "negative_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.negative_grad", NegativeGradOpInterpCtxImpl<schema::NegativeGradOp>);

const HashSet<std::string>& NegativeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> NegativeOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "negative op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.negative", NegativeOpInterpCtxImpl<schema::NegativeOp>);

const HashSet<std::string>& NllGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ignore_index", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> NllGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ignore_index") {
    return CastAttr(&internal_->ignore_index);
  }
  return Error::RuntimeError() << "nll_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.nll_grad", NllGradOpInterpCtxImpl<schema::NllGradOp>);

const HashSet<std::string>& NllOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ignore_index", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> NllOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ignore_index") {
    return CastAttr(&internal_->ignore_index);
  }
  return Error::RuntimeError() << "nll op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.nll", NllOpInterpCtxImpl<schema::NllOp>);

const HashSet<std::string>& NmsOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "iou_threshold", "keep_n", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> NmsOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "iou_threshold") {
    return CastAttr(&internal_->iou_threshold);
  }
  if(attr_name == "keep_n") {
    return CastAttr(&internal_->keep_n);
  }
  return Error::RuntimeError() << "nms op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.nms", NmsOpInterpCtxImpl<schema::NmsOp>);

const HashSet<std::string>& NormalOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dtype", "mean", "nd_sbp", "seed", "shape", "std", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> NormalOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dtype") {
    return CastAttr(&internal_->dtype);
  }
  if(attr_name == "mean") {
    return CastAttr(&internal_->mean);
  }
  if(attr_name == "nd_sbp") {
    return CastAttr(&internal_->nd_sbp);
  }
  if(attr_name == "seed") {
    return CastAttr(&internal_->seed);
  }
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  if(attr_name == "std") {
    return CastAttr(&internal_->std);
  }
  return Error::RuntimeError() << "normal op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.normal", NormalOpInterpCtxImpl<schema::NormalOp>);

const HashSet<std::string>& NormalizationAddReluGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "epsilon", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> NormalizationAddReluGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "epsilon") {
    return CastAttr(&internal_->epsilon);
  }
  return Error::RuntimeError() << "normalization_add_relu_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.normalization_add_relu_grad", NormalizationAddReluGradOpInterpCtxImpl<schema::NormalizationAddReluGradOp>);

const HashSet<std::string>& NormalizationGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "epsilon", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> NormalizationGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "epsilon") {
    return CastAttr(&internal_->epsilon);
  }
  return Error::RuntimeError() << "normalization_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.normalization_grad", NormalizationGradOpInterpCtxImpl<schema::NormalizationGradOp>);

const HashSet<std::string>& NormalizationOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "epsilon", "momentum", "training", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> NormalizationOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "epsilon") {
    return CastAttr(&internal_->epsilon);
  }
  if(attr_name == "momentum") {
    return CastAttr(&internal_->momentum);
  }
  if(attr_name == "training") {
    return CastAttr(&internal_->training);
  }
  return Error::RuntimeError() << "normalization op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.normalization", NormalizationOpInterpCtxImpl<schema::NormalizationOp>);

const HashSet<std::string>& NvtxEndOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "mark_prefix", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> NvtxEndOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "mark_prefix") {
    return CastAttr(&internal_->mark_prefix);
  }
  return Error::RuntimeError() << "nvtx_end op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.nvtx_end", NvtxEndOpInterpCtxImpl<schema::NvtxEndOp>);

const HashSet<std::string>& NvtxStartOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "mark_prefix", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> NvtxStartOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "mark_prefix") {
    return CastAttr(&internal_->mark_prefix);
  }
  return Error::RuntimeError() << "nvtx_start op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.nvtx_start", NvtxStartOpInterpCtxImpl<schema::NvtxStartOp>);

const HashSet<std::string>& OFRecordReaderOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "batch_size", "data_dir", "data_part_num", "nd_sbp", "part_name_prefix", "part_name_suffix_length", "random_shuffle", "seed", "shuffle_after_epoch", "shuffle_buffer_size", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> OFRecordReaderOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "batch_size") {
    return CastAttr(&internal_->batch_size);
  }
  if(attr_name == "data_dir") {
    return CastAttr(&internal_->data_dir);
  }
  if(attr_name == "data_part_num") {
    return CastAttr(&internal_->data_part_num);
  }
  if(attr_name == "nd_sbp") {
    return CastAttr(&internal_->nd_sbp);
  }
  if(attr_name == "part_name_prefix") {
    return CastAttr(&internal_->part_name_prefix);
  }
  if(attr_name == "part_name_suffix_length") {
    return CastAttr(&internal_->part_name_suffix_length);
  }
  if(attr_name == "random_shuffle") {
    return CastAttr(&internal_->random_shuffle);
  }
  if(attr_name == "seed") {
    return CastAttr(&internal_->seed);
  }
  if(attr_name == "shuffle_after_epoch") {
    return CastAttr(&internal_->shuffle_after_epoch);
  }
  if(attr_name == "shuffle_buffer_size") {
    return CastAttr(&internal_->shuffle_buffer_size);
  }
  return Error::RuntimeError() << "OFRecordReader op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.OFRecordReader", OFRecordReaderOpInterpCtxImpl<schema::OFRecordReaderOp>);

const HashSet<std::string>& ObjectBboxFlipOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ObjectBboxFlipOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "object_bbox_flip op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.object_bbox_flip", ObjectBboxFlipOpInterpCtxImpl<schema::ObjectBboxFlipOp>);

const HashSet<std::string>& ObjectBboxScaleOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ObjectBboxScaleOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "object_bbox_scale op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.object_bbox_scale", ObjectBboxScaleOpInterpCtxImpl<schema::ObjectBboxScaleOp>);

const HashSet<std::string>& ObjectSegmentationPolygonFlipOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ObjectSegmentationPolygonFlipOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "object_segmentation_polygon_flip op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.object_segmentation_polygon_flip", ObjectSegmentationPolygonFlipOpInterpCtxImpl<schema::ObjectSegmentationPolygonFlipOp>);

const HashSet<std::string>& ObjectSegmentationPolygonScaleOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ObjectSegmentationPolygonScaleOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "object_segmentation_polygon_scale op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.object_segmentation_polygon_scale", ObjectSegmentationPolygonScaleOpInterpCtxImpl<schema::ObjectSegmentationPolygonScaleOp>);

const HashSet<std::string>& ObjectSegmentationPolygonToMaskOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ObjectSegmentationPolygonToMaskOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "object_segmentation_polygon_to_mask op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.object_segmentation_polygon_to_mask", ObjectSegmentationPolygonToMaskOpInterpCtxImpl<schema::ObjectSegmentationPolygonToMaskOp>);

const HashSet<std::string>& OfrecordBytesDecoderOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "name", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> OfrecordBytesDecoderOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "name") {
    return CastAttr(&internal_->name);
  }
  return Error::RuntimeError() << "ofrecord_bytes_decoder op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.ofrecord_bytes_decoder", OfrecordBytesDecoderOpInterpCtxImpl<schema::OfrecordBytesDecoderOp>);

const HashSet<std::string>& OfrecordImageClassificationReaderOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "batch_size", "color_space", "data_dir", "data_part_num", "decode_buffer_size_per_thread", "image_feature_name", "label_feature_name", "num_decode_threads_per_machine", "part_name_prefix", "part_name_suffix_length", "random_shuffle", "seed", "shuffle_after_epoch", "shuffle_buffer_size", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> OfrecordImageClassificationReaderOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "batch_size") {
    return CastAttr(&internal_->batch_size);
  }
  if(attr_name == "color_space") {
    return CastAttr(&internal_->color_space);
  }
  if(attr_name == "data_dir") {
    return CastAttr(&internal_->data_dir);
  }
  if(attr_name == "data_part_num") {
    return CastAttr(&internal_->data_part_num);
  }
  if(attr_name == "decode_buffer_size_per_thread") {
    return CastAttr(&internal_->decode_buffer_size_per_thread);
  }
  if(attr_name == "image_feature_name") {
    return CastAttr(&internal_->image_feature_name);
  }
  if(attr_name == "label_feature_name") {
    return CastAttr(&internal_->label_feature_name);
  }
  if(attr_name == "num_decode_threads_per_machine") {
    return CastAttr(&internal_->num_decode_threads_per_machine);
  }
  if(attr_name == "part_name_prefix") {
    return CastAttr(&internal_->part_name_prefix);
  }
  if(attr_name == "part_name_suffix_length") {
    return CastAttr(&internal_->part_name_suffix_length);
  }
  if(attr_name == "random_shuffle") {
    return CastAttr(&internal_->random_shuffle);
  }
  if(attr_name == "seed") {
    return CastAttr(&internal_->seed);
  }
  if(attr_name == "shuffle_after_epoch") {
    return CastAttr(&internal_->shuffle_after_epoch);
  }
  if(attr_name == "shuffle_buffer_size") {
    return CastAttr(&internal_->shuffle_buffer_size);
  }
  return Error::RuntimeError() << "ofrecord_image_classification_reader op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.ofrecord_image_classification_reader", OfrecordImageClassificationReaderOpInterpCtxImpl<schema::OfrecordImageClassificationReaderOp>);

const HashSet<std::string>& OfrecordImageDecoderOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "color_space", "name", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> OfrecordImageDecoderOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "color_space") {
    return CastAttr(&internal_->color_space);
  }
  if(attr_name == "name") {
    return CastAttr(&internal_->name);
  }
  return Error::RuntimeError() << "ofrecord_image_decoder op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.ofrecord_image_decoder", OfrecordImageDecoderOpInterpCtxImpl<schema::OfrecordImageDecoderOp>);

const HashSet<std::string>& OfrecordImageDecoderRandomCropOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "color_space", "has_seed", "name", "num_attempts", "random_area", "random_aspect_ratio", "seed", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> OfrecordImageDecoderRandomCropOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "color_space") {
    return CastAttr(&internal_->color_space);
  }
  if(attr_name == "has_seed") {
    return CastAttr(&internal_->has_seed);
  }
  if(attr_name == "name") {
    return CastAttr(&internal_->name);
  }
  if(attr_name == "num_attempts") {
    return CastAttr(&internal_->num_attempts);
  }
  if(attr_name == "random_area") {
    return CastAttr(&internal_->random_area);
  }
  if(attr_name == "random_aspect_ratio") {
    return CastAttr(&internal_->random_aspect_ratio);
  }
  if(attr_name == "seed") {
    return CastAttr(&internal_->seed);
  }
  return Error::RuntimeError() << "ofrecord_image_decoder_random_crop op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.ofrecord_image_decoder_random_crop", OfrecordImageDecoderRandomCropOpInterpCtxImpl<schema::OfrecordImageDecoderRandomCropOp>);

const HashSet<std::string>& OfrecordRawDecoderOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_type", "dim1_varying_length", "name", "shape", "truncate", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> OfrecordRawDecoderOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_type") {
    return CastAttr(&internal_->data_type);
  }
  if(attr_name == "dim1_varying_length") {
    return CastAttr(&internal_->dim1_varying_length);
  }
  if(attr_name == "name") {
    return CastAttr(&internal_->name);
  }
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  if(attr_name == "truncate") {
    return CastAttr(&internal_->truncate);
  }
  return Error::RuntimeError() << "ofrecord_raw_decoder op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.ofrecord_raw_decoder", OfrecordRawDecoderOpInterpCtxImpl<schema::OfrecordRawDecoderOp>);

const HashSet<std::string>& OneHotOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "depth", "dtype", "floating_off_value", "floating_on_value", "integer_off_value", "integer_on_value", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> OneHotOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "depth") {
    return CastAttr(&internal_->depth);
  }
  if(attr_name == "dtype") {
    return CastAttr(&internal_->dtype);
  }
  if(attr_name == "floating_off_value") {
    return CastAttr(&internal_->floating_off_value);
  }
  if(attr_name == "floating_on_value") {
    return CastAttr(&internal_->floating_on_value);
  }
  if(attr_name == "integer_off_value") {
    return CastAttr(&internal_->integer_off_value);
  }
  if(attr_name == "integer_on_value") {
    return CastAttr(&internal_->integer_on_value);
  }
  return Error::RuntimeError() << "one_hot op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.one_hot", OneHotOpInterpCtxImpl<schema::OneHotOp>);

const HashSet<std::string>& OneRecReaderOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "batch_size", "files", "random_shuffle", "seed", "shuffle_after_epoch", "shuffle_buffer_size", "shuffle_mode", "verify_example", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> OneRecReaderOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "batch_size") {
    return CastAttr(&internal_->batch_size);
  }
  if(attr_name == "files") {
    return CastAttr(&internal_->files);
  }
  if(attr_name == "random_shuffle") {
    return CastAttr(&internal_->random_shuffle);
  }
  if(attr_name == "seed") {
    return CastAttr(&internal_->seed);
  }
  if(attr_name == "shuffle_after_epoch") {
    return CastAttr(&internal_->shuffle_after_epoch);
  }
  if(attr_name == "shuffle_buffer_size") {
    return CastAttr(&internal_->shuffle_buffer_size);
  }
  if(attr_name == "shuffle_mode") {
    return CastAttr(&internal_->shuffle_mode);
  }
  if(attr_name == "verify_example") {
    return CastAttr(&internal_->verify_example);
  }
  return Error::RuntimeError() << "OneRecReader op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.OneRecReader", OneRecReaderOpInterpCtxImpl<schema::OneRecReaderOp>);

const HashSet<std::string>& OnerecDecoderOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "batch_padding", "data_type", "has_batch_padding", "has_reshape", "is_dynamic", "key", "reshape", "static_shape", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> OnerecDecoderOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "batch_padding") {
    return CastAttr(&internal_->batch_padding);
  }
  if(attr_name == "data_type") {
    return CastAttr(&internal_->data_type);
  }
  if(attr_name == "has_batch_padding") {
    return CastAttr(&internal_->has_batch_padding);
  }
  if(attr_name == "has_reshape") {
    return CastAttr(&internal_->has_reshape);
  }
  if(attr_name == "is_dynamic") {
    return CastAttr(&internal_->is_dynamic);
  }
  if(attr_name == "key") {
    return CastAttr(&internal_->key);
  }
  if(attr_name == "reshape") {
    return CastAttr(&internal_->reshape);
  }
  if(attr_name == "static_shape") {
    return CastAttr(&internal_->static_shape);
  }
  return Error::RuntimeError() << "onerec_decoder op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.onerec_decoder", OnerecDecoderOpInterpCtxImpl<schema::OnerecDecoderOp>);

const HashSet<std::string>& OnesLikeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> OnesLikeOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "ones_like op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.ones_like", OnesLikeOpInterpCtxImpl<schema::OnesLikeOp>);

const HashSet<std::string>& PackOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "pack_num", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> PackOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "pack_num") {
    return CastAttr(&internal_->pack_num);
  }
  return Error::RuntimeError() << "pack op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.pack", PackOpInterpCtxImpl<schema::PackOp>);

const HashSet<std::string>& PadGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "floating_constant_value", "integral_constant_value", "padding_after", "padding_before", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> PadGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "floating_constant_value") {
    return CastAttr(&internal_->floating_constant_value);
  }
  if(attr_name == "integral_constant_value") {
    return CastAttr(&internal_->integral_constant_value);
  }
  if(attr_name == "padding_after") {
    return CastAttr(&internal_->padding_after);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  return Error::RuntimeError() << "pad_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.pad_grad", PadGradOpInterpCtxImpl<schema::PadGradOp>);

const HashSet<std::string>& PadOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "floating_constant_value", "integral_constant_value", "padding_after", "padding_before", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> PadOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "floating_constant_value") {
    return CastAttr(&internal_->floating_constant_value);
  }
  if(attr_name == "integral_constant_value") {
    return CastAttr(&internal_->integral_constant_value);
  }
  if(attr_name == "padding_after") {
    return CastAttr(&internal_->padding_after);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  return Error::RuntimeError() << "pad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.pad", PadOpInterpCtxImpl<schema::PadOp>);

const HashSet<std::string>& ParallelCastOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "grad_sbp_parallel", "sbp_parallel", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ParallelCastOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "grad_sbp_parallel") {
    return CastAttr(&internal_->grad_sbp_parallel);
  }
  if(attr_name == "sbp_parallel") {
    return CastAttr(&internal_->sbp_parallel);
  }
  return Error::RuntimeError() << "parallel_cast op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.parallel_cast", ParallelCastOpInterpCtxImpl<schema::ParallelCastOp>);

const HashSet<std::string>& PowOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> PowOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "pow op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.pow", PowOpInterpCtxImpl<schema::PowOp>);

const HashSet<std::string>& PowXGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> PowXGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "pow_x_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.pow_x_grad", PowXGradOpInterpCtxImpl<schema::PowXGradOp>);

const HashSet<std::string>& PowYGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> PowYGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "pow_y_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.pow_y_grad", PowYGradOpInterpCtxImpl<schema::PowYGradOp>);

const HashSet<std::string>& PreluGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> PreluGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "prelu_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.prelu_grad", PreluGradOpInterpCtxImpl<schema::PreluGradOp>);

const HashSet<std::string>& PreluOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> PreluOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "prelu op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.prelu", PreluOpInterpCtxImpl<schema::PreluOp>);

const HashSet<std::string>& QuantizationOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "quantization_bit", "quantization_formula", "quantization_scheme", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> QuantizationOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "quantization_bit") {
    return CastAttr(&internal_->quantization_bit);
  }
  if(attr_name == "quantization_formula") {
    return CastAttr(&internal_->quantization_formula);
  }
  if(attr_name == "quantization_scheme") {
    return CastAttr(&internal_->quantization_scheme);
  }
  return Error::RuntimeError() << "quantization op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.quantization", QuantizationOpInterpCtxImpl<schema::QuantizationOp>);

const HashSet<std::string>& RandomMaskLikeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "rate", "seed", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> RandomMaskLikeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "rate") {
    return CastAttr(&internal_->rate);
  }
  if(attr_name == "seed") {
    return CastAttr(&internal_->seed);
  }
  return Error::RuntimeError() << "random_mask_like op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.random_mask_like", RandomMaskLikeOpInterpCtxImpl<schema::RandomMaskLikeOp>);

const HashSet<std::string>& RandpermOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "n", "nd_sbp", "seed", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> RandpermOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "n") {
    return CastAttr(&internal_->n);
  }
  if(attr_name == "nd_sbp") {
    return CastAttr(&internal_->nd_sbp);
  }
  if(attr_name == "seed") {
    return CastAttr(&internal_->seed);
  }
  return Error::RuntimeError() << "randperm op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.randperm", RandpermOpInterpCtxImpl<schema::RandpermOp>);

const HashSet<std::string>& ReciprocalGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReciprocalGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "reciprocal_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reciprocal_grad", ReciprocalGradOpInterpCtxImpl<schema::ReciprocalGradOp>);

const HashSet<std::string>& ReciprocalNoNanGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReciprocalNoNanGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "reciprocal_no_nan_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reciprocal_no_nan_grad", ReciprocalNoNanGradOpInterpCtxImpl<schema::ReciprocalNoNanGradOp>);

const HashSet<std::string>& ReciprocalNoNanOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReciprocalNoNanOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "reciprocal_no_nan op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reciprocal_no_nan", ReciprocalNoNanOpInterpCtxImpl<schema::ReciprocalNoNanOp>);

const HashSet<std::string>& ReciprocalOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReciprocalOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "reciprocal op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reciprocal", ReciprocalOpInterpCtxImpl<schema::ReciprocalOp>);

const HashSet<std::string>& RecvOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "device_id", "device_type", "dtype", "shape", "src_process_id", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> RecvOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "device_id") {
    return CastAttr(&internal_->device_id);
  }
  if(attr_name == "device_type") {
    return CastAttr(&internal_->device_type);
  }
  if(attr_name == "dtype") {
    return CastAttr(&internal_->dtype);
  }
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  if(attr_name == "src_process_id") {
    return CastAttr(&internal_->src_process_id);
  }
  return Error::RuntimeError() << "recv op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.recv", RecvOpInterpCtxImpl<schema::RecvOp>);

const HashSet<std::string>& ReduceAllOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "keepdims", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReduceAllOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "keepdims") {
    return CastAttr(&internal_->keepdims);
  }
  return Error::RuntimeError() << "reduce_all op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reduce_all", ReduceAllOpInterpCtxImpl<schema::ReduceAllOp>);

const HashSet<std::string>& ReduceAnyOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "keepdims", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReduceAnyOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "keepdims") {
    return CastAttr(&internal_->keepdims);
  }
  return Error::RuntimeError() << "reduce_any op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reduce_any", ReduceAnyOpInterpCtxImpl<schema::ReduceAnyOp>);

const HashSet<std::string>& ReduceMaxDeviceStageGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReduceMaxDeviceStageGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  return Error::RuntimeError() << "reduce_max_device_stage_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reduce_max_device_stage_grad", ReduceMaxDeviceStageGradOpInterpCtxImpl<schema::ReduceMaxDeviceStageGradOp>);

const HashSet<std::string>& ReduceMaxDeviceStageOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReduceMaxDeviceStageOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  return Error::RuntimeError() << "reduce_max_device_stage op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reduce_max_device_stage", ReduceMaxDeviceStageOpInterpCtxImpl<schema::ReduceMaxDeviceStageOp>);

const HashSet<std::string>& ReduceMaxGlobalStageGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "keepdims", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReduceMaxGlobalStageGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "keepdims") {
    return CastAttr(&internal_->keepdims);
  }
  return Error::RuntimeError() << "reduce_max_global_stage_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reduce_max_global_stage_grad", ReduceMaxGlobalStageGradOpInterpCtxImpl<schema::ReduceMaxGlobalStageGradOp>);

const HashSet<std::string>& ReduceMaxGlobalStageOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "keepdims", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReduceMaxGlobalStageOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "keepdims") {
    return CastAttr(&internal_->keepdims);
  }
  return Error::RuntimeError() << "reduce_max_global_stage op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reduce_max_global_stage", ReduceMaxGlobalStageOpInterpCtxImpl<schema::ReduceMaxGlobalStageOp>);

const HashSet<std::string>& ReduceMaxOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "keepdims", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReduceMaxOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "keepdims") {
    return CastAttr(&internal_->keepdims);
  }
  return Error::RuntimeError() << "reduce_max op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reduce_max", ReduceMaxOpInterpCtxImpl<schema::ReduceMaxOp>);

const HashSet<std::string>& ReduceMinDeviceStageGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReduceMinDeviceStageGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  return Error::RuntimeError() << "reduce_min_device_stage_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reduce_min_device_stage_grad", ReduceMinDeviceStageGradOpInterpCtxImpl<schema::ReduceMinDeviceStageGradOp>);

const HashSet<std::string>& ReduceMinDeviceStageOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReduceMinDeviceStageOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  return Error::RuntimeError() << "reduce_min_device_stage op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reduce_min_device_stage", ReduceMinDeviceStageOpInterpCtxImpl<schema::ReduceMinDeviceStageOp>);

const HashSet<std::string>& ReduceMinGlobalStageGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "keepdims", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReduceMinGlobalStageGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "keepdims") {
    return CastAttr(&internal_->keepdims);
  }
  return Error::RuntimeError() << "reduce_min_global_stage_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reduce_min_global_stage_grad", ReduceMinGlobalStageGradOpInterpCtxImpl<schema::ReduceMinGlobalStageGradOp>);

const HashSet<std::string>& ReduceMinGlobalStageOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "keepdims", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReduceMinGlobalStageOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "keepdims") {
    return CastAttr(&internal_->keepdims);
  }
  return Error::RuntimeError() << "reduce_min_global_stage op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reduce_min_global_stage", ReduceMinGlobalStageOpInterpCtxImpl<schema::ReduceMinGlobalStageOp>);

const HashSet<std::string>& ReduceMinOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "keepdims", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReduceMinOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "keepdims") {
    return CastAttr(&internal_->keepdims);
  }
  return Error::RuntimeError() << "reduce_min op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reduce_min", ReduceMinOpInterpCtxImpl<schema::ReduceMinOp>);

const HashSet<std::string>& ReduceProdOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "keepdims", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReduceProdOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "keepdims") {
    return CastAttr(&internal_->keepdims);
  }
  return Error::RuntimeError() << "reduce_prod op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reduce_prod", ReduceProdOpInterpCtxImpl<schema::ReduceProdOp>);

const HashSet<std::string>& ReduceSumLikeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReduceSumLikeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  return Error::RuntimeError() << "reduce_sum_like op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reduce_sum_like", ReduceSumLikeOpInterpCtxImpl<schema::ReduceSumLikeOp>);

const HashSet<std::string>& ReduceSumOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "keepdims", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReduceSumOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "keepdims") {
    return CastAttr(&internal_->keepdims);
  }
  return Error::RuntimeError() << "reduce_sum op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reduce_sum", ReduceSumOpInterpCtxImpl<schema::ReduceSumOp>);

const HashSet<std::string>& ReflectionPad2DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "padding", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReflectionPad2DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  return Error::RuntimeError() << "reflection_pad2d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reflection_pad2d_grad", ReflectionPad2DGradOpInterpCtxImpl<schema::ReflectionPad2DGradOp>);

const HashSet<std::string>& ReflectionPad2DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "padding", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReflectionPad2DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  return Error::RuntimeError() << "reflection_pad2d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reflection_pad2d", ReflectionPad2DOpInterpCtxImpl<schema::ReflectionPad2DOp>);

const HashSet<std::string>& ReluGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReluGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "relu_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.relu_grad", ReluGradOpInterpCtxImpl<schema::ReluGradOp>);

const HashSet<std::string>& ReluOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReluOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "relu op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.relu", ReluOpInterpCtxImpl<schema::ReluOp>);

const HashSet<std::string>& RepeatOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "repeat_num", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> RepeatOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "repeat_num") {
    return CastAttr(&internal_->repeat_num);
  }
  return Error::RuntimeError() << "repeat op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.repeat", RepeatOpInterpCtxImpl<schema::RepeatOp>);

const HashSet<std::string>& ReplicationPad2DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "padding", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReplicationPad2DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  return Error::RuntimeError() << "replication_pad2d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.replication_pad2d_grad", ReplicationPad2DGradOpInterpCtxImpl<schema::ReplicationPad2DGradOp>);

const HashSet<std::string>& ReplicationPad2DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "padding", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReplicationPad2DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  return Error::RuntimeError() << "replication_pad2d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.replication_pad2d", ReplicationPad2DOpInterpCtxImpl<schema::ReplicationPad2DOp>);

const HashSet<std::string>& ReshapeLikeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReshapeLikeOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "reshape_like op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reshape_like", ReshapeLikeOpInterpCtxImpl<schema::ReshapeLikeOp>);

const HashSet<std::string>& ReshapeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "shape", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ReshapeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  return Error::RuntimeError() << "reshape op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.reshape", ReshapeOpInterpCtxImpl<schema::ReshapeOp>);

const HashSet<std::string>& RintGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> RintGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "rint_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.rint_grad", RintGradOpInterpCtxImpl<schema::RintGradOp>);

const HashSet<std::string>& RintOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> RintOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "rint op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.rint", RintOpInterpCtxImpl<schema::RintOp>);

const HashSet<std::string>& RmspropUpdateOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "centered", "decay_rate", "epsilon", "l1", "l2", "learning_rate_val", "scale", "weight_decay", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> RmspropUpdateOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "centered") {
    return CastAttr(&internal_->centered);
  }
  if(attr_name == "decay_rate") {
    return CastAttr(&internal_->decay_rate);
  }
  if(attr_name == "epsilon") {
    return CastAttr(&internal_->epsilon);
  }
  if(attr_name == "l1") {
    return CastAttr(&internal_->l1);
  }
  if(attr_name == "l2") {
    return CastAttr(&internal_->l2);
  }
  if(attr_name == "learning_rate_val") {
    return CastAttr(&internal_->learning_rate_val);
  }
  if(attr_name == "scale") {
    return CastAttr(&internal_->scale);
  }
  if(attr_name == "weight_decay") {
    return CastAttr(&internal_->weight_decay);
  }
  return Error::RuntimeError() << "rmsprop_update op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.rmsprop_update", RmspropUpdateOpInterpCtxImpl<schema::RmspropUpdateOp>);

const HashSet<std::string>& RoiAlignGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "aligned", "pooled_h", "pooled_w", "sampling_ratio", "spatial_scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> RoiAlignGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "aligned") {
    return CastAttr(&internal_->aligned);
  }
  if(attr_name == "pooled_h") {
    return CastAttr(&internal_->pooled_h);
  }
  if(attr_name == "pooled_w") {
    return CastAttr(&internal_->pooled_w);
  }
  if(attr_name == "sampling_ratio") {
    return CastAttr(&internal_->sampling_ratio);
  }
  if(attr_name == "spatial_scale") {
    return CastAttr(&internal_->spatial_scale);
  }
  return Error::RuntimeError() << "roi_align_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.roi_align_grad", RoiAlignGradOpInterpCtxImpl<schema::RoiAlignGradOp>);

const HashSet<std::string>& RoiAlignOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "aligned", "pooled_h", "pooled_w", "sampling_ratio", "spatial_scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> RoiAlignOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "aligned") {
    return CastAttr(&internal_->aligned);
  }
  if(attr_name == "pooled_h") {
    return CastAttr(&internal_->pooled_h);
  }
  if(attr_name == "pooled_w") {
    return CastAttr(&internal_->pooled_w);
  }
  if(attr_name == "sampling_ratio") {
    return CastAttr(&internal_->sampling_ratio);
  }
  if(attr_name == "spatial_scale") {
    return CastAttr(&internal_->spatial_scale);
  }
  return Error::RuntimeError() << "roi_align op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.roi_align", RoiAlignOpInterpCtxImpl<schema::RoiAlignOp>);

const HashSet<std::string>& RollOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dims", "shifts", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> RollOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dims") {
    return CastAttr(&internal_->dims);
  }
  if(attr_name == "shifts") {
    return CastAttr(&internal_->shifts);
  }
  return Error::RuntimeError() << "roll op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.roll", RollOpInterpCtxImpl<schema::RollOp>);

const HashSet<std::string>& RoundGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> RoundGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "round_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.round_grad", RoundGradOpInterpCtxImpl<schema::RoundGradOp>);

const HashSet<std::string>& RoundOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> RoundOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "round op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.round", RoundOpInterpCtxImpl<schema::RoundOp>);

const HashSet<std::string>& RsqrtGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> RsqrtGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "rsqrt_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.rsqrt_grad", RsqrtGradOpInterpCtxImpl<schema::RsqrtGradOp>);

const HashSet<std::string>& RsqrtOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> RsqrtOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "rsqrt op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.rsqrt", RsqrtOpInterpCtxImpl<schema::RsqrtOp>);

const HashSet<std::string>& SamePaddingGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "dilation_rate", "kernel_size", "padding", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SamePaddingGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "dilation_rate") {
    return CastAttr(&internal_->dilation_rate);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "same_padding_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.same_padding_grad", SamePaddingGradOpInterpCtxImpl<schema::SamePaddingGradOp>);

const HashSet<std::string>& SamePaddingOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "dilation_rate", "kernel_size", "padding", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SamePaddingOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "dilation_rate") {
    return CastAttr(&internal_->dilation_rate);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "same_padding op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.same_padding", SamePaddingOpInterpCtxImpl<schema::SamePaddingOp>);

const HashSet<std::string>& ScalarAddByTensorOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarAddByTensorOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "scalar_add_by_tensor op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_add_by_tensor", ScalarAddByTensorOpInterpCtxImpl<schema::ScalarAddByTensorOp>);

const HashSet<std::string>& ScalarAddOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarAddOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "scalar_add op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_add", ScalarAddOpInterpCtxImpl<schema::ScalarAddOp>);

const HashSet<std::string>& ScalarDivByTensorOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarDivByTensorOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "scalar_div_by_tensor op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_div_by_tensor", ScalarDivByTensorOpInterpCtxImpl<schema::ScalarDivByTensorOp>);

const HashSet<std::string>& ScalarFloordivOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarFloordivOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "scalar_floordiv op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_floordiv", ScalarFloordivOpInterpCtxImpl<schema::ScalarFloordivOp>);

const HashSet<std::string>& ScalarFmodOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarFmodOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "scalar_fmod op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_fmod", ScalarFmodOpInterpCtxImpl<schema::ScalarFmodOp>);

const HashSet<std::string>& ScalarLogicalAndOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarLogicalAndOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "scalar_logical_and op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_logical_and", ScalarLogicalAndOpInterpCtxImpl<schema::ScalarLogicalAndOp>);

const HashSet<std::string>& ScalarLogicalEqualOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarLogicalEqualOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "scalar_logical_equal op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_logical_equal", ScalarLogicalEqualOpInterpCtxImpl<schema::ScalarLogicalEqualOp>);

const HashSet<std::string>& ScalarLogicalGreaterEqualOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarLogicalGreaterEqualOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "scalar_logical_greater_equal op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_logical_greater_equal", ScalarLogicalGreaterEqualOpInterpCtxImpl<schema::ScalarLogicalGreaterEqualOp>);

const HashSet<std::string>& ScalarLogicalGreaterOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarLogicalGreaterOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "scalar_logical_greater op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_logical_greater", ScalarLogicalGreaterOpInterpCtxImpl<schema::ScalarLogicalGreaterOp>);

const HashSet<std::string>& ScalarLogicalLessEqualOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarLogicalLessEqualOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "scalar_logical_less_equal op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_logical_less_equal", ScalarLogicalLessEqualOpInterpCtxImpl<schema::ScalarLogicalLessEqualOp>);

const HashSet<std::string>& ScalarLogicalLessOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarLogicalLessOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "scalar_logical_less op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_logical_less", ScalarLogicalLessOpInterpCtxImpl<schema::ScalarLogicalLessOp>);

const HashSet<std::string>& ScalarLogicalNotEqualOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarLogicalNotEqualOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "scalar_logical_not_equal op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_logical_not_equal", ScalarLogicalNotEqualOpInterpCtxImpl<schema::ScalarLogicalNotEqualOp>);

const HashSet<std::string>& ScalarLogicalOrOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarLogicalOrOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "scalar_logical_or op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_logical_or", ScalarLogicalOrOpInterpCtxImpl<schema::ScalarLogicalOrOp>);

const HashSet<std::string>& ScalarLogicalXorOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarLogicalXorOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "scalar_logical_xor op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_logical_xor", ScalarLogicalXorOpInterpCtxImpl<schema::ScalarLogicalXorOp>);

const HashSet<std::string>& ScalarMulByTensorOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarMulByTensorOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "scalar_mul_by_tensor op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_mul_by_tensor", ScalarMulByTensorOpInterpCtxImpl<schema::ScalarMulByTensorOp>);

const HashSet<std::string>& ScalarMulOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarMulOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "scalar_mul op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_mul", ScalarMulOpInterpCtxImpl<schema::ScalarMulOp>);

const HashSet<std::string>& ScalarPowGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarPowGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "scalar_pow_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_pow_grad", ScalarPowGradOpInterpCtxImpl<schema::ScalarPowGradOp>);

const HashSet<std::string>& ScalarPowOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarPowOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "scalar_pow op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_pow", ScalarPowOpInterpCtxImpl<schema::ScalarPowOp>);

const HashSet<std::string>& ScalarSubByTensorOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScalarSubByTensorOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "scalar_sub_by_tensor op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scalar_sub_by_tensor", ScalarSubByTensorOpInterpCtxImpl<schema::ScalarSubByTensorOp>);

const HashSet<std::string>& ScatterNdLikeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScatterNdLikeOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "scatter_nd_like op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scatter_nd_like", ScatterNdLikeOpInterpCtxImpl<schema::ScatterNdLikeOp>);

const HashSet<std::string>& ScatterNdOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "shape", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ScatterNdOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  return Error::RuntimeError() << "scatter_nd op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.scatter_nd", ScatterNdOpInterpCtxImpl<schema::ScatterNdOp>);

const HashSet<std::string>& SeluGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SeluGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "selu_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.selu_grad", SeluGradOpInterpCtxImpl<schema::SeluGradOp>);

const HashSet<std::string>& SeluOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SeluOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "selu op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.selu", SeluOpInterpCtxImpl<schema::SeluOp>);

const HashSet<std::string>& SendOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dst_process_id", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SendOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dst_process_id") {
    return CastAttr(&internal_->dst_process_id);
  }
  return Error::RuntimeError() << "send op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.send", SendOpInterpCtxImpl<schema::SendOp>);

const HashSet<std::string>& SgdUpdateOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "l1", "l2", "learning_rate_val", "scale", "weight_decay", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SgdUpdateOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "l1") {
    return CastAttr(&internal_->l1);
  }
  if(attr_name == "l2") {
    return CastAttr(&internal_->l2);
  }
  if(attr_name == "learning_rate_val") {
    return CastAttr(&internal_->learning_rate_val);
  }
  if(attr_name == "scale") {
    return CastAttr(&internal_->scale);
  }
  if(attr_name == "weight_decay") {
    return CastAttr(&internal_->weight_decay);
  }
  return Error::RuntimeError() << "sgd_update op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sgd_update", SgdUpdateOpInterpCtxImpl<schema::SgdUpdateOp>);

const HashSet<std::string>& SigmoidCrossEntropyGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SigmoidCrossEntropyGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "sigmoid_cross_entropy_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sigmoid_cross_entropy_grad", SigmoidCrossEntropyGradOpInterpCtxImpl<schema::SigmoidCrossEntropyGradOp>);

const HashSet<std::string>& SigmoidCrossEntropyOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SigmoidCrossEntropyOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "sigmoid_cross_entropy op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sigmoid_cross_entropy", SigmoidCrossEntropyOpInterpCtxImpl<schema::SigmoidCrossEntropyOp>);

const HashSet<std::string>& SigmoidGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SigmoidGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "sigmoid_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sigmoid_grad", SigmoidGradOpInterpCtxImpl<schema::SigmoidGradOp>);

const HashSet<std::string>& SigmoidOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SigmoidOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "sigmoid op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sigmoid", SigmoidOpInterpCtxImpl<schema::SigmoidOp>);

const HashSet<std::string>& SigmoidV2GradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SigmoidV2GradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "sigmoid_v2_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sigmoid_v2_grad", SigmoidV2GradOpInterpCtxImpl<schema::SigmoidV2GradOp>);

const HashSet<std::string>& SigmoidV2OpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SigmoidV2Op::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "sigmoid_v2 op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sigmoid_v2", SigmoidV2OpInterpCtxImpl<schema::SigmoidV2Op>);

const HashSet<std::string>& SignGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SignGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "sign_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sign_grad", SignGradOpInterpCtxImpl<schema::SignGradOp>);

const HashSet<std::string>& SignOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SignOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "sign op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sign", SignOpInterpCtxImpl<schema::SignOp>);

const HashSet<std::string>& SiluGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SiluGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "silu_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.silu_grad", SiluGradOpInterpCtxImpl<schema::SiluGradOp>);

const HashSet<std::string>& SiluOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SiluOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "silu op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.silu", SiluOpInterpCtxImpl<schema::SiluOp>);

const HashSet<std::string>& SinGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SinGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "sin_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sin_grad", SinGradOpInterpCtxImpl<schema::SinGradOp>);

const HashSet<std::string>& SinOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SinOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "sin op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sin", SinOpInterpCtxImpl<schema::SinOp>);

const HashSet<std::string>& SinhGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SinhGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "sinh_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sinh_grad", SinhGradOpInterpCtxImpl<schema::SinhGradOp>);

const HashSet<std::string>& SinhOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SinhOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "sinh op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sinh", SinhOpInterpCtxImpl<schema::SinhOp>);

const HashSet<std::string>& SliceGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "start", "step", "stop", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SliceGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "start") {
    return CastAttr(&internal_->start);
  }
  if(attr_name == "step") {
    return CastAttr(&internal_->step);
  }
  if(attr_name == "stop") {
    return CastAttr(&internal_->stop);
  }
  return Error::RuntimeError() << "slice_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.slice_grad", SliceGradOpInterpCtxImpl<schema::SliceGradOp>);

const HashSet<std::string>& SliceOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "start", "step", "stop", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SliceOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "start") {
    return CastAttr(&internal_->start);
  }
  if(attr_name == "step") {
    return CastAttr(&internal_->step);
  }
  if(attr_name == "stop") {
    return CastAttr(&internal_->stop);
  }
  return Error::RuntimeError() << "slice op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.slice", SliceOpInterpCtxImpl<schema::SliceOp>);

const HashSet<std::string>& SliceUpdateOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "start", "step", "stop", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SliceUpdateOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "start") {
    return CastAttr(&internal_->start);
  }
  if(attr_name == "step") {
    return CastAttr(&internal_->step);
  }
  if(attr_name == "stop") {
    return CastAttr(&internal_->stop);
  }
  return Error::RuntimeError() << "slice_update op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.slice_update", SliceUpdateOpInterpCtxImpl<schema::SliceUpdateOp>);

const HashSet<std::string>& SmoothL1LossGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "beta", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SmoothL1LossGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "beta") {
    return CastAttr(&internal_->beta);
  }
  return Error::RuntimeError() << "smooth_l1_loss_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.smooth_l1_loss_grad", SmoothL1LossGradOpInterpCtxImpl<schema::SmoothL1LossGradOp>);

const HashSet<std::string>& SmoothL1LossOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "beta", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SmoothL1LossOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "beta") {
    return CastAttr(&internal_->beta);
  }
  return Error::RuntimeError() << "smooth_l1_loss op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.smooth_l1_loss", SmoothL1LossOpInterpCtxImpl<schema::SmoothL1LossOp>);

const HashSet<std::string>& SoftmaxCrossEntropyGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SoftmaxCrossEntropyGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "softmax_cross_entropy_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.softmax_cross_entropy_grad", SoftmaxCrossEntropyGradOpInterpCtxImpl<schema::SoftmaxCrossEntropyGradOp>);

const HashSet<std::string>& SoftmaxCrossEntropyOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SoftmaxCrossEntropyOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "softmax_cross_entropy op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.softmax_cross_entropy", SoftmaxCrossEntropyOpInterpCtxImpl<schema::SoftmaxCrossEntropyOp>);

const HashSet<std::string>& SoftmaxGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SoftmaxGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "softmax_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.softmax_grad", SoftmaxGradOpInterpCtxImpl<schema::SoftmaxGradOp>);

const HashSet<std::string>& SoftmaxOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SoftmaxOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "softmax op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.softmax", SoftmaxOpInterpCtxImpl<schema::SoftmaxOp>);

const HashSet<std::string>& SoftplusGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SoftplusGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "softplus_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.softplus_grad", SoftplusGradOpInterpCtxImpl<schema::SoftplusGradOp>);

const HashSet<std::string>& SoftplusOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SoftplusOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "softplus op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.softplus", SoftplusOpInterpCtxImpl<schema::SoftplusOp>);

const HashSet<std::string>& SoftsignGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SoftsignGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "softsign_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.softsign_grad", SoftsignGradOpInterpCtxImpl<schema::SoftsignGradOp>);

const HashSet<std::string>& SoftsignOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SoftsignOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "softsign op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.softsign", SoftsignOpInterpCtxImpl<schema::SoftsignOp>);

const HashSet<std::string>& SortOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "direction", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SortOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "direction") {
    return CastAttr(&internal_->direction);
  }
  return Error::RuntimeError() << "sort op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sort", SortOpInterpCtxImpl<schema::SortOp>);

const HashSet<std::string>& SparseCrossEntropyGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "depth", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SparseCrossEntropyGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "depth") {
    return CastAttr(&internal_->depth);
  }
  return Error::RuntimeError() << "sparse_cross_entropy_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sparse_cross_entropy_grad", SparseCrossEntropyGradOpInterpCtxImpl<schema::SparseCrossEntropyGradOp>);

const HashSet<std::string>& SparseCrossEntropyMsGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "depth", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SparseCrossEntropyMsGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "depth") {
    return CastAttr(&internal_->depth);
  }
  return Error::RuntimeError() << "sparse_cross_entropy_ms_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sparse_cross_entropy_ms_grad", SparseCrossEntropyMsGradOpInterpCtxImpl<schema::SparseCrossEntropyMsGradOp>);

const HashSet<std::string>& SparseCrossEntropyMsOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "depth", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SparseCrossEntropyMsOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "depth") {
    return CastAttr(&internal_->depth);
  }
  return Error::RuntimeError() << "sparse_cross_entropy_ms op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sparse_cross_entropy_ms", SparseCrossEntropyMsOpInterpCtxImpl<schema::SparseCrossEntropyMsOp>);

const HashSet<std::string>& SparseCrossEntropyOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "depth", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SparseCrossEntropyOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "depth") {
    return CastAttr(&internal_->depth);
  }
  return Error::RuntimeError() << "sparse_cross_entropy op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sparse_cross_entropy", SparseCrossEntropyOpInterpCtxImpl<schema::SparseCrossEntropyOp>);

const HashSet<std::string>& SparseSoftmaxCrossEntropyGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "depth", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SparseSoftmaxCrossEntropyGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "depth") {
    return CastAttr(&internal_->depth);
  }
  return Error::RuntimeError() << "sparse_softmax_cross_entropy_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sparse_softmax_cross_entropy_grad", SparseSoftmaxCrossEntropyGradOpInterpCtxImpl<schema::SparseSoftmaxCrossEntropyGradOp>);

const HashSet<std::string>& SparseSoftmaxCrossEntropyMsGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "depth", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SparseSoftmaxCrossEntropyMsGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "depth") {
    return CastAttr(&internal_->depth);
  }
  return Error::RuntimeError() << "sparse_softmax_cross_entropy_ms_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sparse_softmax_cross_entropy_ms_grad", SparseSoftmaxCrossEntropyMsGradOpInterpCtxImpl<schema::SparseSoftmaxCrossEntropyMsGradOp>);

const HashSet<std::string>& SparseSoftmaxCrossEntropyMsOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "depth", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SparseSoftmaxCrossEntropyMsOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "depth") {
    return CastAttr(&internal_->depth);
  }
  return Error::RuntimeError() << "sparse_softmax_cross_entropy_ms op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sparse_softmax_cross_entropy_ms", SparseSoftmaxCrossEntropyMsOpInterpCtxImpl<schema::SparseSoftmaxCrossEntropyMsOp>);

const HashSet<std::string>& SparseSoftmaxCrossEntropyOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "depth", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SparseSoftmaxCrossEntropyOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "depth") {
    return CastAttr(&internal_->depth);
  }
  return Error::RuntimeError() << "sparse_softmax_cross_entropy op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sparse_softmax_cross_entropy", SparseSoftmaxCrossEntropyOpInterpCtxImpl<schema::SparseSoftmaxCrossEntropyOp>);

const HashSet<std::string>& SplitLikeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SplitLikeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  return Error::RuntimeError() << "split_like op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.split_like", SplitLikeOpInterpCtxImpl<schema::SplitLikeOp>);

const HashSet<std::string>& SqrtGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SqrtGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "sqrt_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sqrt_grad", SqrtGradOpInterpCtxImpl<schema::SqrtGradOp>);

const HashSet<std::string>& SqrtOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SqrtOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "sqrt op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.sqrt", SqrtOpInterpCtxImpl<schema::SqrtOp>);

const HashSet<std::string>& SquareGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SquareGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "square_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.square_grad", SquareGradOpInterpCtxImpl<schema::SquareGradOp>);

const HashSet<std::string>& SquareOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SquareOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "square op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.square", SquareOpInterpCtxImpl<schema::SquareOp>);

const HashSet<std::string>& SquareSumOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SquareSumOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "square_sum op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.square_sum", SquareSumOpInterpCtxImpl<schema::SquareSumOp>);

const HashSet<std::string>& SqueezeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axes", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SqueezeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axes") {
    return CastAttr(&internal_->axes);
  }
  return Error::RuntimeError() << "squeeze op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.squeeze", SqueezeOpInterpCtxImpl<schema::SqueezeOp>);

const HashSet<std::string>& SspVariableProxyOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "buffer_size", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SspVariableProxyOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "buffer_size") {
    return CastAttr(&internal_->buffer_size);
  }
  return Error::RuntimeError() << "ssp_variable_proxy op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.ssp_variable_proxy", SspVariableProxyOpInterpCtxImpl<schema::SspVariableProxyOp>);

const HashSet<std::string>& SummaryWriteHistogramOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SummaryWriteHistogramOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "summary_write_histogram op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.summary_write_histogram", SummaryWriteHistogramOpInterpCtxImpl<schema::SummaryWriteHistogramOp>);

const HashSet<std::string>& SummaryWriteImageOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SummaryWriteImageOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "summary_write_image op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.summary_write_image", SummaryWriteImageOpInterpCtxImpl<schema::SummaryWriteImageOp>);

const HashSet<std::string>& SummaryWritePbOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SummaryWritePbOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "summary_write_pb op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.summary_write_pb", SummaryWritePbOpInterpCtxImpl<schema::SummaryWritePbOp>);

const HashSet<std::string>& SummaryWriteScalarOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> SummaryWriteScalarOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "summary_write_scalar op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.summary_write_scalar", SummaryWriteScalarOpInterpCtxImpl<schema::SummaryWriteScalarOp>);

const HashSet<std::string>& TanGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TanGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "tan_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tan_grad", TanGradOpInterpCtxImpl<schema::TanGradOp>);

const HashSet<std::string>& TanOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TanOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "tan op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tan", TanOpInterpCtxImpl<schema::TanOp>);

const HashSet<std::string>& TanhGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TanhGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "tanh_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tanh_grad", TanhGradOpInterpCtxImpl<schema::TanhGradOp>);

const HashSet<std::string>& TanhOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TanhOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "tanh op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tanh", TanhOpInterpCtxImpl<schema::TanhOp>);

const HashSet<std::string>& TensorBufferToListOfTensorsOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dynamic_out", "out_dtype", "out_shape", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TensorBufferToListOfTensorsOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dynamic_out") {
    return CastAttr(&internal_->dynamic_out);
  }
  if(attr_name == "out_dtype") {
    return CastAttr(&internal_->out_dtype);
  }
  if(attr_name == "out_shape") {
    return CastAttr(&internal_->out_shape);
  }
  return Error::RuntimeError() << "tensor_buffer_to_list_of_tensors op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tensor_buffer_to_list_of_tensors", TensorBufferToListOfTensorsOpInterpCtxImpl<schema::TensorBufferToListOfTensorsOp>);

const HashSet<std::string>& TensorBufferToListOfTensorsV2OpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dynamic_out", "out_dtypes", "out_shapes", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TensorBufferToListOfTensorsV2Op::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dynamic_out") {
    return CastAttr(&internal_->dynamic_out);
  }
  if(attr_name == "out_dtypes") {
    return CastAttr(&internal_->out_dtypes);
  }
  if(attr_name == "out_shapes") {
    return CastAttr(&internal_->out_shapes);
  }
  return Error::RuntimeError() << "tensor_buffer_to_list_of_tensors_v2 op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tensor_buffer_to_list_of_tensors_v2", TensorBufferToListOfTensorsV2OpInterpCtxImpl<schema::TensorBufferToListOfTensorsV2Op>);

const HashSet<std::string>& TensorBufferToTensorOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dtype", "instance_shape", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TensorBufferToTensorOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dtype") {
    return CastAttr(&internal_->dtype);
  }
  if(attr_name == "instance_shape") {
    return CastAttr(&internal_->instance_shape);
  }
  return Error::RuntimeError() << "tensor_buffer_to_tensor op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tensor_buffer_to_tensor", TensorBufferToTensorOpInterpCtxImpl<schema::TensorBufferToTensorOp>);

const HashSet<std::string>& TensorScatterNdAddOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TensorScatterNdAddOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "tensor_scatter_nd_add op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tensor_scatter_nd_add", TensorScatterNdAddOpInterpCtxImpl<schema::TensorScatterNdAddOp>);

const HashSet<std::string>& TensorScatterNdUpdateOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TensorScatterNdUpdateOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "tensor_scatter_nd_update op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tensor_scatter_nd_update", TensorScatterNdUpdateOpInterpCtxImpl<schema::TensorScatterNdUpdateOp>);

const HashSet<std::string>& TensorToTensorBufferOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "instance_dims", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TensorToTensorBufferOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "instance_dims") {
    return CastAttr(&internal_->instance_dims);
  }
  return Error::RuntimeError() << "tensor_to_tensor_buffer op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tensor_to_tensor_buffer", TensorToTensorBufferOpInterpCtxImpl<schema::TensorToTensorBufferOp>);

const HashSet<std::string>& TestDataTypeAttrOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "output_type", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TestDataTypeAttrOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "output_type") {
    return CastAttr(&internal_->output_type);
  }
  return Error::RuntimeError() << "TestDataTypeAttr op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.TestDataTypeAttr", TestDataTypeAttrOpInterpCtxImpl<schema::TestDataTypeAttrOp>);

const HashSet<std::string>& TestDynamicSourceOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TestDynamicSourceOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "TestDynamicSource op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.TestDynamicSource", TestDynamicSourceOpInterpCtxImpl<schema::TestDynamicSourceOp>);

const HashSet<std::string>& TestListDataTypeAndListShapeAndListStringAttrOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "out_shapes", "out_types", "string_list", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TestListDataTypeAndListShapeAndListStringAttrOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "out_shapes") {
    return CastAttr(&internal_->out_shapes);
  }
  if(attr_name == "out_types") {
    return CastAttr(&internal_->out_types);
  }
  if(attr_name == "string_list") {
    return CastAttr(&internal_->string_list);
  }
  return Error::RuntimeError() << "TestListDataTypeAndListShapeAndListStringAttr op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.TestListDataTypeAndListShapeAndListStringAttr", TestListDataTypeAndListShapeAndListStringAttrOpInterpCtxImpl<schema::TestListDataTypeAndListShapeAndListStringAttrOp>);

const HashSet<std::string>& TestMultiInputGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TestMultiInputGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "TestMultiInputGrad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.TestMultiInputGrad", TestMultiInputGradOpInterpCtxImpl<schema::TestMultiInputGradOp>);

const HashSet<std::string>& TestMultiInputOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TestMultiInputOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "TestMultiInput op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.TestMultiInput", TestMultiInputOpInterpCtxImpl<schema::TestMultiInputOp>);

const HashSet<std::string>& TestMultiOutputOrderOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TestMultiOutputOrderOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "TestMultiOutputOrder op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.TestMultiOutputOrder", TestMultiOutputOrderOpInterpCtxImpl<schema::TestMultiOutputOrderOp>);

const HashSet<std::string>& TestRandomSourceOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "seed", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TestRandomSourceOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "seed") {
    return CastAttr(&internal_->seed);
  }
  return Error::RuntimeError() << "TestRandomSource op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.TestRandomSource", TestRandomSourceOpInterpCtxImpl<schema::TestRandomSourceOp>);

const HashSet<std::string>& TestReshapeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "shape", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TestReshapeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  return Error::RuntimeError() << "TestReshape op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.TestReshape", TestReshapeOpInterpCtxImpl<schema::TestReshapeOp>);

const HashSet<std::string>& TestSourceMultiGpuFixedOutNumOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "out_num", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TestSourceMultiGpuFixedOutNumOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "out_num") {
    return CastAttr(&internal_->out_num);
  }
  return Error::RuntimeError() << "TestSourceMultiGpuFixedOutNum op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.TestSourceMultiGpuFixedOutNum", TestSourceMultiGpuFixedOutNumOpInterpCtxImpl<schema::TestSourceMultiGpuFixedOutNumOp>);

const HashSet<std::string>& TestSourceOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TestSourceOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "TestSource op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.TestSource", TestSourceOpInterpCtxImpl<schema::TestSourceOp>);

const HashSet<std::string>& TestUserOpAttrAutoTypeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "int1", "int2", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TestUserOpAttrAutoTypeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "int1") {
    return CastAttr(&internal_->int1);
  }
  if(attr_name == "int2") {
    return CastAttr(&internal_->int2);
  }
  return Error::RuntimeError() << "test_user_op_attr_auto_type op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.test_user_op_attr_auto_type", TestUserOpAttrAutoTypeOpInterpCtxImpl<schema::TestUserOpAttrAutoTypeOp>);

const HashSet<std::string>& TfAvgPool1DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "data_format", "padding", "padding_after", "padding_before", "pool_size", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TfAvgPool1DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "padding_after") {
    return CastAttr(&internal_->padding_after);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "pool_size") {
    return CastAttr(&internal_->pool_size);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "tf_avg_pool_1d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tf_avg_pool_1d_grad", TfAvgPool1DGradOpInterpCtxImpl<schema::TfAvgPool1DGradOp>);

const HashSet<std::string>& TfAvgPool1DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "data_format", "padding", "padding_after", "padding_before", "pool_size", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TfAvgPool1DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "padding_after") {
    return CastAttr(&internal_->padding_after);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "pool_size") {
    return CastAttr(&internal_->pool_size);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "tf_avg_pool_1d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tf_avg_pool_1d", TfAvgPool1DOpInterpCtxImpl<schema::TfAvgPool1DOp>);

const HashSet<std::string>& TfAvgPool2DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "data_format", "padding", "padding_after", "padding_before", "pool_size", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TfAvgPool2DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "padding_after") {
    return CastAttr(&internal_->padding_after);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "pool_size") {
    return CastAttr(&internal_->pool_size);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "tf_avg_pool_2d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tf_avg_pool_2d_grad", TfAvgPool2DGradOpInterpCtxImpl<schema::TfAvgPool2DGradOp>);

const HashSet<std::string>& TfAvgPool2DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "data_format", "padding", "padding_after", "padding_before", "pool_size", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TfAvgPool2DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "padding_after") {
    return CastAttr(&internal_->padding_after);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "pool_size") {
    return CastAttr(&internal_->pool_size);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "tf_avg_pool_2d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tf_avg_pool_2d", TfAvgPool2DOpInterpCtxImpl<schema::TfAvgPool2DOp>);

const HashSet<std::string>& TfAvgPool3DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "data_format", "padding", "padding_after", "padding_before", "pool_size", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TfAvgPool3DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "padding_after") {
    return CastAttr(&internal_->padding_after);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "pool_size") {
    return CastAttr(&internal_->pool_size);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "tf_avg_pool_3d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tf_avg_pool_3d_grad", TfAvgPool3DGradOpInterpCtxImpl<schema::TfAvgPool3DGradOp>);

const HashSet<std::string>& TfAvgPool3DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "data_format", "padding", "padding_after", "padding_before", "pool_size", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TfAvgPool3DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "padding_after") {
    return CastAttr(&internal_->padding_after);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "pool_size") {
    return CastAttr(&internal_->pool_size);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "tf_avg_pool_3d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tf_avg_pool_3d", TfAvgPool3DOpInterpCtxImpl<schema::TfAvgPool3DOp>);

const HashSet<std::string>& TfMaxPool1DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "data_format", "padding", "padding_after", "padding_before", "pool_size", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TfMaxPool1DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "padding_after") {
    return CastAttr(&internal_->padding_after);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "pool_size") {
    return CastAttr(&internal_->pool_size);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "tf_max_pool_1d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tf_max_pool_1d_grad", TfMaxPool1DGradOpInterpCtxImpl<schema::TfMaxPool1DGradOp>);

const HashSet<std::string>& TfMaxPool1DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "data_format", "padding", "padding_after", "padding_before", "pool_size", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TfMaxPool1DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "padding_after") {
    return CastAttr(&internal_->padding_after);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "pool_size") {
    return CastAttr(&internal_->pool_size);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "tf_max_pool_1d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tf_max_pool_1d", TfMaxPool1DOpInterpCtxImpl<schema::TfMaxPool1DOp>);

const HashSet<std::string>& TfMaxPool2DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "data_format", "padding", "padding_after", "padding_before", "pool_size", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TfMaxPool2DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "padding_after") {
    return CastAttr(&internal_->padding_after);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "pool_size") {
    return CastAttr(&internal_->pool_size);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "tf_max_pool_2d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tf_max_pool_2d_grad", TfMaxPool2DGradOpInterpCtxImpl<schema::TfMaxPool2DGradOp>);

const HashSet<std::string>& TfMaxPool2DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "data_format", "padding", "padding_after", "padding_before", "pool_size", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TfMaxPool2DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "padding_after") {
    return CastAttr(&internal_->padding_after);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "pool_size") {
    return CastAttr(&internal_->pool_size);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "tf_max_pool_2d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tf_max_pool_2d", TfMaxPool2DOpInterpCtxImpl<schema::TfMaxPool2DOp>);

const HashSet<std::string>& TfMaxPool3DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "data_format", "padding", "padding_after", "padding_before", "pool_size", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TfMaxPool3DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "padding_after") {
    return CastAttr(&internal_->padding_after);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "pool_size") {
    return CastAttr(&internal_->pool_size);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "tf_max_pool_3d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tf_max_pool_3d_grad", TfMaxPool3DGradOpInterpCtxImpl<schema::TfMaxPool3DGradOp>);

const HashSet<std::string>& TfMaxPool3DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "ceil_mode", "data_format", "padding", "padding_after", "padding_before", "pool_size", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TfMaxPool3DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "ceil_mode") {
    return CastAttr(&internal_->ceil_mode);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "padding_after") {
    return CastAttr(&internal_->padding_after);
  }
  if(attr_name == "padding_before") {
    return CastAttr(&internal_->padding_before);
  }
  if(attr_name == "pool_size") {
    return CastAttr(&internal_->pool_size);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "tf_max_pool_3d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tf_max_pool_3d", TfMaxPool3DOpInterpCtxImpl<schema::TfMaxPool3DOp>);

const HashSet<std::string>& TfPreluGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TfPreluGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "tf_prelu_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tf_prelu_grad", TfPreluGradOpInterpCtxImpl<schema::TfPreluGradOp>);

const HashSet<std::string>& TfPreluOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TfPreluOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "tf_prelu op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tf_prelu", TfPreluOpInterpCtxImpl<schema::TfPreluOp>);

const HashSet<std::string>& TopKOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "k", "sorted", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TopKOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "k") {
    return CastAttr(&internal_->k);
  }
  if(attr_name == "sorted") {
    return CastAttr(&internal_->sorted);
  }
  return Error::RuntimeError() << "top_k op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.top_k", TopKOpInterpCtxImpl<schema::TopKOp>);

const HashSet<std::string>& TransposeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "perm", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TransposeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "perm") {
    return CastAttr(&internal_->perm);
  }
  return Error::RuntimeError() << "transpose op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.transpose", TransposeOpInterpCtxImpl<schema::TransposeOp>);

const HashSet<std::string>& TrilOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "diagonal", "floating_fill_value", "integer_fill_value", "is_floating_fill_value", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TrilOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "diagonal") {
    return CastAttr(&internal_->diagonal);
  }
  if(attr_name == "floating_fill_value") {
    return CastAttr(&internal_->floating_fill_value);
  }
  if(attr_name == "integer_fill_value") {
    return CastAttr(&internal_->integer_fill_value);
  }
  if(attr_name == "is_floating_fill_value") {
    return CastAttr(&internal_->is_floating_fill_value);
  }
  return Error::RuntimeError() << "tril op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tril", TrilOpInterpCtxImpl<schema::TrilOp>);

const HashSet<std::string>& TriuOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "diagonal", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TriuOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "diagonal") {
    return CastAttr(&internal_->diagonal);
  }
  return Error::RuntimeError() << "triu op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.triu", TriuOpInterpCtxImpl<schema::TriuOp>);

const HashSet<std::string>& TupleIdentityOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> TupleIdentityOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "tuple_identity op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.tuple_identity", TupleIdentityOpInterpCtxImpl<schema::TupleIdentityOp>);

const HashSet<std::string>& UnfoldOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "dilation_rate", "kernel_size", "padding", "strides", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UnfoldOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "dilation_rate") {
    return CastAttr(&internal_->dilation_rate);
  }
  if(attr_name == "kernel_size") {
    return CastAttr(&internal_->kernel_size);
  }
  if(attr_name == "padding") {
    return CastAttr(&internal_->padding);
  }
  if(attr_name == "strides") {
    return CastAttr(&internal_->strides);
  }
  return Error::RuntimeError() << "unfold op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.unfold", UnfoldOpInterpCtxImpl<schema::UnfoldOp>);

const HashSet<std::string>& UnfoldTensorGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dimension", "size", "step", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UnfoldTensorGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dimension") {
    return CastAttr(&internal_->dimension);
  }
  if(attr_name == "size") {
    return CastAttr(&internal_->size);
  }
  if(attr_name == "step") {
    return CastAttr(&internal_->step);
  }
  return Error::RuntimeError() << "unfold_tensor_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.unfold_tensor_grad", UnfoldTensorGradOpInterpCtxImpl<schema::UnfoldTensorGradOp>);

const HashSet<std::string>& UnfoldTensorOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dimension", "size", "step", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UnfoldTensorOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dimension") {
    return CastAttr(&internal_->dimension);
  }
  if(attr_name == "size") {
    return CastAttr(&internal_->size);
  }
  if(attr_name == "step") {
    return CastAttr(&internal_->step);
  }
  return Error::RuntimeError() << "unfold_tensor op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.unfold_tensor", UnfoldTensorOpInterpCtxImpl<schema::UnfoldTensorOp>);

const HashSet<std::string>& UniformIntOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dtype", "from", "nd_sbp", "seed", "shape", "to", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UniformIntOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dtype") {
    return CastAttr(&internal_->dtype);
  }
  if(attr_name == "from") {
    return CastAttr(&internal_->from);
  }
  if(attr_name == "nd_sbp") {
    return CastAttr(&internal_->nd_sbp);
  }
  if(attr_name == "seed") {
    return CastAttr(&internal_->seed);
  }
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  if(attr_name == "to") {
    return CastAttr(&internal_->to);
  }
  return Error::RuntimeError() << "uniform_int op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.uniform_int", UniformIntOpInterpCtxImpl<schema::UniformIntOp>);

const HashSet<std::string>& UniformOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "dtype", "from", "nd_sbp", "seed", "shape", "to", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UniformOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "dtype") {
    return CastAttr(&internal_->dtype);
  }
  if(attr_name == "from") {
    return CastAttr(&internal_->from);
  }
  if(attr_name == "nd_sbp") {
    return CastAttr(&internal_->nd_sbp);
  }
  if(attr_name == "seed") {
    return CastAttr(&internal_->seed);
  }
  if(attr_name == "shape") {
    return CastAttr(&internal_->shape);
  }
  if(attr_name == "to") {
    return CastAttr(&internal_->to);
  }
  return Error::RuntimeError() << "uniform op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.uniform", UniformOpInterpCtxImpl<schema::UniformOp>);

const HashSet<std::string>& UniqueWithCountsOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "out_idx", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UniqueWithCountsOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "out_idx") {
    return CastAttr(&internal_->out_idx);
  }
  return Error::RuntimeError() << "unique_with_counts op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.unique_with_counts", UniqueWithCountsOpInterpCtxImpl<schema::UniqueWithCountsOp>);

const HashSet<std::string>& UnpackOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "unpack_num", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UnpackOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "unpack_num") {
    return CastAttr(&internal_->unpack_num);
  }
  return Error::RuntimeError() << "unpack op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.unpack", UnpackOpInterpCtxImpl<schema::UnpackOp>);

const HashSet<std::string>& UnsortedBatchSegmentSumOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "num_segments", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UnsortedBatchSegmentSumOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "num_segments") {
    return CastAttr(&internal_->num_segments);
  }
  return Error::RuntimeError() << "unsorted_batch_segment_sum op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.unsorted_batch_segment_sum", UnsortedBatchSegmentSumOpInterpCtxImpl<schema::UnsortedBatchSegmentSumOp>);

const HashSet<std::string>& UnsortedSegmentSumLikeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UnsortedSegmentSumLikeOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  return Error::RuntimeError() << "unsorted_segment_sum_like op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.unsorted_segment_sum_like", UnsortedSegmentSumLikeOpInterpCtxImpl<schema::UnsortedSegmentSumLikeOp>);

const HashSet<std::string>& UnsortedSegmentSumOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "axis", "num_segments", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UnsortedSegmentSumOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "axis") {
    return CastAttr(&internal_->axis);
  }
  if(attr_name == "num_segments") {
    return CastAttr(&internal_->num_segments);
  }
  return Error::RuntimeError() << "unsorted_segment_sum op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.unsorted_segment_sum", UnsortedSegmentSumOpInterpCtxImpl<schema::UnsortedSegmentSumOp>);

const HashSet<std::string>& UpsampleBicubic2DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "align_corners", "data_format", "height_scale", "width_scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UpsampleBicubic2DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "align_corners") {
    return CastAttr(&internal_->align_corners);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "height_scale") {
    return CastAttr(&internal_->height_scale);
  }
  if(attr_name == "width_scale") {
    return CastAttr(&internal_->width_scale);
  }
  return Error::RuntimeError() << "upsample_bicubic_2d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.upsample_bicubic_2d_grad", UpsampleBicubic2DGradOpInterpCtxImpl<schema::UpsampleBicubic2DGradOp>);

const HashSet<std::string>& UpsampleBicubic2DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "align_corners", "data_format", "height_scale", "width_scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UpsampleBicubic2DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "align_corners") {
    return CastAttr(&internal_->align_corners);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "height_scale") {
    return CastAttr(&internal_->height_scale);
  }
  if(attr_name == "width_scale") {
    return CastAttr(&internal_->width_scale);
  }
  return Error::RuntimeError() << "upsample_bicubic_2d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.upsample_bicubic_2d", UpsampleBicubic2DOpInterpCtxImpl<schema::UpsampleBicubic2DOp>);

const HashSet<std::string>& UpsampleBilinear2DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "align_corners", "data_format", "height_scale", "width_scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UpsampleBilinear2DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "align_corners") {
    return CastAttr(&internal_->align_corners);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "height_scale") {
    return CastAttr(&internal_->height_scale);
  }
  if(attr_name == "width_scale") {
    return CastAttr(&internal_->width_scale);
  }
  return Error::RuntimeError() << "upsample_bilinear_2d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.upsample_bilinear_2d_grad", UpsampleBilinear2DGradOpInterpCtxImpl<schema::UpsampleBilinear2DGradOp>);

const HashSet<std::string>& UpsampleBilinear2DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "align_corners", "data_format", "height_scale", "width_scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UpsampleBilinear2DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "align_corners") {
    return CastAttr(&internal_->align_corners);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "height_scale") {
    return CastAttr(&internal_->height_scale);
  }
  if(attr_name == "width_scale") {
    return CastAttr(&internal_->width_scale);
  }
  return Error::RuntimeError() << "upsample_bilinear_2d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.upsample_bilinear_2d", UpsampleBilinear2DOpInterpCtxImpl<schema::UpsampleBilinear2DOp>);

const HashSet<std::string>& UpsampleGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "align_corners", "data_format", "height_scale", "interpolation", "width_scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UpsampleGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "align_corners") {
    return CastAttr(&internal_->align_corners);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "height_scale") {
    return CastAttr(&internal_->height_scale);
  }
  if(attr_name == "interpolation") {
    return CastAttr(&internal_->interpolation);
  }
  if(attr_name == "width_scale") {
    return CastAttr(&internal_->width_scale);
  }
  return Error::RuntimeError() << "upsample_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.upsample_grad", UpsampleGradOpInterpCtxImpl<schema::UpsampleGradOp>);

const HashSet<std::string>& UpsampleLinear1DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "align_corners", "data_format", "scale_factor", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UpsampleLinear1DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "align_corners") {
    return CastAttr(&internal_->align_corners);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "scale_factor") {
    return CastAttr(&internal_->scale_factor);
  }
  return Error::RuntimeError() << "upsample_linear_1d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.upsample_linear_1d_grad", UpsampleLinear1DGradOpInterpCtxImpl<schema::UpsampleLinear1DGradOp>);

const HashSet<std::string>& UpsampleLinear1DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "align_corners", "data_format", "scale_factor", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UpsampleLinear1DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "align_corners") {
    return CastAttr(&internal_->align_corners);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "scale_factor") {
    return CastAttr(&internal_->scale_factor);
  }
  return Error::RuntimeError() << "upsample_linear_1d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.upsample_linear_1d", UpsampleLinear1DOpInterpCtxImpl<schema::UpsampleLinear1DOp>);

const HashSet<std::string>& UpsampleNearest1DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "scale_factor", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UpsampleNearest1DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "scale_factor") {
    return CastAttr(&internal_->scale_factor);
  }
  return Error::RuntimeError() << "upsample_nearest_1d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.upsample_nearest_1d_grad", UpsampleNearest1DGradOpInterpCtxImpl<schema::UpsampleNearest1DGradOp>);

const HashSet<std::string>& UpsampleNearest1DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "scale_factor", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UpsampleNearest1DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "scale_factor") {
    return CastAttr(&internal_->scale_factor);
  }
  return Error::RuntimeError() << "upsample_nearest_1d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.upsample_nearest_1d", UpsampleNearest1DOpInterpCtxImpl<schema::UpsampleNearest1DOp>);

const HashSet<std::string>& UpsampleNearest2DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "height_scale", "width_scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UpsampleNearest2DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "height_scale") {
    return CastAttr(&internal_->height_scale);
  }
  if(attr_name == "width_scale") {
    return CastAttr(&internal_->width_scale);
  }
  return Error::RuntimeError() << "upsample_nearest_2d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.upsample_nearest_2d_grad", UpsampleNearest2DGradOpInterpCtxImpl<schema::UpsampleNearest2DGradOp>);

const HashSet<std::string>& UpsampleNearest2DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "height_scale", "width_scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UpsampleNearest2DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "height_scale") {
    return CastAttr(&internal_->height_scale);
  }
  if(attr_name == "width_scale") {
    return CastAttr(&internal_->width_scale);
  }
  return Error::RuntimeError() << "upsample_nearest_2d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.upsample_nearest_2d", UpsampleNearest2DOpInterpCtxImpl<schema::UpsampleNearest2DOp>);

const HashSet<std::string>& UpsampleNearest3DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "depth_scale", "height_scale", "width_scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UpsampleNearest3DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "depth_scale") {
    return CastAttr(&internal_->depth_scale);
  }
  if(attr_name == "height_scale") {
    return CastAttr(&internal_->height_scale);
  }
  if(attr_name == "width_scale") {
    return CastAttr(&internal_->width_scale);
  }
  return Error::RuntimeError() << "upsample_nearest_3d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.upsample_nearest_3d_grad", UpsampleNearest3DGradOpInterpCtxImpl<schema::UpsampleNearest3DGradOp>);

const HashSet<std::string>& UpsampleNearest3DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "data_format", "depth_scale", "height_scale", "width_scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UpsampleNearest3DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "depth_scale") {
    return CastAttr(&internal_->depth_scale);
  }
  if(attr_name == "height_scale") {
    return CastAttr(&internal_->height_scale);
  }
  if(attr_name == "width_scale") {
    return CastAttr(&internal_->width_scale);
  }
  return Error::RuntimeError() << "upsample_nearest_3d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.upsample_nearest_3d", UpsampleNearest3DOpInterpCtxImpl<schema::UpsampleNearest3DOp>);

const HashSet<std::string>& UpsampleOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "align_corners", "data_format", "height_scale", "interpolation", "width_scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UpsampleOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "align_corners") {
    return CastAttr(&internal_->align_corners);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "height_scale") {
    return CastAttr(&internal_->height_scale);
  }
  if(attr_name == "interpolation") {
    return CastAttr(&internal_->interpolation);
  }
  if(attr_name == "width_scale") {
    return CastAttr(&internal_->width_scale);
  }
  return Error::RuntimeError() << "upsample op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.upsample", UpsampleOpInterpCtxImpl<schema::UpsampleOp>);

const HashSet<std::string>& UpsampleTrilinear3DGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "align_corners", "data_format", "depth_scale", "height_scale", "width_scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UpsampleTrilinear3DGradOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "align_corners") {
    return CastAttr(&internal_->align_corners);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "depth_scale") {
    return CastAttr(&internal_->depth_scale);
  }
  if(attr_name == "height_scale") {
    return CastAttr(&internal_->height_scale);
  }
  if(attr_name == "width_scale") {
    return CastAttr(&internal_->width_scale);
  }
  return Error::RuntimeError() << "upsample_trilinear_3d_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.upsample_trilinear_3d_grad", UpsampleTrilinear3DGradOpInterpCtxImpl<schema::UpsampleTrilinear3DGradOp>);

const HashSet<std::string>& UpsampleTrilinear3DOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "align_corners", "data_format", "depth_scale", "height_scale", "width_scale", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> UpsampleTrilinear3DOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "align_corners") {
    return CastAttr(&internal_->align_corners);
  }
  if(attr_name == "data_format") {
    return CastAttr(&internal_->data_format);
  }
  if(attr_name == "depth_scale") {
    return CastAttr(&internal_->depth_scale);
  }
  if(attr_name == "height_scale") {
    return CastAttr(&internal_->height_scale);
  }
  if(attr_name == "width_scale") {
    return CastAttr(&internal_->width_scale);
  }
  return Error::RuntimeError() << "upsample_trilinear_3d op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.upsample_trilinear_3d", UpsampleTrilinear3DOpInterpCtxImpl<schema::UpsampleTrilinear3DOp>);

const HashSet<std::string>& WhereOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> WhereOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "where op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.where", WhereOpInterpCtxImpl<schema::WhereOp>);

const HashSet<std::string>& WhereScalarXOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> WhereScalarXOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "where_scalar_x op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.where_scalar_x", WhereScalarXOpInterpCtxImpl<schema::WhereScalarXOp>);

const HashSet<std::string>& WhereScalarXyOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "has_x_float_operand", "has_x_int_operand", "has_y_float_operand", "has_y_int_operand", "x_float_operand", "x_int_operand", "y_float_operand", "y_int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> WhereScalarXyOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "has_x_float_operand") {
    return CastAttr(&internal_->has_x_float_operand);
  }
  if(attr_name == "has_x_int_operand") {
    return CastAttr(&internal_->has_x_int_operand);
  }
  if(attr_name == "has_y_float_operand") {
    return CastAttr(&internal_->has_y_float_operand);
  }
  if(attr_name == "has_y_int_operand") {
    return CastAttr(&internal_->has_y_int_operand);
  }
  if(attr_name == "x_float_operand") {
    return CastAttr(&internal_->x_float_operand);
  }
  if(attr_name == "x_int_operand") {
    return CastAttr(&internal_->x_int_operand);
  }
  if(attr_name == "y_float_operand") {
    return CastAttr(&internal_->y_float_operand);
  }
  if(attr_name == "y_int_operand") {
    return CastAttr(&internal_->y_int_operand);
  }
  return Error::RuntimeError() << "where_scalar_xy op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.where_scalar_xy", WhereScalarXyOpInterpCtxImpl<schema::WhereScalarXyOp>);

const HashSet<std::string>& WhereScalarYOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "float_operand", "has_float_operand", "has_int_operand", "int_operand", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> WhereScalarYOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "float_operand") {
    return CastAttr(&internal_->float_operand);
  }
  if(attr_name == "has_float_operand") {
    return CastAttr(&internal_->has_float_operand);
  }
  if(attr_name == "has_int_operand") {
    return CastAttr(&internal_->has_int_operand);
  }
  if(attr_name == "int_operand") {
    return CastAttr(&internal_->int_operand);
  }
  return Error::RuntimeError() << "where_scalar_y op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.where_scalar_y", WhereScalarYOpInterpCtxImpl<schema::WhereScalarYOp>);

const HashSet<std::string>& XdivyOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> XdivyOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "xdivy op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.xdivy", XdivyOpInterpCtxImpl<schema::XdivyOp>);

const HashSet<std::string>& XdivyXGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> XdivyXGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "xdivy_x_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.xdivy_x_grad", XdivyXGradOpInterpCtxImpl<schema::XdivyXGradOp>);

const HashSet<std::string>& XdivyYGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> XdivyYGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "xdivy_y_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.xdivy_y_grad", XdivyYGradOpInterpCtxImpl<schema::XdivyYGradOp>);

const HashSet<std::string>& XlogyOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> XlogyOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "xlogy op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.xlogy", XlogyOpInterpCtxImpl<schema::XlogyOp>);

const HashSet<std::string>& XlogyXGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> XlogyXGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "xlogy_x_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.xlogy_x_grad", XlogyXGradOpInterpCtxImpl<schema::XlogyXGradOp>);

const HashSet<std::string>& XlogyYGradOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> XlogyYGradOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "xlogy_y_grad op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.xlogy_y_grad", XlogyYGradOpInterpCtxImpl<schema::XlogyYGradOp>);

const HashSet<std::string>& ZeroLikeOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> ZeroLikeOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "zero_like op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user.zero_like", ZeroLikeOpInterpCtxImpl<schema::ZeroLikeOp>);

const HashSet<std::string>& _ncclLogicalAllGatherNoncontinuousOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "in_split_axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> _ncclLogicalAllGatherNoncontinuousOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "in_split_axis") {
    return CastAttr(&internal_->in_split_axis);
  }
  return Error::RuntimeError() << "_nccl_logical_all_gather_noncontinuous op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user._nccl_logical_all_gather_noncontinuous", _ncclLogicalAllGatherNoncontinuousOpInterpCtxImpl<schema::_ncclLogicalAllGatherNoncontinuousOp>);

const HashSet<std::string>& _ncclLogicalAllGatherOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> _ncclLogicalAllGatherOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "_nccl_logical_all_gather op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user._nccl_logical_all_gather", _ncclLogicalAllGatherOpInterpCtxImpl<schema::_ncclLogicalAllGatherOp>);

const HashSet<std::string>& _ncclLogicalAllReduceOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> _ncclLogicalAllReduceOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "_nccl_logical_all_reduce op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user._nccl_logical_all_reduce", _ncclLogicalAllReduceOpInterpCtxImpl<schema::_ncclLogicalAllReduceOp>);

const HashSet<std::string>& _ncclLogicalReduceScatterOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> _ncclLogicalReduceScatterOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "_nccl_logical_reduce_scatter op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user._nccl_logical_reduce_scatter", _ncclLogicalReduceScatterOpInterpCtxImpl<schema::_ncclLogicalReduceScatterOp>);

const HashSet<std::string>& _ncclLogicalS2sOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "in_split_axis", "out_split_axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> _ncclLogicalS2sOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "in_split_axis") {
    return CastAttr(&internal_->in_split_axis);
  }
  if(attr_name == "out_split_axis") {
    return CastAttr(&internal_->out_split_axis);
  }
  return Error::RuntimeError() << "_nccl_logical_s2s op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user._nccl_logical_s2s", _ncclLogicalS2sOpInterpCtxImpl<schema::_ncclLogicalS2sOp>);

const HashSet<std::string>& _ncclLogical_2DSameDim0All2allOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "in_dim1_split_axis", "out_dim1_split_axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> _ncclLogical_2DSameDim0All2allOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "in_dim1_split_axis") {
    return CastAttr(&internal_->in_dim1_split_axis);
  }
  if(attr_name == "out_dim1_split_axis") {
    return CastAttr(&internal_->out_dim1_split_axis);
  }
  return Error::RuntimeError() << "_nccl_logical_2D_same_dim0_all2all op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user._nccl_logical_2D_same_dim0_all2all", _ncclLogical_2DSameDim0All2allOpInterpCtxImpl<schema::_ncclLogical_2DSameDim0All2allOp>);

const HashSet<std::string>& _ncclLogical_2DSameDim0AllGatherNoncontinuousOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { "in_dim1_split_axis", };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> _ncclLogical_2DSameDim0AllGatherNoncontinuousOp::GetAttr(const std::string& attr_name) const {
  if(attr_name == "in_dim1_split_axis") {
    return CastAttr(&internal_->in_dim1_split_axis);
  }
  return Error::RuntimeError() << "_nccl_logical_2D_same_dim0_all_gather_noncontinuous op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user._nccl_logical_2D_same_dim0_all_gather_noncontinuous", _ncclLogical_2DSameDim0AllGatherNoncontinuousOpInterpCtxImpl<schema::_ncclLogical_2DSameDim0AllGatherNoncontinuousOp>);

const HashSet<std::string>& _ncclLogical_2DSameDim0AllGatherOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> _ncclLogical_2DSameDim0AllGatherOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "_nccl_logical_2D_same_dim0_all_gather op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user._nccl_logical_2D_same_dim0_all_gather", _ncclLogical_2DSameDim0AllGatherOpInterpCtxImpl<schema::_ncclLogical_2DSameDim0AllGatherOp>);

const HashSet<std::string>& _ncclLogical_2DSameDim0AllReduceOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> _ncclLogical_2DSameDim0AllReduceOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "_nccl_logical_2D_same_dim0_all_reduce op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user._nccl_logical_2D_same_dim0_all_reduce", _ncclLogical_2DSameDim0AllReduceOpInterpCtxImpl<schema::_ncclLogical_2DSameDim0AllReduceOp>);

const HashSet<std::string>& _ncclLogical_2DSameDim1AllReduceOpInterpCtx::AttrNames() const {
  static const HashSet<std::string> attr_names = { };
  return attr_names;
}

namespace schema {
Maybe<AttrVal> _ncclLogical_2DSameDim1AllReduceOp::GetAttr(const std::string& attr_name) const {
  return Error::RuntimeError() << "_nccl_logical_2D_same_dim1_all_reduce op has no attribute named " << attr_name;
}
}  // namespace schema

REGISTER_OP_INTERP_CTX("user._nccl_logical_2D_same_dim1_all_reduce", _ncclLogical_2DSameDim1AllReduceOpInterpCtxImpl<schema::_ncclLogical_2DSameDim1AllReduceOp>);

} // namespace oneflow
