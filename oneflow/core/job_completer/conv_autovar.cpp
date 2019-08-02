#include "oneflow/core/job_completer/autovar.h"

namespace oneflow {

namespace {

template<typename T>
struct GetConfHelper {
  static T* MuatableConvNdConf(OperatorConf* conv_nd_conf);
  static T ConvNdConf(const OperatorConf& conv_nd_conf);
};

template<>
struct GetConfHelper<Conv1DOpConf> {
  static Conv1DOpConf* MuatableConvNdConf(OperatorConf* conv_nd_conf) {
    return conv_nd_conf->mutable_conv_1d_conf();
  }
  static Conv1DOpConf ConvNdConf(const OperatorConf& conv_nd_conf) {
    CHECK(conv_nd_conf.has_conv_1d_conf());
    return conv_nd_conf.conv_1d_conf();
  }
};

template<>
struct GetConfHelper<Conv2DOpConf> {
  static Conv2DOpConf* MuatableConvNdConf(OperatorConf* conv_nd_conf) {
    return conv_nd_conf->mutable_conv_2d_conf();
  }
  static Conv2DOpConf ConvNdConf(const OperatorConf& conv_nd_conf) {
    CHECK(conv_nd_conf.has_conv_2d_conf());
    return conv_nd_conf.conv_2d_conf();
  }
};

template<>
struct GetConfHelper<Conv3DOpConf> {
  static Conv3DOpConf* MuatableConvNdConf(OperatorConf* conv_nd_conf) {
    return conv_nd_conf->mutable_conv_3d_conf();
  }
  static Conv3DOpConf ConvNdConf(const OperatorConf& conv_nd_conf) {
    CHECK(conv_nd_conf.has_conv_3d_conf());
    return conv_nd_conf.conv_3d_conf();
  }
};

template<typename T>
void GenerateInputVarOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<const BlobDesc&(const std::string&)>& BlobDesc4ModelBn) {
  const T& conf = GetConfHelper<T>::ConvNdConf(op.op_conf());
  OperatorConf conv_nd_conf(op.op_conf());
  T* mut_conf = GetConfHelper<T>::MuatableConvNdConf(&conv_nd_conf);

  if (!conf.has_weight()) {
    OperatorConf weight_var_op =
        GenerateVariableOpConf(BlobDesc4ModelBn("weight"), op.op_name() + "-weight", "weight");
    if (conf.has_weight_initializer()) {
      *(weight_var_op.mutable_variable_conf()->mutable_initializer()) = conf.weight_initializer();
    }
    op_confs->push_back(weight_var_op);
    mut_conf->set_weight(weight_var_op.name() + "/out");
  }
  if (conf.use_bias()) {
    if (!conf.has_bias()) {
      OperatorConf bias_var_op =
          GenerateVariableOpConf(BlobDesc4ModelBn("bias"), op.op_name() + "-bias", "bias");
      if (conf.has_bias_initializer()) {
        *(bias_var_op.mutable_variable_conf()->mutable_initializer()) = conf.bias_initializer();
      }
      op_confs->push_back(bias_var_op);
      mut_conf->set_bias(bias_var_op.name() + "/out");
    }
  }
  op_confs->push_back(conv_nd_conf);
}

}  // namespace

REGISTER_OP_INPUT_VAR(OperatorConf::kConv1DConf, (&GenerateInputVarOpConf<Conv1DOpConf>));
REGISTER_OP_INPUT_VAR(OperatorConf::kConv2DConf, (&GenerateInputVarOpConf<Conv2DOpConf>));
REGISTER_OP_INPUT_VAR(OperatorConf::kConv3DConf, (&GenerateInputVarOpConf<Conv3DOpConf>));

}  // namespace oneflow
