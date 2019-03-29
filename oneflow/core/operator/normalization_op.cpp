#include "oneflow/core/operator/normalization_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

inline bool IsFwBwSplit() {
  return Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf();
}

}  // namespace

void NormalizationOp::InitFromOpConf() {
  const NormalizationOpConf& conf = op_conf().normalization_conf();
#ifdef WITH_CUDA
  if (DevIsGpuAndEnableCudnn()) { CHECK_GE(conf.epsilon(), CUDNN_BN_MIN_EPSILON); }
#endif
  CHECK_GE(conf.momentum(), 0);
  CHECK_LE(conf.momentum(), 1);
  EnrollInputBn("in");
  EnrollOutputBn("out");
  if (IsFwBwSplit()) {
    CHECK(conf.has_moving_mean());
    CHECK(conf.has_moving_variance());
    EnrollInputBn("moving_mean")->set_is_mutable(op_conf().trainable());
    EnrollInputBn("moving_variance")->set_is_mutable(op_conf().trainable());
    if (conf.has_gamma()) {
      EnrollInputBn("gamma");
    } else {
      if (DevIsGpuAndEnableCudnn()) {
        EnrollConstBufBn("gamma");
      } else {
        UNIMPLEMENTED();
      }
    }
    if (conf.has_beta()) {
      EnrollInputBn("beta");
    } else {
      if (DevIsGpuAndEnableCudnn()) {
        EnrollConstBufBn("beta");
      } else {
        UNIMPLEMENTED();
      }
    }
    if (conf.is_training()) {
      EnrollOutputBn("mean", false);
      EnrollOutputBn("inv_variance", false);
    }
  } else {
    EnrollForwardModelBn("moving_mean");
    EnrollForwardModelBn("moving_variance");
    if (conf.center()) {
      EnrollModelBn("beta");
    } else {
      if (DevIsGpuAndEnableCudnn()) {
        EnrollConstBufBn("beta");
        EnrollBwBufBn("beta_diff");
      } else {
        UNIMPLEMENTED();
      }
    }
    if (conf.scale()) {
      EnrollModelBn("gamma");
    } else {
      if (DevIsGpuAndEnableCudnn()) {
        EnrollConstBufBn("gamma");
        EnrollBwBufBn("gamma_diff");
      } else {
        UNIMPLEMENTED();
      }
    }
    if (conf.is_training()) {
      EnrollDataTmpBn("mean");
      EnrollDataTmpBn("inv_variance");
    } else {
      EnrollBwBufBn("inv_variance");
    }
  }
}

const PbMessage& NormalizationOp::GetCustomizedConf() const {
  return op_conf().normalization_conf();
}

void NormalizationOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const NormalizationOpConf& conf = op_conf().normalization_conf();
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const DataType data_type = in->data_type();
  *GetBlobDesc4BnInOp("out") = *in;
  const bool is_fw_bw_split = IsFwBwSplit();
  const Shape param_shape({in->shape().At(conf.axis())});
  const auto CheckParamBlobDesc = [&](const std::string& bn) {
    const BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn);
    if (blob_desc != nullptr) {
      CHECK_EQ(blob_desc->data_type(), data_type);
      CHECK_EQ(blob_desc->shape(), param_shape);
    }
  };
  const auto SetParamBlobDesc = [&](const std::string& bn) {
    BlobDesc* blob_desc = GetBlobDesc4BnInOp(bn);
    if (blob_desc != nullptr) {
      blob_desc->set_data_type(data_type);
      blob_desc->mut_shape() = param_shape;
    }
  };
  if (is_fw_bw_split) {
    CheckParamBlobDesc("moving_mean");
    CheckParamBlobDesc("moving_variance");
    CheckParamBlobDesc("beta");
    CheckParamBlobDesc("gamma");
  } else {
    SetParamBlobDesc("moving_mean");
    SetParamBlobDesc("moving_variance");
    SetParamBlobDesc("beta");
    SetParamBlobDesc("gamma");
  }
  SetParamBlobDesc("mean");
  SetParamBlobDesc("inv_variance");
}

void NormalizationOp::InferBwBufBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const Shape param_shape({in->shape().At(op_conf().normalization_conf().axis())});
  const DataType data_type = in->data_type();
  BlobDesc* beta_diff = GetBlobDesc4BnInOp("beta_diff");
  if (beta_diff != nullptr) {
    beta_diff->set_data_type(data_type);
    beta_diff->mut_shape() = param_shape;
  }
  BlobDesc* gamma_diff = GetBlobDesc4BnInOp("gamma_diff");
  if (gamma_diff != nullptr) {
    gamma_diff->set_data_type(data_type);
    gamma_diff->mut_shape() = param_shape;
  }
  BlobDesc* inv_variance = GetBlobDesc4BnInOp("inv_variance");
  if (inv_variance != nullptr) {
    inv_variance->set_data_type(data_type);
    inv_variance->mut_shape() = param_shape;
  }
}

REGISTER_OP(OperatorConf::kNormalizationConf, NormalizationOp);

}  // namespace oneflow
