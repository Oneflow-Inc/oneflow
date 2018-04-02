#include "oneflow/core/kernel/basic_gru_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BasicGruKernel<device_type, T>::VirtualKernelInit(
    const ParallelContext* parallel_ctx) {
  ActivationType activation_type =
      this->op_conf().basic_rnn_conf().activation();
  if (activation_type == kTanH) {
    activation_fw_func_ = &KernelUtil<device_type, T>::TanH;
    activation_bw_func_ = &KernelUtil<device_type, T>::TanHBackward;
  } else if (activation_type == kSigmoid) {
    activation_fw_func_ = &KernelUtil<device_type, T>::Sigmoid;
    activation_bw_func_ = &KernelUtil<device_type, T>::SigmoidBackward;
  } else if (activation_type == kRelu) {
    activation_fw_func_ = &KernelUtil<device_type, T>::Relu;
    activation_bw_func_ = &KernelUtil<device_type, T>::ReluBackward;
  } else {
    UNEXPECTED_RUN();
  }
}

template<DeviceType device_type, typename T>
const PbMessage& BasicGruKernel<device_type, T>::GetRecurrentOpConf() const {
  return this->op_conf().basic_gru_conf();
}

template<DeviceType device_type, typename T>
bool BasicGruKernel<device_type, T>::HasInitHiddenInitializer() const {
  return this->op_conf().basic_gru_conf().has_init_hidden_initializer();
}

template<DeviceType device_type, typename T>
void BasicGruKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);
  Blob* reset_out_blob = BnInOp2Blob("reset_out");
  Blob* update_out_blob = BnInOp2Blob("update_out");
  Blob* candidate_out_blob = BnInOp2Blob("candidate_out");
  Blob* tmp_data_blob = BnInOp2Blob("tmp_data");
  Blob* out_blob = BnInOp2Blob("out");

  BasicGruKernelUtil<device_type, T>::ComputeGateForward(
      ctx, hidden_blob, BnInOp2Blob("i2h_r_weight"),
      BnInOp2Blob("h2h_r_weight"), BnInOp2Blob("bias_r"), reset_out_blob,
      BnInOp2Blob);

  BasicGruKernelUtil<device_type, T>::ComputeGateForward(
      ctx, hidden_blob, BnInOp2Blob("i2h_z_weight"),
      BnInOp2Blob("h2h_z_weight"), BnInOp2Blob("bias_z"), update_out_blob,
      BnInOp2Blob);

  BasicGruKernelUtil<device_type, T>::ComputeCandidateForward(
      ctx, hidden_blob, reset_out_blob, tmp_data_blob, candidate_out_blob,
      activation_fw_func_, BnInOp2Blob);

  BasicGruKernelUtil<device_type, T>::ComputeOutForward(
      ctx, hidden_blob, candidate_out_blob, tmp_data_blob, update_out_blob,
      out_blob);

  // rec_out = out
  BnInOp2Blob("rec_out")->CopyDataContentFrom<device_type>(ctx.device_ctx,
                                                           out_blob);
}

template<DeviceType device_type, typename T>
void BasicGruKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* candidate_data_diff_blob = BnInOp2Blob("candidate_data_diff");
  Blob* update_data_diff_blob = BnInOp2Blob("update_data_diff");
  Blob* tmp_data_blob = BnInOp2Blob("tmp_data");
  const Blob* reset_out_blob = BnInOp2Blob("reset_out");
  Blob* reset_data_diff_blob = BnInOp2Blob("reset_data_diff");
  Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);
  Blob* hidden_diff_blob = this->GetHiddenDiffBlob(BnInOp2Blob);
  // tmp blob to storage out diff
  Blob* plus_diff_blob = BnInOp2Blob("tmp_data");
  plus_diff_blob->CopyDataContentFrom<device_type>(ctx.device_ctx,
                                                   BnInOp2Blob("out_diff"));

  if (BnInOp2Blob("in")->col_id() != BnInOp2Blob("in")->max_col_id()) {
    // plus_diff += rec_out_diff
    KernelUtil<device_type, T>::Axpy(
        ctx.device_ctx, plus_diff_blob->shape().elem_cnt(), static_cast<T>(1),
        BnInOp2Blob("rec_out_diff")->dptr<T>(), 1,
        plus_diff_blob->mut_dptr<T>(), 1);
  }

  BasicGruKernelUtil<device_type, T>::ComputeTmpActivationDataDiff(
      ctx, plus_diff_blob, hidden_blob, update_data_diff_blob,
      candidate_data_diff_blob, tmp_data_blob, reset_data_diff_blob,
      activation_bw_func_, BnInOp2Blob);

  BasicGruKernelUtil<device_type, T>::ComputeWeightDiff(
      ctx, BnInOp2Blob("in"), hidden_blob, reset_data_diff_blob,
      BnInOp2Blob("i2h_r_weight_diff"), BnInOp2Blob("h2h_r_weight_diff"));
  BasicGruKernelUtil<device_type, T>::ComputeWeightDiff(
      ctx, BnInOp2Blob("in"), hidden_blob, update_data_diff_blob,
      BnInOp2Blob("i2h_z_weight_diff"), BnInOp2Blob("h2h_z_weight_diff"));
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, reset_out_blob->shape().elem_cnt(),
      hidden_blob->dptr<T>(), reset_out_blob->dptr<T>(),
      tmp_data_blob->mut_dptr<T>());
  BasicGruKernelUtil<device_type, T>::ComputeWeightDiff(
      ctx, BnInOp2Blob("in"), tmp_data_blob, candidate_data_diff_blob,
      BnInOp2Blob("i2h_weight_diff"), BnInOp2Blob("h2h_weight_diff"));

  BasicGruKernelUtil<device_type, T>::ComputeBiasDiff(
      ctx, reset_data_diff_blob, update_data_diff_blob,
      candidate_data_diff_blob, BnInOp2Blob("bias_r_diff"),
      BnInOp2Blob("bias_z_diff"), BnInOp2Blob("bias_diff"), BnInOp2Blob);

  if (BnInOp2Blob("in_diff") != nullptr) {
    BasicGruKernelUtil<device_type, T>::ComputeInDiff(
        ctx, reset_data_diff_blob, update_data_diff_blob,
        candidate_data_diff_blob, BnInOp2Blob("in_diff"), BnInOp2Blob);
  }

  if (BnInOp2Blob("in")->col_id() != 0 || this->NeedExternalH0()
      || this->op_conf().basic_gru_conf().is_init_hidden_trainable()) {
    BasicGruKernelUtil<device_type, T>::ComputeHiddenDiff(
        ctx, hidden_blob, hidden_diff_blob, tmp_data_blob,
        candidate_data_diff_blob, reset_data_diff_blob, update_data_diff_blob,
        plus_diff_blob, BnInOp2Blob);
  }
}

template<DeviceType device_type, typename T>
void BasicGruKernel<device_type, T>::VirtualInitModelBlobsWithRandomSeed(
    const KernelCtx& ctx, std::mt19937 random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
#define OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(modelname, type)               \
  KernelUtil<device_type, T>::InitializeWithProperConf(                        \
      ctx.device_ctx,                                                          \
      OF_PB_POINTER_GET(this->op_conf().basic_gru_conf(), type##_initializer), \
      random_seed_gen(), BnInOp2Blob(#modelname))
  OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(i2h_r_weight, weight);
  OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(h2h_r_weight, weight);
  OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(i2h_z_weight, weight);
  OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(h2h_z_weight, weight);
  OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(i2h_weight, weight);
  OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(h2h_weight, weight);
  if ((OF_PB_POINTER_GET(this->op_conf().basic_gru_conf(), bias_initializer))
      == nullptr) {
#define OF_GRU_INIT_MODEL_BIAS(modelname, default_value)                   \
  InitializerConf modelname##_fill_conf;                                   \
  modelname##_fill_conf.mutable_constant_conf()->set_value(default_value); \
  InitializerConf* modelname##_init_conf = &modelname##_fill_conf;         \
  KernelUtil<device_type, T>::InitializeWithProperConf(                    \
      ctx.device_ctx, modelname##_init_conf, 0, BnInOp2Blob(#modelname))
    OF_GRU_INIT_MODEL_BIAS(bias_r, 1.f);
    OF_GRU_INIT_MODEL_BIAS(bias_z, 1.f);
    OF_GRU_INIT_MODEL_BIAS(bias, 0.f);
#undef OF_GRU_INIT_MODEL_BIAS
  } else {
    OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(bias_r, bias);
    OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(bias_z, bias);
    OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(bias, bias);
  }
#undef OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED
}

template<DeviceType device_type, typename T>
void BasicGruKernel<device_type, T>::VirtualInitModelBlobsWithDir(
    const KernelCtx& ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
#define OF_GRU_INIT_MODEL_BLOB_WITH_DIR(modelname)       \
  KernelUtil<device_type, T>::InitializeWithModelDir(    \
      ctx.device_ctx, part_id, part_num, model_load_dir, \
      BnInOp2Blob(#modelname), #modelname,               \
      BnInOp2Blob(#modelname)->shape().At(0),            \
      BnInOp2Blob(#modelname)->shape().Count(1))
  OF_GRU_INIT_MODEL_BLOB_WITH_DIR(i2h_r_weight);
  OF_GRU_INIT_MODEL_BLOB_WITH_DIR(h2h_r_weight);
  OF_GRU_INIT_MODEL_BLOB_WITH_DIR(i2h_z_weight);
  OF_GRU_INIT_MODEL_BLOB_WITH_DIR(h2h_z_weight);
#undef OF_GRU_INIT_MODEL_BLOB_WITH_DIR
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx.device_ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias_r"),
      "bias_r", BnInOp2Blob("bias_r")->shape().At(0), 1);
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx.device_ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias_z"),
      "bias_z", BnInOp2Blob("bias_z")->shape().At(0), 1);
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx.device_ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias"),
      "bias", BnInOp2Blob("bias")->shape().At(0), 1);
}

template<DeviceType device_type, typename T>
void BasicGruKernel<device_type, T>::InitModelTmpBlobs(
    const KernelCtx& ctx, const ParallelContext* parallel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* h0_blob = BnInOp2Blob("h0");
  if (!(this->NeedExternalH0()) && h0_blob != nullptr) {
    const InitializerConf* init_hidden_initializer = nullptr;
    if (HasInitHiddenInitializer()) {
      init_hidden_initializer =
          static_cast<const InitializerConf*>(&GetMessageFromPbMessage(
              GetRecurrentOpConf(), "init_hidden_initializer"));
    }
    int64_t random_seed = *static_cast<int64_t*>(ctx.other);
    std::mt19937 random_seed_gen(random_seed);
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx.device_ctx, init_hidden_initializer, random_seed_gen(), h0_blob);
  }
  InitializerConf bias_multiplier_fill_conf;
  bias_multiplier_fill_conf.mutable_constant_conf()->set_value(1.f);
  InitializerConf* bias_multiplier_init_conf = &bias_multiplier_fill_conf;
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx.device_ctx, bias_multiplier_init_conf, 0,
      BnInOp2Blob("bias_multiplier"));
}

template<DeviceType device_type, typename T>
void BasicGruKernelUtil<device_type, T>::ComputeGateForward(
    const KernelCtx& ctx, const Blob* hidden, const Blob* i2h_weight,
    const Blob* h2h_weight, const Blob* bias, Blob* gate_out,
    std::function<Blob*(const std::string&)> BnInOp2Blob) {
  Blob* gate_input_blob =
      BnInOp2Blob("gate_input");  // reused by two activation
  // gate_input = in * i2h_weight'
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, static_cast<T>(1),
      static_cast<T>(0), BnInOp2Blob("in"), i2h_weight, gate_input_blob);
  // gate_inout += hidden * h2h_weight'
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(1),
                                       hidden, h2h_weight, gate_input_blob);
  // gate_input += bias_multiplier * bias
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), BnInOp2Blob("bias_multiplier"), bias, gate_input_blob);
  // gate_out = sigmoid(gate_input)
  KernelUtil<device_type, T>::Sigmoid(
      ctx.device_ctx, gate_input_blob->shape().elem_cnt(),
      gate_input_blob->dptr<T>(), gate_out->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void BasicGruKernelUtil<device_type, T>::ComputeCandidateForward(
    const KernelCtx& ctx, const Blob* hidden, Blob* reset_out, Blob* tmp_data,
    Blob* candidate_out, FwActivationFunc<device_type, T> activation_fw_func_,
    std::function<Blob*(const std::string&)> BnInOp2Blob) {
  Blob* candidate_data_blob = BnInOp2Blob("candidate_data");
  // tmp_data = hidden .*reset_out
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, reset_out->shape().elem_cnt(),
                                  reset_out->dptr<T>(), hidden->dptr<T>(),
                                  tmp_data->mut_dptr<T>());
  // candidate_data = tmp_data * h2h_weight'
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       tmp_data, BnInOp2Blob("h2h_weight"),
                                       candidate_data_blob);
  // candidate_data += in * i2h_weight'
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, static_cast<T>(1),
      static_cast<T>(1), BnInOp2Blob("in"), BnInOp2Blob("i2h_weight"),
      candidate_data_blob);
  // candidate_data += bias_multiplier * bias
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), BnInOp2Blob("bias_multiplier"), BnInOp2Blob(" bias"),
      candidate_data_blob);
  // candidate_out = activation(candidate_data)
  (*activation_fw_func_)(
      ctx.device_ctx, candidate_data_blob->shape().elem_cnt(),
      candidate_data_blob->dptr<T>(), candidate_out->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void BasicGruKernelUtil<device_type, T>::ComputeOutForward(
    const KernelCtx& ctx, const Blob* hidden, Blob* candidate_out,
    Blob* temp_data, Blob* update_out, Blob* out) {
  // out = candidate_out .* update_out
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, update_out->shape().elem_cnt(), candidate_out->dptr<T>(),
      update_out->dptr<T>(), out->mut_dptr<T>());
  // temp_data = hidden .* update_out
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, update_out->shape().elem_cnt(), hidden->dptr<T>(),
      out->dptr<T>(), temp_data->mut_dptr<T>());
  // out -= temp_data
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, temp_data->shape().elem_cnt(), static_cast<T>(-1),
      temp_data->dptr<T>(), 1, out->mut_dptr<T>(), 1);
  // out += hidden
  KernelUtil<device_type, T>::Axpy(ctx.device_ctx, hidden->shape().elem_cnt(),
                                   static_cast<T>(1), hidden->dptr<T>(), 1,
                                   out->mut_dptr<T>(), 1);
}

template<DeviceType device_type, typename T>
void BasicGruKernelUtil<device_type, T>::ComputeTmpActivationDataDiff(
    const KernelCtx& ctx, Blob* out_diff, Blob* hidden, Blob* update_d_diff,
    Blob* candidate_d_diff, Blob* tmp_data, Blob* reset_d_diff,
    BwActivationFunc<device_type, T> activation_bw_func_,
    std::function<Blob*(const std::string&)> BnInOp2Blob) {
  // reuse memory
  Blob* update_out_bran_diff_blob = BnInOp2Blob("out");
  // candidate_o_diff = update_out .* out_diff
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, BnInOp2Blob("update_out")->shape().elem_cnt(),
      BnInOp2Blob("update_out")->dptr<T>(), out_diff->dptr<T>(),
      BnInOp2Blob("candidate_out_diff")->mut_dptr<T>());
  // candidate_d_diff = activation_bw_func_(canidate_out, canidate_o_diff)
  (*activation_bw_func_)(ctx.device_ctx,
                         BnInOp2Blob("candidate_out")->shape().elem_cnt(),
                         BnInOp2Blob("candidate_data")->dptr<T>(),
                         BnInOp2Blob("candidate_out")->dptr<T>(),
                         BnInOp2Blob("candidate_out_diff")->dptr<T>(),
                         candidate_d_diff->mut_dptr<T>());
  // update_o_diff = candidate_out .* out_diff
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, out_diff->shape().elem_cnt(),
      BnInOp2Blob("candidate_out")->dptr<T>(), out_diff->dptr<T>(),
      BnInOp2Blob("update_out_diff")->mut_dptr<T>());
  // update_o_bran_diff = hidden .* out_diff
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, hidden->shape().elem_cnt(),
                                  hidden->dptr<T>(), out_diff->dptr<T>(),
                                  update_out_bran_diff_blob->mut_dptr<T>());
  // update_o_diff -= update_o_bran_diff
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, update_out_bran_diff_blob->shape().elem_cnt(),
      static_cast<T>(-1), update_out_bran_diff_blob->dptr<T>(), 1,
      BnInOp2Blob("update_out_diff")->mut_dptr<T>(), 1);
  // update_d_diff = update_out * (1 - update_out) * update_o_diff
  KernelUtil<device_type, T>::SigmoidBackward(
      ctx.device_ctx, BnInOp2Blob("update_out")->shape().elem_cnt(),
      BnInOp2Blob("gate_input")->dptr<T>(),
      BnInOp2Blob("update_out")->dptr<T>(),
      BnInOp2Blob("update_out_diff")->dptr<T>(), update_d_diff->mut_dptr<T>());
  // tmp_data = candidate_d_diff * h2h_weight
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(0), candidate_d_diff, BnInOp2Blob("h2h_weight"), tmp_data);
  // reset_o_diff = hidden .* tmp_data
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, hidden->shape().elem_cnt(),
                                  hidden->dptr<T>(), tmp_data->dptr<T>(),
                                  BnInOp2Blob("reset_out_diff")->mut_dptr<T>());
  // reset_d_diff
  KernelUtil<device_type, T>::SigmoidBackward(
      ctx.device_ctx, BnInOp2Blob("reset_out")->shape().elem_cnt(),
      BnInOp2Blob("gate_input")->dptr<T>(), BnInOp2Blob("reset_out")->dptr<T>(),
      BnInOp2Blob("reset_out_diff")->dptr<T>(), reset_d_diff->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void BasicGruKernelUtil<device_type, T>::ComputeWeightDiff(
    const KernelCtx& ctx, const Blob* in_data, Blob* hidden, Blob* out_diff,
    Blob* i2h_weight_diff, Blob* h2h_weight_diff) {
  // h2h_weght_diff = out_diff' * hidden
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       out_diff, hidden, h2h_weight_diff);
  // i2h_weght_diff = out_diff' * in
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       out_diff, in_data, i2h_weight_diff);
}

template<DeviceType device_type, typename T>
void BasicGruKernelUtil<device_type, T>::ComputeBiasDiff(
    const KernelCtx& ctx, Blob* reset_d_diff, Blob* update_d_diff,
    Blob* candidate_d_diff, Blob* bias_r_diff, Blob* bias_z_diff,
    Blob* bias_diff, std::function<Blob*(const std::string&)> BnInOp2Blob) {
// bias_diff = bias_multiplier' * data_diff
#define OF_GRU_COMPUTE_BIAS_DIFF(bias_diff, gate_d_diff)                \
  if (bias_diff != nullptr) {                                           \
    KernelUtil<device_type, T>::BlobGemm(                               \
        ctx.device_ctx, CblasTrans, CblasNoTrans, static_cast<T>(1),    \
        static_cast<T>(0), BnInOp2Blob("bias_multiplier"), gate_d_diff, \
        bias_diff);                                                     \
  }
  OF_GRU_COMPUTE_BIAS_DIFF(bias_r_diff, reset_d_diff)
  OF_GRU_COMPUTE_BIAS_DIFF(bias_z_diff, update_d_diff)
  OF_GRU_COMPUTE_BIAS_DIFF(bias_diff, candidate_d_diff)
#undef OF_GRU_COMPUTE_BIAS_DIFF
}

template<DeviceType device_type, typename T>
void BasicGruKernelUtil<device_type, T>::ComputeInDiff(
    const KernelCtx& ctx, Blob* reset_d_diff, Blob* update_d_diff,
    Blob* candidate_d_diff, Blob* in_diff,
    std::function<Blob*(const std::string&)> BnInOp2Blob) {
  // in_diff = reset_data_diff * i2h_r_weght
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(0), reset_d_diff, BnInOp2Blob("i2h_r_weight"), in_diff);
  // in_diff += update_data_diff * i2h_z_weght
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), update_d_diff, BnInOp2Blob("i2h_z_weight"), in_diff);
  // in_diff += candidate_data_diff * i2h_weght
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), candidate_d_diff, BnInOp2Blob("i2h_weight"), in_diff);
}

template<DeviceType device_type, typename T>
void BasicGruKernelUtil<device_type, T>::ComputeHiddenDiff(
    const KernelCtx& ctx, Blob* hidden, Blob* hidden_diff, Blob* tmp_data,
    Blob* candidate_d_diff, Blob* reset_d_diff, Blob* update_d_diff,
    Blob* out_diff, std::function<Blob*(const std::string&)> BnInOp2Blob) {
  // tmp_data  = candidate_d_diff * h2h_weight
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(0), candidate_d_diff, BnInOp2Blob("h2h_weight"), tmp_data);
  // hidden_diff = reset_out .* tmp_data
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, hidden_diff->shape().elem_cnt(),
      BnInOp2Blob("reset_out")->dptr<T>(), tmp_data->dptr<T>(),
      hidden_diff->mut_dptr<T>());
  // hidden_diff += reset_d_diff * h2h_r_weight
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), reset_d_diff, BnInOp2Blob("h2h_r_weight"),
      hidden_diff);
  // hidden_diff += update_d_diff * h2h_z_weight
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), update_d_diff, BnInOp2Blob("h2h_z_weight"),
      hidden_diff);
  // reuse hidden_blob
  // hidden = out_diff .* update_out
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, hidden->shape().elem_cnt(), out_diff->dptr<T>(),
      BnInOp2Blob("update_out")->dptr<T>(), hidden->mut_dptr<T>());
  // hidden -= out_diff
  KernelUtil<device_type, T>::Axpy(ctx.device_ctx, hidden->shape().elem_cnt(),
                                   static_cast<T>(-1), out_diff->dptr<T>(), 1,
                                   hidden->mut_dptr<T>(), 1);
  // hidden_diff -= hidden
  KernelUtil<device_type, T>::Axpy(ctx.device_ctx, hidden->shape().elem_cnt(),
                                   static_cast<T>(-1), hidden->dptr<T>(), 1,
                                   hidden_diff->mut_dptr<T>(), 1);
}
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBasicGruConf, BasicGruKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
