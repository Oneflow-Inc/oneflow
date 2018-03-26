#include "oneflow/core/kernel/basic_gru_kernel.h"

namespace oneflow {

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
  Blob* reset_gate_data_blob = BnInOp2Blob("reset_gate_data");
  Blob* reset_gate_out_blob = BnInOp2Blob("reset_gate_out");
  Blob* update_gate_data_blob = BnInOp2Blob("update_gate_data");
  Blob* update_gate_out_blob = BnInOp2Blob("update_gate_out");
  Blob* candidate_hidden_data_blob = BnInOp2Blob("candidate_hidden_data");
  Blob* candidate_hidden_out_blob = BnInOp2Blob("candidate_hidden_out");
  Blob* temp_data_blob = BnInOp2Blob("temp_data");
  Blob* plus_op_out_blob = BnInOp2Blob("plus_op_out");

  BasicGruKernelUtil<device_type, T>::ComputeGateForward(
      ctx, BnInOp2Blob("in"), hidden_blob, BnInOp2Blob("bias_multiplier_r"),
      BnInOp2Blob("i2h_weight_r"), BnInOp2Blob("h2h_weight_r"),
      BnInOp2Blob("bias_r"), reset_gate_data_blob, reset_gate_out_blob);

  BasicGruKernelUtil<device_type, T>::ComputeGateForward(
      ctx, BnInOp2Blob("in"), hidden_blob, BnInOp2Blob("bias_multiplier_z"),
      BnInOp2Blob("i2h_weight_z"), BnInOp2Blob("h2h_weight_z"),
      BnInOp2Blob("bias_z"), update_gate_data_blob, update_gate_out_blob);

  BasicGruKernelUtil<device_type, T>::ComputeCandidateHiddenForward(
      ctx, BnInOp2Blob("in"), hidden_blob, BnInOp2Blob("bias_multiplier"),
      BnInOp2Blob("i2h_weight"), BnInOp2Blob("h2h_weight"), BnInOp2Blob("bias"),
      candidate_hidden_data_blob, candidate_hidden_out_blob,
      reset_gate_out_blob, temp_data_blob);

  BasicGruKernelUtil<device_type, T>::ComputePlusOutForward(
      ctx, hidden_blob, candidate_hidden_out_blob, temp_data_blob,
      update_gate_out_blob, plus_op_out_blob);

  // out = plus_op_out
  BnInOp2Blob("out")->CopyDataContentFrom<device_type>(ctx.device_ctx,
                                                       plus_op_out_blob);
  // rec_out = plus_op_out
  BnInOp2Blob("rec_out")->CopyDataContentFrom<device_type>(ctx.device_ctx,
                                                           plus_op_out_blob);
}

template<DeviceType device_type, typename T>
void BasicGruKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* update_gate_data_blob = BnInOp2Blob("update_gate_data");
  const Blob* update_gate_out_blob = BnInOp2Blob("update_gate_out");
  Blob* update_gate_out_diff_blob = BnInOp2Blob("update_gate_out_diff");
  Blob* update_gate_data_diff_blob = BnInOp2Blob("update_gate_data_diff");
  const Blob* candidate_hidden_data_blob = BnInOp2Blob("candidate_hidden_data");
  const Blob* candidate_hidden_out_blob = BnInOp2Blob("candidate_hidden_out");
  Blob* candidate_hidden_data_diff_blob =
      BnInOp2Blob("candidate_hidden_data_diff");
  Blob* candidate_hidden_out_diff_blob =
      BnInOp2Blob("candidate_hidden_out_diff");
  Blob* temp_data_blob = BnInOp2Blob("temp_data");
  const Blob* reset_gate_data_blob = BnInOp2Blob("reset_gate_data");
  const Blob* reset_gate_out_blob = BnInOp2Blob("reset_gate_out");
  Blob* reset_gate_out_diff_blob = BnInOp2Blob("reset_gate_out_diff");
  Blob* reset_gate_data_diff_blob = BnInOp2Blob("reset_gate_data_diff");
  Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);
  Blob* hidden_diff_blob = this->GetHiddenBlob(BnInOp2Blob);
  // reuse memory
  Blob* update_gate_out_bran_diff_blob = BnInOp2Blob("update_mul_hidden");
  // plus_op_out_diff = out_diff
  Blob* plus_op_out_diff_blob = BnInOp2Blob("plus_op_out");
  KernelUtil<device_type, T>::Copy(ctx.device_ctx,
                                   plus_op_out_diff_blob->shape().elem_cnt(),
                                   BnInOp2Blob("out_diff")->dptr<T>(), 1,
                                   plus_op_out_diff_blob->mut_dptr<T>(), 1);

  if (BnInOp2Blob("in")->col_id() != BnInOp2Blob("in")->max_col_id()) {
    // plus_op_out_diff += rec_out_diff
    KernelUtil<device_type, T>::Axpy(
        ctx.device_ctx,
        static_cast<T>(plus_op_out_diff_blob->shape().elem_cnt()),
        static_cast<T>(1), BnInOp2Blob("rec_out_diff")->dptr<T>(),
        static_cast<T>(1), plus_op_out_diff_blob->mut_dptr<T>(),
        static_cast<T>(1));
  }

  BasicGruKernelUtil<device_type, T>::ComputeTmpModelDiff(
      ctx, update_gate_out_blob, update_gate_data_blob,
      candidate_hidden_out_blob, candidate_hidden_data_blob,
      reset_gate_out_blob, reset_gate_data_blob, plus_op_out_diff_blob,
      hidden_blob, update_gate_out_diff_blob, update_gate_out_bran_diff_blob,
      update_gate_data_diff_blob, candidate_hidden_out_diff_blob,
      candidate_hidden_data_diff_blob, BnInOp2Blob("h2h_weight"),
      temp_data_blob, reset_gate_out_diff_blob, reset_gate_data_diff_blob);

  BasicGruKernelUtil<device_type, T>::ComputeWeightDiff(
      ctx, BnInOp2Blob("in"), hidden_blob, reset_gate_data_diff_blob,
      BnInOp2Blob("i2h_weight_r_diff"), BnInOp2Blob("h2h_weight_r_diff"));
  BasicGruKernelUtil<device_type, T>::ComputeWeightDiff(
      ctx, BnInOp2Blob("in"), hidden_blob, update_gate_data_diff_blob,
      BnInOp2Blob("i2h_weight_z_diff"), BnInOp2Blob("h2h_weight_z_diff"));
  BasicGruKernelUtil<device_type, T>::ComputeWeightDiff(
      ctx, BnInOp2Blob("in"), hidden_blob, candidate_hidden_data_diff_blob,
      BnInOp2Blob("i2h_weight_diff"), BnInOp2Blob("h2h_weight_diff"));
// bias_diff = bias_nultiplier * data_diff
#define OF_GRU_COMPUTE_BIAS_DIFF(model, tmpmodel, gatename)                   \
  if (BnInOp2Blob(#model) != nullptr) {                                       \
    KernelUtil<device_type, T>::BlobGemm(                                     \
        ctx.device_ctx, CblasTrans, CblasNoTrans, static_cast<T>(1),          \
        static_cast<T>(0), BnInOp2Blob(#tmpmodel), gatename##_data_diff_blob, \
        BnInOp2Blob(#model));                                                 \
  }
  OF_GRU_COMPUTE_BIAS_DIFF(bias_diff_r, bias_multiplier_r, reset_gate)
  OF_GRU_COMPUTE_BIAS_DIFF(bias_diff_z, bias_multiplier_z, update_gate)
  OF_GRU_COMPUTE_BIAS_DIFF(bias_diff, bias_multiplier, candidate_hidden)
#undef OF_GRU_COMPUTE_BIAS_DIFF

  if (BnInOp2Blob("in_diff") != nullptr) {
    // in_diff = reset_gate_data_diff * i2h_weght_r
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
        static_cast<T>(0), reset_gate_data_diff_blob,
        BnInOp2Blob("i2h_weight_r"), BnInOp2Blob("in_diff"));
    // in_diff += update_gate_data_diff * i2h_weght_z
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
        static_cast<T>(1), update_gate_data_diff_blob,
        BnInOp2Blob("i2h_weight_z"), BnInOp2Blob("in_diff"));
    // in_diff += candidate_hidden_data_diff * i2h_weght
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
        static_cast<T>(1), candidate_hidden_data_diff_blob,
        BnInOp2Blob("i2h_weight"), BnInOp2Blob("in_diff"));
  }

  if (BnInOp2Blob("in")->col_id() != 0 || this->NeedExternalH0()
      || this->op_conf().basic_gru_conf().is_init_hidden_trainable()) {
    // compute hidden_diff
    BasicGruKernelUtil<device_type, T>::ComputeHiddenDiff(
        ctx, BnInOp2Blob("h2h_weight_r"), BnInOp2Blob("h2h_weight_z"),
        BnInOp2Blob("h2h_weight"), reset_gate_out_blob, update_gate_out_blob,
        hidden_blob, hidden_diff_blob, candidate_hidden_data_diff_blob,
        reset_gate_data_diff_blob, update_gate_data_diff_blob,
        plus_op_out_diff_blob);
  }
}

template<DeviceType device_type, typename T>
void BasicGruKernel<device_type, T>::VirtualInitModelBlobsWithRandomSeed(
    const KernelCtx& ctx, std::mt19937 random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
#define OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(modelname) \
  KernelUtil<device_type, T>::InitializeWithProperConf(    \
      ctx.device_ctx,                                      \
      OF_PB_POINTER_GET(this->op_conf().basic_gru_conf(),  \
                        modelname##_initializer),          \
      random_seed_gen(), BnInOp2Blob(#modelname))
  OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(i2h_weight_r);
  OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(h2h_weight_r);
  OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(bias_r);
  OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(i2h_weight_z);
  OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(h2h_weight_z);
  OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(bias_z);
  OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(i2h_weight);
  OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(h2h_weight);
  OF_GRU_INIT_MODEL_BLOB_WITH_RANDOM_SEED(bias);
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
  OF_GRU_INIT_MODEL_BLOB_WITH_DIR(i2h_weight_r);
  OF_GRU_INIT_MODEL_BLOB_WITH_DIR(h2h_weight_r);
  OF_GRU_INIT_MODEL_BLOB_WITH_DIR(i2h_weight_z);
  OF_GRU_INIT_MODEL_BLOB_WITH_DIR(h2h_weight_z);
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
#define OF_GRU_INIT_PURE_MODEL_TMP_BLON(modelname)               \
  InitializerConf modelname##_fill_conf;                         \
  modelname##_fill_conf.mutable_constant_conf()->set_value(1.f); \
  KernelUtil<device_type, T>::Initialize(                        \
      ctx.device_ctx, modelname##_fill_conf, 0, BnInOp2Blob(#modelname))
  if (this->op_conf().basic_gru_conf().use_bias()) {
    OF_GRU_INIT_PURE_MODEL_TMP_BLON(bias_miltiplier_r);
    OF_GRU_INIT_PURE_MODEL_TMP_BLON(bias_miltiplier_z);
    OF_GRU_INIT_PURE_MODEL_TMP_BLON(bias_miltiplier);
  }
#undef OF_GRU_INIT_PURE_MODEL_TMP_BLON
}

template<DeviceType device_type, typename T>
void BasicGruKernelUtil<device_type, T>::ComputeGateForward(
    const KernelCtx& ctx, const Blob* in_data, const Blob* hidden,
    const Blob* bias_multiplier, const Blob* i2h_weight, const Blob* h2h_weight,
    const Blob* bias, Blob* gate_data, Blob* gate_out) {
  // gate_data = in * i2h_weight
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       in_data, i2h_weight, gate_data);
  // gate_data += hidden * h2h_weight
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(1),
                                       hidden, h2h_weight, gate_data);
  // gate_data += bias_multiplier * bias
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), bias_multiplier, bias, gate_data);
  // gate_out = sigmoid(gate_data)
  KernelUtil<device_type, T>::Sigmoid(
      ctx.device_ctx, gate_data->shape().elem_cnt(), gate_data->dptr<T>(),
      gate_out->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void BasicGruKernelUtil<device_type, T>::ComputeCandidateHiddenForward(
    const KernelCtx& ctx, const Blob* in_data, const Blob* hidden,
    const Blob* bias_multiplier, const Blob* i2h_weight, const Blob* h2h_weight,
    const Blob* bias, Blob* candidate_data, Blob* candidate_out,
    Blob* reset_out, Blob* temp_data) {
  // temp_data = hidden .*reset_out
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, static_cast<T>(reset_out->shape().elem_cnt()),
      reset_out->dptr<T>(), hidden->dptr<T>(), candidate_data->mut_dptr<T>());
  // candidate_data = temp_data * h2h_weight
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       temp_data, h2h_weight, candidate_data);
  // candidate_data += in * i2h_weight
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(1),
                                       in_data, i2h_weight, candidate_data);
  // candidate_data += bias_multiplier * bias
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), bias_multiplier, bias, candidate_data);
  // candidate_out = tanh(candidate_data)
  KernelUtil<device_type, T>::TanH(
      ctx.device_ctx, candidate_data->shape().elem_cnt(),
      candidate_data->dptr<T>(), candidate_out->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void BasicGruKernelUtil<device_type, T>::ComputePlusOutForward(
    const KernelCtx& ctx, const Blob* hidden, Blob* candidate_out,
    Blob* temp_data, Blob* update_out, Blob* plus_out) {
  // plus_out = candidate_out .* update_out
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, static_cast<T>(update_out->shape().elem_cnt()),
      candidate_out->dptr<T>(), update_out->dptr<T>(), plus_out->mut_dptr<T>());
  // temp_data = hidden .* update_out
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, static_cast<T>(update_out->shape().elem_cnt()),
      hidden->dptr<T>(), update_out->dptr<T>(), temp_data->mut_dptr<T>());
  // plus_out -= temp_data
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, static_cast<T>(temp_data->shape().elem_cnt()),
      static_cast<T>(-1), temp_data->dptr<T>(), static_cast<T>(1),
      plus_out->mut_dptr<T>(), static_cast<T>(1));
  // plus_out += hidden
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, static_cast<T>(hidden->shape().elem_cnt()),
      static_cast<T>(1), hidden->dptr<T>(), static_cast<T>(1),
      plus_out->mut_dptr<T>(), static_cast<T>(1));
}

template<DeviceType device_type, typename T>
void BasicGruKernelUtil<device_type, T>::ComputeWeightDiff(
    const KernelCtx& ctx, const Blob* in_data, Blob* hidden, Blob* out_diff,
    Blob* i2h_weight_diff, Blob* h2h_weight_diff) {
  // h2h_weght_diff = out_diff * hidden
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       out_diff, hidden, h2h_weight_diff);
  // i2h_weght_diff = out_diff * in
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       out_diff, in_data, i2h_weight_diff);
}

template<DeviceType device_type, typename T>
void BasicGruKernelUtil<device_type, T>::ComputeTmpModelDiff(
    const KernelCtx& ctx, const Blob* update_out, const Blob* update_data,
    const Blob* candidate_out, const Blob* candidate_data,
    const Blob* reset_out, const Blob* reset_data, Blob* plus_out_diff,
    Blob* hidden, Blob* update_o_diff, Blob* update_o_bran_diff,
    Blob* update_d_diff, Blob* candidate_o_diff, Blob* candidate_d_diff,
    const Blob* h2h_weight, Blob* temp_data, Blob* reset_o_diff,
    Blob* reset_d_diff) {
  // update_o_diff = candidate_out .* plus_out_diff
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, update_out->shape().elem_cnt(), candidate_out->dptr<T>(),
      plus_out_diff->dptr<T>(), update_o_diff->mut_dptr<T>());
  // update_o_bran_diff = hidden .* plus_out_diff
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, hidden->shape().elem_cnt(),
                                  hidden->dptr<T>(), plus_out_diff->dptr<T>(),
                                  update_o_bran_diff->mut_dptr<T>());
  // update_o_diff -= update_o_bran_diff
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, static_cast<T>(update_o_bran_diff->shape().elem_cnt()),
      static_cast<T>(-1), update_o_bran_diff->dptr<T>(), static_cast<T>(1),
      update_o_diff->mut_dptr<T>(), static_cast<T>(1));
  // update_d_diff = update_out * (1 - update_out) *
  // update_o_diff
  KernelUtil<device_type, T>::SigmoidBackward(
      ctx.device_ctx, update_out->shape().elem_cnt(), update_data->dptr<T>(),
      update_out->dptr<T>(), update_o_diff->dptr<T>(),
      update_d_diff->mut_dptr<T>());
  // candidate_o_diff = update_out .* plus_out_diff
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, update_out->shape().elem_cnt(), update_out->dptr<T>(),
      plus_out_diff->dptr<T>(), candidate_o_diff->mut_dptr<T>());
  // candidate_d_diff = (1 - candidate_out^2) .*
  // candidate_o_diff
  KernelUtil<device_type, T>::TanHBackward(
      ctx.device_ctx, candidate_out->shape().elem_cnt(),
      candidate_data->dptr<T>(), candidate_out->dptr<T>(),
      candidate_o_diff->dptr<T>(), candidate_d_diff->mut_dptr<T>());
  // temp_data = candidate_d_diff * h2h_weight
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(0), candidate_d_diff, h2h_weight, temp_data);
  // reset_o_diff = hidden .* temp_data
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, hidden->shape().elem_cnt(),
                                  hidden->dptr<T>(), temp_data->dptr<T>(),
                                  reset_o_diff->mut_dptr<T>());
  // reset_d_diff
  KernelUtil<device_type, T>::SigmoidBackward(
      ctx.device_ctx, reset_out->shape().elem_cnt(), reset_data->dptr<T>(),
      reset_out->dptr<T>(), reset_o_diff->dptr<T>(),
      reset_d_diff->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void BasicGruKernelUtil<device_type, T>::ComputeHiddenDiff(
    const KernelCtx& ctx, const Blob* h2h_weight_r, const Blob* h2h_weight_z,
    const Blob* h2h_weight, const Blob* reset_out, const Blob* update_out,
    Blob* hidden, Blob* hidden_diff, Blob* candidate_d_diff, Blob* reset_d_diff,
    Blob* update_d_diff, Blob* plus_out_diff) {
  // compute hidden_diff
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(0), candidate_d_diff, h2h_weight, hidden_diff);
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, hidden_diff->shape().elem_cnt(), hidden_diff->dptr<T>(),
      reset_out->dptr<T>(), hidden_diff->mut_dptr<T>());
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), reset_d_diff, h2h_weight_r, hidden_diff);
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), update_d_diff, h2h_weight_z, hidden_diff);
  // reuse hidden_blob
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, static_cast<T>(hidden->shape().elem_cnt()),
      update_out->dptr<T>(), plus_out_diff->dptr<T>(), hidden->mut_dptr<T>());
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, static_cast<T>(hidden->shape().elem_cnt()),
      static_cast<T>(-1), plus_out_diff->dptr<T>(), static_cast<T>(1),
      hidden->mut_dptr<T>(), static_cast<T>(1));
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, static_cast<T>(hidden->shape().elem_cnt()),
      static_cast<T>(-1), hidden->dptr<T>(), static_cast<T>(1),
      hidden_diff->mut_dptr<T>(), static_cast<T>(1));
}
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBasicGruConf, BasicGruKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
