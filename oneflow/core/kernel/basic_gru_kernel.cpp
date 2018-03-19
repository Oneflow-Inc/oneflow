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
  Blob* plus_op_out_blob = BnInOp2Blob("plus_op_out");
  Blob* reset_gate_data_blob = BnInOp2Blob("reset_gate_data");
  Blob* update_gate_data_blob = BnInOp2Blob("update_gate_data");
  Blob* candidate_hidden_data_blob = BnInOp2Blob("candidate_hidden_data");
  Blob* candidate_hidden_out_blob = BnInOp2Blob("candidate_hidden_out");
  Blob* reset_gate_out_blob = BnInOp2Blob("reset_gate_out");
  Blob* update_gate_out_blob = BnInOp2Blob("update_gate_out");
  Blob* update_mul_hidden_blob = BnInOp2Blob("update_mul_hidden");

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
      reset_gate_out_blob);

  BasicGruKernelUtil<device_type, T>::ComputePlusOutForward(
      ctx, hidden_blob, candidate_hidden_out_blob, update_mul_hidden_blob,
      update_gate_out_blob, plus_op_out_blob);

  // out = plus_op_out
  BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, plus_op_out_blob);
  // rec_out = plus_op_out
  BnInOp2Blob("rec_out")->CopyDataContentFrom(ctx.device_ctx, plus_op_out_blob);
}

template<DeviceType device_type, typename T>
void BasicGruKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);
  Blob* hidden_diff_blob = this->GetHiddenBlob(BnInOp2Blob);
  const Blob* rec_out_diff_blob = BnInOp2Blob("rec_out_diff");
  const Blob* update_gate_data_blob = BnInOp2Blob("update_gate_data");
  const Blob* update_gate_out_blob = BnInOp2Blob("update_gate_out");
  Blob* update_gate_out_diff_blob = BnInOp2Blob("update_gate_out_diff");
  Blob* update_gate_data_diff_blob = BnInOp2Blob("update_gate_data_diff");
  const Blob* reset_gate_data_blob = BnInOp2Blob("reset_gate_data");
  const Blob* reset_gate_out_blob = BnInOp2Blob("reset_gate_out");
  Blob* reset_gate_out_diff_blob = BnInOp2Blob("reset_gate_out_diff");
  Blob* reset_gate_data_diff_blob = BnInOp2Blob("reset_gate_data_diff");
  const Blob* candidate_hidden_data_blob = BnInOp2Blob("candidate_hidden_data");
  const Blob* candidate_hidden_out_blob = BnInOp2Blob("candidate_hidden_out");
  Blob* candidate_hidden_data_diff_blob =
      BnInOp2Blob("candidate_hidden_data_diff");

  Blob* candidate_hidden_out_diff_blob =
      BnInOp2Blob("candidate_hidden_out_diff");
  // reuse memory
  Blob* update_gate_out_a_diff_blob = BnInOp2Blob("update_mul_hidden_diff");

  // plus_op_out_diff = out_diff
  BnInOp2Blob("plus_op_out_diff")
      ->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("out_diff"));

  Blob* plus_op_out_diff_blob = BnInOp2Blob("plus_op_out_diff");
  if (in_blob->col_id() != in_blob->max_col_id()) {
    // plus_op_out_diff += rec_out_diff
    KernelUtil<device_type, T>::Axpy(
        ctx.device_ctx,
        static_cast<T>(plus_op_out_diff_blob->shape().elem_cnt()),
        static_cast<T>(1), rec_out_diff_blob->dptr<T>(), static_cast<T>(1),
        plus_op_out_diff_blob->mut_dptr<T>(), static_cast<T>(1));
  }

  // candidate_hidden_out_diff = update_gate_our .* plus_op_out_diff
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, update_gate_out_blob->shape().elem_cnt(),
      update_gate_out_blob->dptr<T>(), plus_op_out_diff_blob->dptr<T>(),
      candidate_hidden_out_diff_blob->mut_dptr<T>());
  // candidate_hidden_data_diff = (1 - candidate_hidden_out^2) .*
  // candidate_hidden_out_diff
  KernelUtil<device_type, T>::TanHBackward(
      ctx.device_ctx, candidate_hidden_out_blob->shape().elem_cnt(),
      candidate_hidden_data_blob->dptr<T>(),
      candidate_hidden_out_blob->dptr<T>(),
      candidate_hidden_out_diff_blob->dptr<T>(),
      candidate_hidden_data_diff_blob->mut_dptr<T>());
  // update_gate_out_diff = candidate_hidden_out .* plus_op_out_diff
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, update_gate_out_blob->shape().elem_cnt(),
      candidate_hidden_out_blob->dptr<T>(), plus_op_out_diff_blob->dptr<T>(),
      update_gate_out_diff_blob->mut_dptr<T>());
  // update_gate_out_a_diff = hidden .* plus_op_out_diff
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, hidden_blob->shape().elem_cnt(), hidden_blob->dptr<T>(),
      plus_op_out_diff_blob->dptr<T>(),
      update_gate_out_a_diff_blob->mut_dptr<T>());
  // update_gate_out_diff += update_gate_out_a
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx,
      static_cast<T>(update_gate_out_a_diff_blob->shape().elem_cnt()),
      static_cast<T>(1), update_gate_out_a_diff_blob->dptr<T>(),
      static_cast<T>(1), update_gate_out_diff_blob->mut_dptr<T>(),
      static_cast<T>(1));
  // update_gate_data_diff = update_gate_out * (1 - update_gate_put) *
  // update_gate_out_diff
  KernelUtil<device_type, T>::SigmoidBackward(
      ctx.device_ctx, update_gate_out_blob->shape().elem_cnt(),
      update_gate_data_blob->dptr<T>(), update_gate_out_blob->dptr<T>(),
      update_gate_out_diff_blob->dptr<T>(),
      update_gate_data_diff_blob->mut_dptr<T>());

  // reset_gate_out_diff = hidden .* candidate_gate_data_diff
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, hidden_blob->shape().elem_cnt(), hidden_blob->dptr<T>(),
      candidate_hidden_data_diff_blob->dptr<T>(),
      reset_gate_out_diff_blob->mut_dptr<T>());
  // reset_gate_data_diff
  KernelUtil<device_type, T>::SigmoidBackward(
      ctx.device_ctx, reset_gate_out_blob->shape().elem_cnt(),
      reset_gate_data_blob->dptr<T>(), reset_gate_out_blob->dptr<T>(),
      reset_gate_out_diff_blob->dptr<T>(),
      reset_gate_data_diff_blob->mut_dptr<T>());

  // h2h_weght_r_diff = reset_gate_data_diff * hidden
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       reset_gate_data_diff_blob, hidden_blob,
                                       BnInOp2Blob("h2h_weight_r"));

  // i2h_weght_r_diff = reset_gate_data_diff * in
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(0), reset_gate_data_diff_blob, BnInOp2Blob("in"),
      BnInOp2Blob("i2h_weight_r"));

  // h2h_weght_z_diff = update_gate_data_diff * hidden
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       update_gate_data_diff_blob, hidden_blob,
                                       BnInOp2Blob("h2h_weight_z"));

  // i2h_weght_z_diff = reset_gate_data_diff * in
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(0), update_gate_data_diff_blob, BnInOp2Blob("in"),
      BnInOp2Blob("i2h_weight_z"));

  // h2h_weght_diff = candidate_hidden_data_diff * hidden
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       reset_gate_data_diff_blob, hidden_blob,
                                       BnInOp2Blob("h2h_weight_r"));

  // i2h_weght_diff = candidate_hidden_data_diff * in
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(0), candidate_hidden_data_diff_blob, BnInOp2Blob("in"),
      BnInOp2Blob("i2h_weight_r"));
  if (BnInOp2Blob("bias_diff_r") != nullptr) {
    // bias_diff_r = bias_nultiplier_r * reset_gate_data_diff
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasTrans, CblasNoTrans, static_cast<T>(1),
        static_cast<T>(0), BnInOp2Blob("bias_multiplier_r"),
        reset_gate_data_diff_blob, BnInOp2Blob("bias_diff_r"));
  }
  if (BnInOp2Blob("bias_diff_z") != nullptr) {
    // bias_diff_z = bias_nultiplier_z * update_gate_data_diff
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasTrans, CblasNoTrans, static_cast<T>(1),
        static_cast<T>(0), BnInOp2Blob("bias_multiplier_z"),
        reset_gate_data_diff_blob, BnInOp2Blob("bias_diff_z"));
  }
  if (BnInOp2Blob("bias_diff") != nullptr) {
    // bias_diff = bias_nultiplier * candidate_hidden_data_diff
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasTrans, CblasNoTrans, static_cast<T>(1),
        static_cast<T>(0), BnInOp2Blob("bias_multiplier"),
        candidate_hidden_data_diff_blob, BnInOp2Blob("bias_diff"));
  }

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
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
        static_cast<T>(0), candidate_hidden_data_diff_blob,
        BnInOp2Blob("h2h_weight"), hidden_diff_blob);
    KernelUtil<device_type, T>::Mul(
        ctx.device_ctx, hidden_diff_blob->shape().elem_cnt(),
        hidden_diff_blob->dptr<T>(), reset_gate_out_blob->dptr<T>(),
        hidden_diff_blob->mut_dptr<T>());
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
        static_cast<T>(1), reset_gate_data_diff_blob,
        BnInOp2Blob("h2h_weight_r"), hidden_diff_blob);
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
        static_cast<T>(1), update_gate_data_diff_blob,
        BnInOp2Blob("h2h_weight_z"), hidden_diff_blob);
    // reuse hidden_blob
    KernelUtil<device_type, T>::Mul(
        ctx.device_ctx, static_cast<T>(hidden_blob->shape().elem_cnt()),
        update_gate_out_blob->dptr<T>(), plus_op_out_diff_blob->dptr<T>(),
        hidden_blob->mut_dptr<T>());
    KernelUtil<device_type, T>::Axpy(
        ctx.device_ctx, static_cast<T>(hidden_blob->shape().elem_cnt()),
        static_cast<T>(-1), plus_op_out_diff_blob->dptr<T>(), static_cast<T>(1),
        hidden_blob->mut_dptr<T>(), static_cast<T>(1));
    KernelUtil<device_type, T>::Axpy(
        ctx.device_ctx, static_cast<T>(hidden_blob->shape().elem_cnt()),
        static_cast<T>(-1), hidden_blob->dptr<T>(), static_cast<T>(1),
        hidden_diff_blob->mut_dptr<T>(), static_cast<T>(1));
  }
}

template<DeviceType device_type, typename T>
void BasicGruKernel<device_type, T>::InitPureModelTmpBlobs(
    DeviceCtx* ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitializerConf bias_multiplier_r_fill_conf;
  bias_multiplier_r_fill_conf.mutable_constant_conf()->set_value(1.f);
  KernelUtil<device_type, T>::Initialize(ctx, bias_multiplier_r_fill_conf, 0,
                                         BnInOp2Blob("bias_miltiplier_r"));
  InitializerConf bias_multiplier_z_fill_conf;
  bias_multiplier_z_fill_conf.mutable_constant_conf()->set_value(1.f);
  KernelUtil<device_type, T>::Initialize(ctx, bias_multiplier_z_fill_conf, 0,
                                         BnInOp2Blob("bias_miltiplier_z"));
  InitializerConf bias_multiplier_fill_conf;
  bias_multiplier_fill_conf.mutable_constant_conf()->set_value(1.f);
  KernelUtil<device_type, T>::Initialize(ctx, bias_multiplier_fill_conf, 0,
                                         BnInOp2Blob("bias_miltiplier"));
}

template<DeviceType device_type, typename T>
void BasicGruKernel<device_type, T>::VirtualInitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_gru_conf(),
                        i2h_weight_r_initializer),
      (*random_seed_gen)(), BnInOp2Blob("i2h_weiht_r"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_gru_conf(),
                        h2h_weight_r_initializer),
      (*random_seed_gen)(), BnInOp2Blob("h2h_weiht_r"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_gru_conf(), bias_r_initializer),
      (*random_seed_gen)(), BnInOp2Blob("bias_r"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_gru_conf(),
                        i2h_weight_z_initializer),
      (*random_seed_gen)(), BnInOp2Blob("i2h_weiht_z"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_gru_conf(),
                        h2h_weight_z_initializer),
      (*random_seed_gen)(), BnInOp2Blob("h2h_weiht_z"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_gru_conf(), bias_z_initializer),
      (*random_seed_gen)(), BnInOp2Blob("bias_z"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_gru_conf(),
                        i2h_weight_initializer),
      (*random_seed_gen)(), BnInOp2Blob("i2h_weiht"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_gru_conf(),
                        h2h_weight_initializer),
      (*random_seed_gen)(), BnInOp2Blob("h2h_weiht"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_gru_conf(), bias_initializer),
      (*random_seed_gen)(), BnInOp2Blob("bias"));
}

template<DeviceType device_type, typename T>
void BasicGruKernel<device_type, T>::VirtualInitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* i2h_weight_r_blob = BnInOp2Blob("i2h_weigth_r");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, i2h_weight_r_blob, "i2h_weight_r",
      i2h_weight_r_blob->shape().At(0), i2h_weight_r_blob->shape().Count(1));
  Blob* h2h_weight_r_blob = BnInOp2Blob("h2h_weigth_r");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, h2h_weight_r_blob, "h2h_weight_r",
      h2h_weight_r_blob->shape().At(0), h2h_weight_r_blob->shape().Count(1));
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias_r"), "bias_r",
      BnInOp2Blob("bias_r")->shape().At(0), 1);

  Blob* i2h_weight_z_blob = BnInOp2Blob("i2h_weigth_z");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, i2h_weight_z_blob, "i2h_weight_z",
      i2h_weight_z_blob->shape().At(0), i2h_weight_z_blob->shape().Count(1));
  Blob* h2h_weight_z_blob = BnInOp2Blob("h2h_weigth_z");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, h2h_weight_z_blob, "h2h_weight_z",
      h2h_weight_z_blob->shape().At(0), h2h_weight_z_blob->shape().Count(1));
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias_z"), "bias_z",
      BnInOp2Blob("bias_z")->shape().At(0), 1);

  Blob* i2h_weight_blob = BnInOp2Blob("i2h_weigth");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, i2h_weight_blob, "i2h_weight",
      i2h_weight_blob->shape().At(0), i2h_weight_blob->shape().Count(1));
  Blob* h2h_weight_blob = BnInOp2Blob("h2h_weigth");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, h2h_weight_blob, "h2h_weight",
      h2h_weight_blob->shape().At(0), h2h_weight_blob->shape().Count(1));
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias"), "bias",
      BnInOp2Blob("bias")->shape().At(0), 1);
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
    Blob* reset_out) {
  // candidate_hidden_data = hidden .*reset_gate_out
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, static_cast<T>(reset_out->shape().elem_cnt()),
      reset_out->dptr<T>(), hidden->dptr<T>(), candidate_data->mut_dptr<T>());
  // candidate_hidden_data *= h2h_weight
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, static_cast<T>(1),
      static_cast<T>(0), candidate_data, h2h_weight, candidate_data);

  // candidate_hidden_data += in * i2h_weight
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(1),
                                       in_data, i2h_weight, candidate_data);

  // candidate_hidden_data += bias_multiplier * bias
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), bias_multiplier, bias, candidate_data);

  // candidate_hidden_out = tanh(candidate_hidden_data)
  KernelUtil<device_type, T>::TanH(
      ctx.device_ctx, candidate_data->shape().elem_cnt(),
      candidate_data->dptr<T>(), candidate_out->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void BasicGruKernelUtil<device_type, T>::ComputePlusOutForward(
    const KernelCtx& ctx, const Blob* hidden, Blob* candidate_out,
    Blob* temp_data, Blob* update_out, Blob* plus_out) {
  // plus_op_out = candidate_hidden_out .* update_gate_out
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, static_cast<T>(update_out->shape().elem_cnt()),
      candidate_out->dptr<T>(), update_out->dptr<T>(), plus_out->mut_dptr<T>());
  // update_mul_hidden = hidden .* update_gate_out
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, static_cast<T>(update_out->shape().elem_cnt()),
      hidden->dptr<T>(), update_out->dptr<T>(), temp_data->mut_dptr<T>());

  // plus_op_out -= update_mul_hidden
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, static_cast<T>(temp_data->shape().elem_cnt()),
      static_cast<T>(-1), temp_data->dptr<T>(), static_cast<T>(1),
      plus_out->mut_dptr<T>(), static_cast<T>(1));
  // plus_op_out += hidden
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, static_cast<T>(hidden->shape().elem_cnt()),
      static_cast<T>(1), hidden->dptr<T>(), static_cast<T>(1),
      plus_out->mut_dptr<T>(), static_cast<T>(1));
}
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBasicGruConf, BasicGruKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
