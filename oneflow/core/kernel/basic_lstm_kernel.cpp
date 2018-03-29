#include "oneflow/core/kernel/basic_lstm_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::VirtualKernelInit(
    const ParallelContext* parallel_ctx) {
  auto& input_bns = this->kernel_conf().input_bns();
  need_external_h0_ =
      std::find(input_bns.begin(), input_bns.end(), "h0") != input_bns.end();
  need_external_c0_ =
      std::find(input_bns.begin(), input_bns.end(), "c0") != input_bns.end();

  ActivationType activation_type =
      this->op_conf().basic_lstm_conf().activation();
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
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
const PbMessage& BasicLstmKernel<device_type, T>::GetBasicLstmOpConf() const {
  return this->op_conf().basic_lstm_conf();
}

template<DeviceType device_type, typename T>
bool BasicLstmKernel<device_type, T>::HasInitHiddenInitializer() const {
  return this->op_conf().basic_lstm_conf().has_init_hidden_initializer();
}

template<DeviceType device_type, typename T>
bool BasicLstmKernel<device_type, T>::HasInitCellInitializer() const {
  return this->op.conf().basic_lstm_conf().has_init_cell_initializer();
}

template<DeviceType device_type, typename T>
bool BasicLstmKernel<device_type, T>::NeedExternalH0() const {
  return need_external_h0_;
}

template<DeviceType device_type, typename T>
bool BasicLstmKernel<device_type, T>::NeedExternalC0() const {
  return need_external_c0_;
}

template<DeviceType device_type, typename T>
Blob* BasicLstmKernel<device_type, T>::GetHiddenBlob(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (BnInOp2Blob("in")->col_id() == 0) { return BnInOp2Blob("h0"); }
  return BnInOp2Blob("rec_in");
}

template<DeviceType device_type, typename T>
Blob* BasicLstmKernel<device_type, T>::GetCellBlob(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (BnInOp2Blob("in")->col_id() == 0) { return BnInOp2Blob("c0"); }
  return BnInOp2Blob("cell_in");
}

template<DeviceType device_type, typename T>
Blob* BasicLstmKernel<device_type, T>::GetHiddenDiffBlob(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (BnInOp2Blob("in")->col_id() == 0) { return BnInOp2Blob("h0_diff"); }
  return BnInOp2Blob("rec_in_diff");
}

template<DeviceType device_type, typename T>
Blob* BasicLstmKernel<device_type, T>::GetCellDiffBlob(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (BnInOp2Blob("in")->col_id() == 0) { return BnInOp2Blob("c0_diff"); }
  return BnInOp2Blob("cell_in_diff");
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::ForwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDataIdFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::ForwardColNum(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyColNumFrom(ctx.device_ctx, BnInOp2Blob("in"));
  BnInOp2Blob("rec_out")->CopyColNumFrom(ctx.device_ctx, BnInOp2Blob("in"));
  BnInOp2Blob("cell_out")->CopyColNumFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::BackwardColNum(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("in_diff")->CopyColNumFrom(ctx.device_ctx,
                                         BnInOp2Blob("out_diff"));
  BnInOp2Blob("rec_in_diff")
      ->CopyColNumFrom(ctx.device_ctx, BnInOp2Blob("out_diff"));
  BnInOp2Blob("cell_in_diff")
      ->CopyColNumFrom(ctx.device_ctx, BnInOp2Blob("out_diff"));
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);
  Blob* state_data_blob = BnInOp2Blob("state_data");
  Blob* candidate_out_blob = BnInOp2Blob("candidate_out");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* cell_in_blob = BnInOp2Blob("cell_in");
  Blob* cell_out_blob = BnInOp2Blob("cell_out");
  Blob* f_out_blob = BnInOp2Blob("f_out");
  Blob* i_out_blob = BnInOp2Blob("i_out");
  Blob* c_out_blob = BnInOp2Blob("c_out");
  Blob* o_out_blob = BnInOp2Blob("o_out");

  BasicLstmKernelUtil<device_type, T>::ComputeForwardGateOut(
      ctx, state_data_blob, BnInOp2Blob("i2h_f_weight"), hidden_blob,
      BnInOp2Blob("h2h_f_weight"), BnInOp2Blob("in"),
      BnInOp2Blob("bias_f_multiplier"), BnInOp2Blob("bias_f"));
  KernelUtil<device_type, T>::Sigmoid(
      ctx.device_ctx, out_blob->shape().elem_cnt(), state_data_blob->dptr<T>(),
      f_out_blob->mut_dptr<T>());

  BasicLstmKernelUtil<device_type, T>::ComputeForwardGateOut(
      ctx, state_data_blob, BnInOp2Blob("i2h_i_weight"), hidden_blob,
      BnInOp2Blob("h2h_i_weight"), BnInOp2Blob("in"),
      BnInOp2Blob("bias_i_multiplier"), BnInOp2Blob("bias_i"));
  KernelUtil<device_type, T>::Sigmoid(
      ctx.device_ctx, out_blob->shape().elem_cnt(), state_data_blob->dptr<T>(),
      i_out_blob->mut_dptr<T>());

  BasicLstmKernelUtil<device_type, T>::ComputeForwardGateOut(
      ctx, state_data_blob, BnInOp2Blob("i2h_o_weight"), hidden_blob,
      BnInOp2Blob("h2h_o_weight"), BnInOp2Blob("in"),
      BnInOp2Blob("bias_o_multiplier"), BnInOp2Blob("bias_o"));
  KernelUtil<device_type, T>::Sigmoid(
      ctx.device_ctx, out_blob->shape().elem_cnt(), state_data_blob->dptr<T>(),
      o_out_blob->mut_dptr<T>());

  BasicLstmKernelUtil<device_type, T>::ComputeForwardGateOut(
      ctx, state_data_blob, BnInOp2Blob("i2h_c_weight"), hidden_blob,
      BnInOp2Blob("h2h_c_weight"), BnInOp2Blob("in"),
      BnInOp2Blob("bias_c_multiplier"), BnInOp2Blob("bias_c"));
  (*activation_fw_func_)(ctx.device_ctx, out_blob->shape().elem_cnt(),
                         state_data_blob->dptr<T>(), c_out_blob->mut_dptr<T>());

  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, out_blob->shape().elem_cnt(), f_out_blob->dptr<T>(),
      cell_in_blob->dptr<T>(), cell_out_blob->mut_dptr<T>());
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, out_blob->shape().elem_cnt(), i_out_blob->dptr<T>(),
      cell_out_blob->dptr<T>(), candidate_out_blob->mut_dptr<T>());
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, out_blob->shape().elem_cnt(), static_cast<T>(1),
      candidate_out_blob->dptr<T>(), static_cast<T>(1),
      cell_out_blob->mut_dptr<T>(), static_cast<T>(1));

  (*activation_fw_func_)(ctx.device_ctx, out_blob->shape().elem_cnt(),
                         cell_out_blob->dptr<T>(),
                         candidate_out_blob->mut_dptr<T>());
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, out_blob->shape().elem_cnt(), o_out_blob->dptr<T>(),
      candidate_out_blob->dptr<T>(), out_blob->mut_dptr<T>());
  // rec_out = out
  BnInOp2Blob("rec_out")->CopyDataContentFrom(ctx.device_ctx, out_blob);
}  // namespace oneflow

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_blob = BnInOp2Blob("out");
  Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);

  const Blob* rec_out_diff_blob = BnInOp2Blob("rec_out_diff");
  const Blob* f_out_blob = BnInOp2Blob("f_out");
  Blob* f_data_diff_blob = BnInOp2Blob("f_data_diff");
  Blob* f_out_diff_blob = BnInOp2Blob("f_out_diff");

  const Blob* i_out_blob = BnInOp2Blob("i_out");
  Blob* i_data_diff_blob = BnInOp2Blob("i_data_diff");
  Blob* i_out_diff_blob = BnInOp2Blob("i_out_diff");

  const Blob* c_out_blob = BnInOp2Blob("c_out");
  Blob* c_data_diff_blob = BnInOp2Blob("c_data_diff");
  Blob* c_out_diff_blob = BnInOp2Blob("c_out_diff");

  Blob* o_out_blob = BnInOp2Blob("o_out");
  Blob* o_data_diff_blob = BnInOp2Blob("o_data_diff");
  Blob* o_out_diff_blob = BnInOp2Blob("o_out_diff");

  const Blob* cell_in_blob = BnInOp2Blob("cell_in");
  Blob* cell_out_blob = BnInOp2Blob("cell_out");
  Blob* cell_out_diff_blob = BnInOp2Blob("cell_out_diff");

  Blob* hidden_diff_blob = this->GetHiddenDiffBlob(BnInOp2Blob);
  Blob* candidate_out_blob = BnInOp2Blob("candidate_out");
  Blob* state_data_blob = BnInOp2Blob("state_data");
  if (in_blob->col_id() != in_blob->max_col_id()) {
    // cell_out_diff = (rec_out_diff + out_diff) * o_out * [1 -
    // tanh(cell_out) * tanh(cell_out)]	+ cell_out_diff
    BasicLstmKernelUtil<device_type, T>::ComputeBackwardCellOutDiff(
        ctx, rec_out_diff_blob, candidate_out_blob, cell_out_blob,
        cell_out_diff_blob, o_out_blob, out_diff_blob, activation_bw_func_);
  } else {
    cell_out_diff_blob->CopyDataContentFrom(ctx.device_ctx, out_diff_blob);
  }
  // f_out_diff = cell_out_diff * cell_in
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                  cell_out_diff_blob->dptr<T>(),
                                  cell_in_blob->dptr<T>(),
                                  f_out_diff_blob->mut_dptr<T>());
  //	f_data_diff = ComputeSigmoidDiff(f_out_diff)
  KernelUtil<device_type, T>::SigmoidBackward(
      ctx.device_ctx, out_blob->shape().elem_cnt(), state_data_blob->dptr<T>(),
      f_out_blob->dptr<T>(), f_out_diff_blob->dptr<T>(),
      f_data_diff_blob->mut_dptr<T>());
  // f_gate weight diff
  BasicLstmKernelUtil<device_type, T>::ComputeBackwardWeightDiff(
      ctx, BnInOp2Blob("in"), hidden_blob, f_out_diff_blob,
      BnInOp2Blob("h2h_f_weight_diff"), BnInOp2Blob("i2h_f_weight_diff"));

  // i_out_diff = cell_out_diff * c_out
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                  cell_out_diff_blob->dptr<T>(),
                                  c_out_blob->dptr<T>(),
                                  i_out_diff_blob->mut_dptr<T>());
  //	i_data_diff = ComputeSigmoidDiff(i_out_diff)
  KernelUtil<device_type, T>::SigmoidBackward(
      ctx.device_ctx, out_blob->shape().elem_cnt(), state_data_blob->dptr<T>(),
      i_out_blob->dptr<T>(), i_out_diff_blob->dptr<T>(),
      i_data_diff_blob->mut_dptr<T>());
  //  i_gate weight_diff
  BasicLstmKernelUtil<device_type, T>::ComputeBackwardWeightDiff(
      ctx, BnInOp2Blob("in"), hidden_blob, i_out_diff_blob,
      BnInOp2Blob("h2h_i_weight_diff"), BnInOp2Blob("i2h_i_weight_diff"));

  //  c_out_diff = cell_out_diff * i_out
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                  cell_out_diff_blob->dptr<T>(),
                                  i_out_blob->dptr<T>(),
                                  c_out_diff_blob->mut_dptr<T>());
  //  c_data_diff = ComputeTanHBackward(c_out_diff)
  (*activation_bw_func_)(ctx.device_ctx, out_blob->shape().elem_cnt(),
                         state_data_blob->dptr<T>(), c_out_blob->dptr<T>(),
                         c_out_diff_blob->dptr<T>(),
                         c_data_diff_blob->mut_dptr<T>());
  //  c_out weight diff
  BasicLstmKernelUtil<device_type, T>::ComputeBackwardWeightDiff(
      ctx, BnInOp2Blob("in"), hidden_blob, c_data_diff_blob,
      BnInOp2Blob("h2h_c_weight_diff"), BnInOp2Blob("i2h_c_weight_diff"));

  // o_out_diff = rec_out_diff * tanh(cell_out)
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                  rec_out_diff_blob->dptr<T>(),
                                  candidate_out_blob->dptr<T>(),
                                  o_out_diff_blob->mut_dptr<T>());
  //	o_data_diff = ComputeSigmoidBackward(o_out_diff)
  KernelUtil<device_type, T>::SigmoidBackward(
      ctx.device_ctx, out_blob->shape().elem_cnt(), state_data_blob->dptr<T>(),
      o_out_blob->dptr<T>(), o_out_diff_blob->dptr<T>(),
      o_data_diff_blob->mut_dptr<T>());
  // o_gate weight diff
  BasicLstmKernelUtil<device_type, T>::ComputeBackwardWeightDiff(
      ctx, BnInOp2Blob("in"), hidden_blob, o_out_diff_blob,
      BnInOp2Blob("h2h_o_weight_diff"), BnInOp2Blob("i2h_o_weight_diff"));

  // in_diff
  if (BnInOp2Blob("in_diff") != nullptr) {
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
        static_cast<T>(0), f_data_diff_blob, BnInOp2Blob("i2h_f_weight_diff"),
        BnInOp2Blob("in_diff"));
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
        static_cast<T>(1), i_data_diff_blob, BnInOp2Blob("i2h_i_weight_diff"),
        BnInOp2Blob("in_diff"));
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
        static_cast<T>(1), c_data_diff_blob, BnInOp2Blob("i2h_c_weight_diff"),
        BnInOp2Blob("in_diff"));
    KernelUtil<device_type, T>::BlobGemm(
        ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
        static_cast<T>(1), o_data_diff_blob, BnInOp2Blob("i2h_o_weight_diff"),
        BnInOp2Blob("in_diff"));
  }
// bias diff
#define OF_LSTM_COMPUTE_BIAS_DIFF(bias_diff, bias_mul_name, data_diff) \
  if (BnInOp2Blob(#bias_diff) != nullptr) {                            \
    KernelUtil<device_type, T>::BlobGemm(                              \
        ctx.device_ctx, CblasTrans, CblasNoTrans, static_cast<T>(1),   \
        static_cast<T>(0), BnInOp2Blob(#bias_mul_name), data_diff,     \
        BnInOp2Blob(#bias_diff));                                      \
  }

  OF_LSTM_COMPUTE_BIAS_DIFF(bias_f_diff, bias_f_multiplier, f_out_diff_blob);
  OF_LSTM_COMPUTE_BIAS_DIFF(bias_i_diff, bias_i_multiplier, i_out_diff_blob);
  OF_LSTM_COMPUTE_BIAS_DIFF(bias_c_diff, bias_c_multiplier, c_out_diff_blob);
  OF_LSTM_COMPUTE_BIAS_DIFF(bias_o_diff, bias_o_multiplier, o_out_diff_blob);
#undef OF_LSTM_COMPUTE_BIAS_DIFF

  // hidden diff
  if (BnInOp2Blob("in")->col_id() != 0 || NeedExternalH0()
      || this->op_conf().basic_lstm_conf().is_init_hidden_trainable()) {
    BasicLstmKernelUtil<device_type, T>::ComputeBackwardHiddenDiff(
        ctx, BnInOp2Blob("h2h_f_weight"), BnInOp2Blob("h2h_i_weight"),
        BnInOp2Blob("h2h_c_weight"), BnInOp2Blob("h2h_o_weight"),
        f_data_diff_blob, i_data_diff_blob, c_data_diff_blob, o_data_diff_blob,
        hidden_diff_blob);
  }
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::VirtualInitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
#define OF_INIT_LSTM_MODEL_WITH_RAND_SEED(model_name)      \
  KernelUtil<device_type, T>::InitializeWithProperConf(    \
      ctx,                                                 \
      OF_PB_POINTER_GET(this->op_conf().basic_lstm_conf(), \
                        model_name##_initializer),         \
      (*random_seed_gen)(), BnInOp2Blob(#model_name))

  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(i2h_f_weight);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(i2h_i_weight);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(i2h_o_weight);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(i2h_c_weight);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(h2h_f_weight);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(h2h_i_weight);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(h2h_o_weight);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(h2h_c_weight);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(bias_f);

#undef OF_INIT_LSTM_MODEL_WITH_RAND_SEED
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::VirtualInitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
#define OF_INIT_LSTM_MODEL_BLOBS_WITH_DIR(i2h_weight, h2h_weight, bias)       \
  Blob* i2h_weight##_blob = BnInOp2Blob(#i2h_weight);                         \
  Blob* h2h_weight##_blob = BnInOp2Blob(#h2h_weight);                         \
  KernelUtil<device_type, T>::InitializeWithModelDir(                         \
      ctx, part_id, part_num, model_load_dir, i2h_weight##_blob, #i2h_weight, \
      i2h_weight##_blob->shape().At(0), i2h_weight##_blob->shape().Count(1)); \
  KernelUtil<device_type, T>::InitializeWithModelDir(                         \
      ctx, part_id, part_num, model_load_dir, h2h_weight##_blob, #h2h_weight, \
      h2h_weight##_blob->shape().At(0), h2h_weight##_blob->shape().Count(1)); \
  KernelUtil<device_type, T>::InitializeWithModelDir(                         \
      ctx, part_id, part_num, model_load_dir, BnInOp2Blob(#bias), #bias,      \
      BnInOp2Blob(#bias)->shape().At(0), 1);

  OF_INIT_LSTM_MODEL_BLOBS_WITH_DIR(i2h_f_weight, h2h_f_weight, bias_f);
  OF_INIT_LSTM_MODEL_BLOBS_WITH_DIR(i2h_i_weight, h2h_i_weight, bias_i);
  OF_INIT_LSTM_MODEL_BLOBS_WITH_DIR(i2h_c_weight, h2h_c_weight, bias_c);
  OF_INIT_LSTM_MODEL_BLOBS_WITH_DIR(i2h_o_weight, h2h_o_weight, bias_o);

#undef OF_INIT_LSTM_MODEL_BLOBS_WITH_DIR
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::InitPureModelTmpBlobs(
    const KernelCtx& ctx, const ParallelContext* parallel_ctx,
    std::function<Blob*(const std ::string&)> BnInOp2Blob) const {
  Blob* h0_blob = BnInOp2Blob("h0");
  if (!(this->NeedExternalH0()) && h0_blob != nullptr) {
    const InitializerConf* init_hidden_initializer = nullptr;
    if (HasInitHiddenInitializer()) {
      init_hidden_initializer =
          static_cast<const InitializerConf*>(&GetMessageFromPbMessage(
              GetBasicLstmOpConf(), "init_hidden_initializer"));
    }
    int64_t random_seed = *static_cast<int64_t*>(ctx.other);
    std::mt19937 random_seed_gen(random_seed);
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx.device_ctx, init_hidden_initializer, random_seed_gen(), h0_blob);
  }
  Blob* c0_blob = BnInOp2Blob("c0");
  if (!(this->NeedExternalC0()) && c0_blob != nullptr) {
    const InitializerConf* init_cell_initializer = nullptr;
    if (HasInitCellInitializer()) {
      init_cell_initializer =
          static_cast<const InitializerConf*>(&GetMessageFromPbMessage(
              GetBasicLstmOpConf(), "init_cell_initializer"));
    }
    int64_t random_seed = *static_cast<int64_t*>(ctx.other);
    std::mt19937 random_seed_gen(random_seed);
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx.device_ctx, init_cell_initializer, random_seed_gen(), c0_blob);
  }
#define OF_INIT_LSTM_MODEL_TMP_BLOBS(bias_mul)                         \
  InitializerConf bias_mul##_fill_conf;                                \
  bias_mul##_fill_conf.mutable_constant_conf()->set_value(1.f);        \
  KernelUtil<device_type, T>::Initialize(ctx, bias_mul##_fill_conf, 0, \
                                         BnInOp2Blob(#bias_mul));
  /*
    InitializerConf bias_f_multiplier_fill_conf;
    bias_f_multiplier_fill_conf.mutable_constant_conf()->set_value(1.f);
    KernelUtil<device_type, T>::Initialize(ctx, bias_f_multiplier_fill_conf, 0,
                                           BnInOp2Blob("bias_f_multiplier"));*/
  OF_INIT_LSTM_MODEL_TMP_BLOBS(bias_f_multiplier);
  OF_INIT_LSTM_MODEL_TMP_BLOBS(bias_i_multiplier);
  OF_INIT_LSTM_MODEL_TMP_BLOBS(bias_c_multiplier);
  OF_INIT_LSTM_MODEL_TMP_BLOBS(bias_o_multiplier);
#undef OF_INIT_LSTM_MODEL_TMP_BLOBS
}

template<DeviceType device_type, typename T>
void BasicLstmKernelUtil<device_type, T>::ComputeForwardGateOut(
    const KernelCtx& ctx, const Blob* i2h_weight, const Blob* hidden,
    const Blob* h2h_weight, const Blob* input, const Blob* bias,
    const Blob* bias_mul, Blob* state_data) {
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       input, i2h_weight, state_data);
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(1),
                                       hidden, h2h_weight, state_data);
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), bias_mul, bias, state_data);
}

template<DeviceType device_type, typename T>
void BasicLstmKernelUtil<device_type, T>::ComputeBackwardCellOutDiff(
    const KernelCtx& ctx, const Blob* rec_out_diff, Blob* candidate_out,
    Blob* cell_out, Blob* cell_out_diff, Blob* o_out, Blob* out_diff,
    BwActivationFunc<device_type, T> activation_bw_func_) {
  KernelUtil<device_type, T>::Axpy(ctx.device_ctx, o_out->shape().elem_cnt(),
                                   static_cast<T>(1), rec_out_diff->dptr<T>(),
                                   static_cast<T>(1), out_diff->mut_dptr<T>(),
                                   static_cast<T>(1));
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, o_out->shape().elem_cnt(),
                                  out_diff->dptr<T>(), o_out->dptr<T>(),
                                  o_out->mut_dptr<T>());
  (*activation_bw_func_)(ctx.device_ctx, o_out->shape().elem_cnt(),
                         cell_out->dptr<T>(), candidate_out->dptr<T>(),
                         o_out->dptr<T>(), o_out->mut_dptr<T>());
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, o_out->shape().elem_cnt(), static_cast<T>(1),
      o_out->dptr<T>(), static_cast<T>(1), cell_out_diff->mut_dptr<T>(),
      static_cast<T>(1));
}

template<DeviceType device_type, typename T>
void BasicLstmKernelUtil<device_type, T>::ComputeBackwardWeightDiff(
    const KernelCtx& ctx, const Blob* input, const Blob* hidden,
    Blob* gate_out_diff, Blob* h2h_weight_diff, Blob* i2h_weight_diff) {
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       gate_out_diff, hidden, h2h_weight_diff);

  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       gate_out_diff, input, i2h_weight_diff);
}

template<DeviceType device_type, typename T>
void BasicLstmKernelUtil<device_type, T>::ComputeBackwardHiddenDiff(
    const KernelCtx& ctx, const Blob* h2h_f_weight, const Blob* h2h_i_weight,
    const Blob* h2h_c_weight, const Blob* h2h_o_weight, Blob* f_data_diff,
    Blob* i_data_diff, Blob* c_data_diff, Blob* o_data_diff,
    Blob* hidden_diff) {
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(0), f_data_diff, h2h_f_weight, hidden_diff);
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), i_data_diff, h2h_i_weight, hidden_diff);
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), c_data_diff, h2h_c_weight, hidden_diff);
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), o_data_diff, h2h_o_weight, hidden_diff);
}
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBasicLstmConf, BasicLstmKernel,
                           FLOATING_DATA_TYPE_SEQ)
}  // namespace oneflow
