#include "oneflow/core/kernel/basic_lstm_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::VirtualKernelInit(
    const ParallelContext* parallel_ctx) {
  auto& input_bns = this->kernel_conf().input_bns();
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
    UNEXPECTED_RUN();
  }
}

template<DeviceType device_type, typename T>
const PbMessage& BasicLstmKernel<device_type, T>::GetRecurrentOpConf() const {
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
bool BasicLstmKernel<device_type, T>::NeedExternalC0() const {
  return need_external_c0_;
}

template<DeviceType device_type, typename T>
Blob* BasicLstmKernel<device_type, T>::GetCellBlob(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (BnInOp2Blob("in")->col_id() == 0) { return BnInOp2Blob("c0"); }
  return BnInOp2Blob("rec_cell_in");
}

template<DeviceType device_type, typename T>
Blob* BasicLstmKernel<device_type, T>::GetCellDiffBlob(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (BnInOp2Blob("in")->col_id() == 0) { return BnInOp2Blob("c0_diff"); }
  return BnInOp2Blob("rec_cell_in_diff");
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::VirtualForwardColNum(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("rec_cell_out")
      ->CopyColNumFrom<device_type>(ctx.device_ctx, BnInOp2Blob("in"));
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::VirtualBackwardColNum(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("rec_cell_in_diff")
      ->CopyColNumFrom<device_type>(ctx.device_ctx, BnInOp2Blob("out_diff"));
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);
  Blob* gate_tmp_data_blob = BnInOp2Blob("gate_tmp_data");
  Blob* c_data_blob = BnInOp2Blob("c_data");
  Blob* candidate_out_blob = BnInOp2Blob("candidate_out");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* rec_cell_in_blob = BnInOp2Blob("rec_cell_in");
  Blob* rec_cell_out_blob = BnInOp2Blob("rec_cell_out");
  Blob* f_out_blob = BnInOp2Blob("f_out");
  Blob* i_out_blob = BnInOp2Blob("i_out");
  Blob* c_out_blob = BnInOp2Blob("c_out");
  Blob* o_out_blob = BnInOp2Blob("o_out");
  //  f_out = sigmoid(W[x, h] + bias)
  BasicLstmKernelUtil<device_type, T>::ComputeForwardGateOut(
      ctx, gate_tmp_data_blob, BnInOp2Blob("i2h_f_weight"), hidden_blob,
      BnInOp2Blob("h2h_f_weight"), BnInOp2Blob("in"),
      BnInOp2Blob("bias_multiplier"), BnInOp2Blob("bias_f"));
  KernelUtil<device_type, T>::Sigmoid(
      ctx.device_ctx, out_blob->shape().elem_cnt(),
      gate_tmp_data_blob->dptr<T>(), f_out_blob->mut_dptr<T>());

  //  i_out = sigmoid(W[x, h] + bias)
  BasicLstmKernelUtil<device_type, T>::ComputeForwardGateOut(
      ctx, gate_tmp_data_blob, BnInOp2Blob("i2h_i_weight"), hidden_blob,
      BnInOp2Blob("h2h_i_weight"), BnInOp2Blob("in"),
      BnInOp2Blob("bias_multiplier"), BnInOp2Blob("bias_i"));
  KernelUtil<device_type, T>::Sigmoid(
      ctx.device_ctx, out_blob->shape().elem_cnt(),
      gate_tmp_data_blob->dptr<T>(), i_out_blob->mut_dptr<T>());

  // c_out = activation(W[x, h] + bias)
  BasicLstmKernelUtil<device_type, T>::ComputeForwardGateOut(
      ctx, gate_tmp_data_blob, BnInOp2Blob("i2h_c_weight"), hidden_blob,
      BnInOp2Blob("h2h_c_weight"), BnInOp2Blob("in"),
      BnInOp2Blob("bias_multiplier"), BnInOp2Blob("bias_c"));
  (*activation_fw_func_)(ctx.device_ctx, out_blob->shape().elem_cnt(),
                         c_data_blob->dptr<T>(), c_out_blob->mut_dptr<T>());

  //  o_out = sigmoid(W[x, h] + bias)
  BasicLstmKernelUtil<device_type, T>::ComputeForwardGateOut(
      ctx, gate_tmp_data_blob, BnInOp2Blob("i2h_o_weight"), hidden_blob,
      BnInOp2Blob("h2h_o_weight"), BnInOp2Blob("in"),
      BnInOp2Blob("bias_multiplier"), BnInOp2Blob("bias_o"));
  KernelUtil<device_type, T>::Sigmoid(
      ctx.device_ctx, out_blob->shape().elem_cnt(),
      gate_tmp_data_blob->dptr<T>(), o_out_blob->mut_dptr<T>());

  // rec_cell_out = f_out .* rec_cell_in
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, out_blob->shape().elem_cnt(), f_out_blob->dptr<T>(),
      rec_cell_in_blob->dptr<T>(), rec_cell_out_blob->mut_dptr<T>());
  // candidate_out = i_out .* c_out
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                  i_out_blob->dptr<T>(), c_out_blob->dptr<T>(),
                                  candidate_out_blob->mut_dptr<T>());
  // rec_cell_out = candidate_out + rec_cell_out
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, out_blob->shape().elem_cnt(), static_cast<T>(1),
      candidate_out_blob->dptr<T>(), static_cast<T>(1),
      rec_cell_out_blob->mut_dptr<T>(), static_cast<T>(1));

  //  candidate_out = activation(rec_cell_out)
  (*activation_fw_func_)(ctx.device_ctx, out_blob->shape().elem_cnt(),
                         rec_cell_out_blob->dptr<T>(),
                         candidate_out_blob->mut_dptr<T>());
  //  out = o_out * candidate_out
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, out_blob->shape().elem_cnt(), o_out_blob->dptr<T>(),
      candidate_out_blob->dptr<T>(), out_blob->mut_dptr<T>());
  // rec_out = out
  BnInOp2Blob("rec_out")->CopyDataContentFrom<device_type>(ctx.device_ctx,
                                                           out_blob);
}
template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);
  Blob* hidden_diff_blob = this->GetHiddenDiffBlob(BnInOp2Blob);

  Blob* f_data_diff_blob = BnInOp2Blob("f_data_diff");
  Blob* i_data_diff_blob = BnInOp2Blob("i_data_diff");
  Blob* c_data_diff_blob = BnInOp2Blob("c_data_diff");
  Blob* o_data_diff_blob = BnInOp2Blob("o_data_diff");

  Blob* rec_cell_out_diff_blob = BnInOp2Blob("rec_cell_out_diff");
  Blob* candidate_out_blob = BnInOp2Blob("candidate_out");

  if (in_blob->col_id() == in_blob->max_col_id()) {
    // rec_cell_out_diff = out_diff * o_out * bw(rec_cell_out)
    BasicLstmKernelUtil<device_type, T>::ComputeRecCellOutDiff(
        ctx, candidate_out_blob, rec_cell_out_diff_blob, activation_bw_func_,
        BnInOp2Blob);
  }

  // activation_data_diff
  BasicLstmKernelUtil<device_type, T>::ComputeActivationDataDiff(
      ctx, rec_cell_out_diff_blob, f_data_diff_blob, i_data_diff_blob,
      c_data_diff_blob, o_data_diff_blob, BnInOp2Blob, activation_bw_func_);

  // weights diff
  BasicLstmKernelUtil<device_type, T>::ComputeAllWeightDiff(
      ctx, in_blob, hidden_blob, f_data_diff_blob, i_data_diff_blob,
      c_data_diff_blob, o_data_diff_blob, BnInOp2Blob("h2h_f_weight_diff"),
      BnInOp2Blob("i2h_f_weight_diff"), BnInOp2Blob("h2h_i_weight_diff"),
      BnInOp2Blob("i2h_i_weight_diff"), BnInOp2Blob("h2h_c_weight_diff"),
      BnInOp2Blob("i2h_c_weight_diff"), BnInOp2Blob("h2h_o_weight_diff"),
      BnInOp2Blob("i2h_o_weight_diff"));

  // in diff
  if (BnInOp2Blob("in_diff") != nullptr) {
    BasicLstmKernelUtil<device_type, T>::ComputeInDiff(
        ctx, f_data_diff_blob, i_data_diff_blob, c_data_diff_blob,
        o_data_diff_blob, BnInOp2Blob("in_diff"), BnInOp2Blob);
  }
  // bias diff
  BasicLstmKernelUtil<device_type, T>::ComputeAllBiasDiff(
      ctx, f_data_diff_blob, i_data_diff_blob, c_data_diff_blob,
      o_data_diff_blob, BnInOp2Blob("bias_f_diff"), BnInOp2Blob("bias_i_diff"),
      BnInOp2Blob("bias_c_diff"), BnInOp2Blob("bias_o_diff"), BnInOp2Blob);

  // hidden diff
  if (BnInOp2Blob("in")->col_id() != 0 || this->NeedExternalH0()
      || this->op_conf().basic_lstm_conf().is_init_hidden_trainable()) {
    BasicLstmKernelUtil<device_type, T>::ComputeHiddenDiff(
        ctx, f_data_diff_blob, i_data_diff_blob, c_data_diff_blob,
        o_data_diff_blob, hidden_diff_blob, BnInOp2Blob);
  }
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::VirtualInitModelBlobsWithRandomSeed(
    const KernelCtx& ctx, std::mt19937 random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
#define OF_INIT_LSTM_MODEL_WITH_RAND_SEED(model_name, type) \
  KernelUtil<device_type, T>::InitializeWithProperConf(     \
      ctx.device_ctx,                                       \
      OF_PB_POINTER_GET(this->op_conf().basic_lstm_conf(),  \
                        type##_initializer),                \
      random_seed_gen(), BnInOp2Blob(#model_name));

  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(i2h_f_weight, i2h_weight);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(i2h_i_weight, i2h_weight);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(i2h_o_weight, i2h_weight);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(i2h_c_weight, i2h_weight);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(h2h_f_weight, h2h_weight);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(h2h_i_weight, h2h_weight);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(h2h_o_weight, h2h_weight);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(h2h_c_weight, h2h_weight);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(bias_f, bias);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(bias_i, bias);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(bias_c, bias);
  OF_INIT_LSTM_MODEL_WITH_RAND_SEED(bias_o, bias);
#undef OF_INIT_LSTM_MODEL_WITH_RAND_SEED
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::VirtualInitModelBlobsWithDir(
    const KernelCtx& ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
#define OF_INIT_LSTM_MODEL_BLOBS_WITH_DIR(i2h_weight, h2h_weight, bias)      \
  Blob* i2h_weight##_blob = BnInOp2Blob(#i2h_weight);                        \
  Blob* h2h_weight##_blob = BnInOp2Blob(#h2h_weight);                        \
  KernelUtil<device_type, T>::InitializeWithModelDir(                        \
      ctx.device_ctx, part_id, part_num, model_load_dir, i2h_weight##_blob,  \
      #i2h_weight, i2h_weight##_blob->shape().At(0),                         \
      i2h_weight##_blob->shape().Count(1));                                  \
  KernelUtil<device_type, T>::InitializeWithModelDir(                        \
      ctx.device_ctx, part_id, part_num, model_load_dir, h2h_weight##_blob,  \
      #h2h_weight, h2h_weight##_blob->shape().At(0),                         \
      h2h_weight##_blob->shape().Count(1));                                  \
  KernelUtil<device_type, T>::InitializeWithModelDir(                        \
      ctx.device_ctx, part_id, part_num, model_load_dir, BnInOp2Blob(#bias), \
      #bias, BnInOp2Blob(#bias)->shape().At(0), 1);

  OF_INIT_LSTM_MODEL_BLOBS_WITH_DIR(i2h_f_weight, h2h_f_weight, bias_f);
  OF_INIT_LSTM_MODEL_BLOBS_WITH_DIR(i2h_i_weight, h2h_i_weight, bias_i);
  OF_INIT_LSTM_MODEL_BLOBS_WITH_DIR(i2h_c_weight, h2h_c_weight, bias_c);
  OF_INIT_LSTM_MODEL_BLOBS_WITH_DIR(i2h_o_weight, h2h_o_weight, bias_o);

#undef OF_INIT_LSTM_MODEL_BLOBS_WITH_DIR
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::InitModelTmpBlobs(
    const KernelCtx& ctx, const ParallelContext* parallel_ctx,
    std::function<Blob*(const std ::string&)> BnInOp2Blob) const {
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

  Blob* c0_blob = BnInOp2Blob("c0");
  if (!(this->NeedExternalC0()) && c0_blob != nullptr) {
    InitializerConf init_cell_fill_conf;
    init_cell_fill_conf.mutable_constant_conf()->set_value(0.f);
    InitializerConf* init_cell_initializer = &init_cell_fill_conf;
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx.device_ctx, init_cell_initializer, 0, c0_blob);
  }

  InitializerConf bias_multiplier_fill_conf;
  bias_multiplier_fill_conf.mutable_constant_conf()->set_value(1.f);
  KernelUtil<device_type, T>::Initialize(ctx.device_ctx,
                                         bias_multiplier_fill_conf, 0,
                                         BnInOp2Blob("bias_multiplier"));
}

template<DeviceType device_type, typename T>
void BasicLstmKernelUtil<device_type, T>::ComputeForwardGateOut(
    const KernelCtx& ctx, const Blob* i2h_weight, const Blob* hidden,
    const Blob* h2h_weight, const Blob* input, const Blob* bias,
    const Blob* bias_mul, Blob* gate_tmp_data) {
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       input, i2h_weight, gate_tmp_data);
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(1),
                                       hidden, h2h_weight, gate_tmp_data);
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), bias_mul, bias, gate_tmp_data);
}

template<DeviceType device_type, typename T>
void BasicLstmKernelUtil<device_type, T>::ComputeRecCellOutDiff(
    const KernelCtx& ctx, const Blob* out_diff, Blob* rec_cell_out_diff,
    BwActivationFunc<device_type, T> activation_bw_func_,
    std::function<Blob*(const std::string&)> BnInOp2Blob) {
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, BnInOp2Blob("out")->shape().elem_cnt(),
      out_diff->dptr<T>(), BnInOp2Blob("o_out")->dptr<T>(),
      BnInOp2Blob("rec_out_diff")->mut_dptr<T>());
  (*activation_bw_func_)(ctx.device_ctx, BnInOp2Blob("out")->shape().elem_cnt(),
                         BnInOp2Blob("rec_cell_out")->dptr<T>(),
                         BnInOp2Blob("candidate_out")->dptr<T>(),
                         BnInOp2Blob("rec_out_diff")->dptr<T>(),
                         BnInOp2Blob("o_out")->mut_dptr<T>());
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, BnInOp2Blob("out")->shape().elem_cnt(), static_cast<T>(1),
      BnInOp2Blob("o_out")->dptr<T>(), static_cast<T>(1),
      rec_cell_out_diff->mut_dptr<T>(), static_cast<T>(1));
}

template<DeviceType device_type, typename T>
void BasicLstmKernelUtil<device_type, T>::ComputeActivationDataDiff(
    const KernelCtx& ctx, const Blob* rec_cell_out_diff, Blob* f_data_diff,
    Blob* i_data_diff, Blob* c_data_diff, Blob* o_data_diff,
    std::function<Blob*(const std::string&)> BnInOp2Blob,
    BwActivationFunc<device_type, T> activation_bw_func_) {
  // f_data_diff = sigmoidbw( rec_cell_out_diff * c_cell_in)
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, BnInOp2Blob("out")->shape().elem_cnt(),
      rec_cell_out_diff->dptr<T>(), BnInOp2Blob("rec_cell_in")->dptr<T>(),
      BnInOp2Blob("f_out_diff")->mut_dptr<T>());
  KernelUtil<device_type, T>::SigmoidBackward(
      ctx.device_ctx, BnInOp2Blob("out")->shape().elem_cnt(),
      BnInOp2Blob("gate_tmp_data")->dptr<T>(), BnInOp2Blob("f_out")->dptr<T>(),
      BnInOp2Blob("f_out_diff")->dptr<T>(), f_data_diff->mut_dptr<T>());

  // i_data_diff = sigmoidbw( rec_cell_out_diff * c_out)
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, BnInOp2Blob("out")->shape().elem_cnt(),
      rec_cell_out_diff->dptr<T>(), BnInOp2Blob("c_out")->dptr<T>(),
      BnInOp2Blob("i_out_diff")->mut_dptr<T>());
  KernelUtil<device_type, T>::SigmoidBackward(
      ctx.device_ctx, BnInOp2Blob("out")->shape().elem_cnt(),
      BnInOp2Blob("gate_tmp_data")->dptr<T>(), BnInOp2Blob("i_out")->dptr<T>(),
      BnInOp2Blob("i_out_diff")->dptr<T>(), i_data_diff->mut_dptr<T>());

  // c_data_diff = activation_bw_func_(rec_cell_out_diff * i_out)
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, BnInOp2Blob("out")->shape().elem_cnt(),
      rec_cell_out_diff->dptr<T>(), BnInOp2Blob("i_out")->dptr<T>(),
      BnInOp2Blob("c_out_diff")->mut_dptr<T>());
  (*activation_bw_func_)(
      ctx.device_ctx, BnInOp2Blob("out")->shape().elem_cnt(),
      BnInOp2Blob("gate_tmp_data")->dptr<T>(), BnInOp2Blob("c_out")->dptr<T>(),
      BnInOp2Blob("c_out_diff")->dptr<T>(), c_data_diff->mut_dptr<T>());

  // o_data_diff = sigmoid_bw(rec_out_diff * tanh(rec_cell_out))
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, BnInOp2Blob("out")->shape().elem_cnt(),
      rec_cell_out_diff->dptr<T>(), BnInOp2Blob("candidate_out")->dptr<T>(),
      BnInOp2Blob("o_out_diff")->mut_dptr<T>());
  KernelUtil<device_type, T>::SigmoidBackward(
      ctx.device_ctx, BnInOp2Blob("out")->shape().elem_cnt(),
      BnInOp2Blob("gate_tmp_data")->dptr<T>(), BnInOp2Blob("o_out")->dptr<T>(),
      BnInOp2Blob("o_out_diff")->dptr<T>(), o_data_diff->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void BasicLstmKernelUtil<device_type, T>::ComputeAllWeightDiff(
    const KernelCtx& ctx, const Blob* input, const Blob* hidden,
    const Blob* f_data_diff, const Blob* i_data_diff, const Blob* c_data_diff,
    const Blob* o_data_diff, Blob* f_h2h_diff, Blob* f_i2h_diff,
    Blob* i_h2h_diff, Blob* i_i2h_diff, Blob* c_h2h_diff, Blob* c_i2h_diff,
    Blob* o_h2h_diff, Blob* o_i2h_diff) {
#define OF_LSTM_COMPUTE_WEIGHTS_DIFF(gate)                           \
  KernelUtil<device_type, T>::BlobGemm(                              \
      ctx.device_ctx, CblasTrans, CblasNoTrans, static_cast<T>(1),   \
      static_cast<T>(0), gate##_data_diff, hidden, gate##_h2h_diff); \
  KernelUtil<device_type, T>::BlobGemm(                              \
      ctx.device_ctx, CblasTrans, CblasNoTrans, static_cast<T>(1),   \
      static_cast<T>(0), gate##_data_diff, input, gate##_i2h_diff);

  OF_LSTM_COMPUTE_WEIGHTS_DIFF(f);
  OF_LSTM_COMPUTE_WEIGHTS_DIFF(i);
  OF_LSTM_COMPUTE_WEIGHTS_DIFF(c);
  OF_LSTM_COMPUTE_WEIGHTS_DIFF(o);
#undef OF_LSTM_COMPUTE_WEIGHTS_DIFF
}

template<DeviceType device_type, typename T>
void BasicLstmKernelUtil<device_type, T>::ComputeAllBiasDiff(
    const KernelCtx& ctx, const Blob* f_data_diff, const Blob* i_data_diff,
    const Blob* c_data_diff, const Blob* o_data_diff, Blob* bias_f_diff,
    Blob* bias_i_diff, Blob* bias_c_diff, Blob* bias_o_diff,
    std::function<Blob*(const std::string&)> BnInOp2Blob) {
#define OF_LSTM_COMPUTE_BIAS_DIFF(bias_diff, data_diff)               \
  if (bias_diff != nullptr) {                                         \
    KernelUtil<device_type, T>::BlobGemm(                             \
        ctx.device_ctx, CblasTrans, CblasNoTrans, static_cast<T>(1),  \
        static_cast<T>(0), BnInOp2Blob("bias_multiplier"), data_diff, \
        BnInOp2Blob(#bias_diff));                                     \
  }
  OF_LSTM_COMPUTE_BIAS_DIFF(bias_f_diff, f_data_diff);
  OF_LSTM_COMPUTE_BIAS_DIFF(bias_i_diff, i_data_diff);
  OF_LSTM_COMPUTE_BIAS_DIFF(bias_c_diff, c_data_diff);
  OF_LSTM_COMPUTE_BIAS_DIFF(bias_o_diff, o_data_diff);
#undef OF_LSTM_COMPUTE_BIAS_DIFF
}

template<DeviceType device_type, typename T>
void BasicLstmKernelUtil<device_type, T>::ComputeInDiff(
    const KernelCtx& ctx, const Blob* f_data_diff, const Blob* i_data_diff,
    const Blob* c_data_diff, const Blob* o_data_diff, Blob* in_diff,
    std::function<Blob*(const std::string&)> BnInOp2Blob) {
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(0), f_data_diff, BnInOp2Blob("i2h_f_weight_diff"),
      in_diff);
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), i_data_diff, BnInOp2Blob("i2h_i_weight_diff"),
      in_diff);
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), c_data_diff, BnInOp2Blob("i2h_c_weight_diff"),
      in_diff);
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), o_data_diff, BnInOp2Blob("i2h_o_weight_diff"),
      in_diff);
}

template<DeviceType device_type, typename T>
void BasicLstmKernelUtil<device_type, T>::ComputeHiddenDiff(
    const KernelCtx& ctx, const Blob* f_data_diff, const Blob* i_data_diff,
    const Blob* c_data_diff, const Blob* o_data_diff, Blob* hidden_diff,
    std::function<Blob*(const std::string&)> BnInOp2Blob) {
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(0), f_data_diff, BnInOp2Blob("h2h_f_weight"), hidden_diff);
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), i_data_diff, BnInOp2Blob("h2h_i_weight"), hidden_diff);
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), c_data_diff, BnInOp2Blob("h2h_c_weight"), hidden_diff);
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), o_data_diff, BnInOp2Blob("h2h_o_weight"), hidden_diff);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBasicLstmConf, BasicLstmKernel,
                           FLOATING_DATA_TYPE_SEQ)
}  // namespace oneflow
