#include "oneflow/core/kernel/basic_lstm_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::VirtualKernelInit(
    const ParallelContext*) {
  auto& input_bns = this->kernel_conf().input_bns();
  need_external_h0_ =
      std::find(input_bns.begin(), input_bns.end(), "h0") != input_bns.end();
  need_external_c0_ =
      std::find(input_bns.begin(), input_bns.end(), "c0") != input_bns.end();
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
  const Blob* cell_blob = this->GetCellBlob(BnInOp2Blob);
  Blob* f_gate_out_blob = BnInOp2Blob("f_gate_out");
  Blob* i_gate_out_blob = BnInOp2Blob("i_gate_out");
  Blob* o_gate_out_blob = BnInOp2Blob("o_gate_out");
  Blob* c_gate_out_blob = BnInOp2Blob("c_gate_out");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* cell_in_blob = BnInOp2Blob("cell_in");
  Blob* cell_out_blob = BnInOp2Blob("cell_out");
  Blob* update_blob = BnInOp2Blob("update_out");
  Blob* f_out_blob = BnInOp2Blob("f_out");
  Blob* i_out_blob = BnInOp2Blob("i_out");
  Blob* o_out_blob = BnInOp2Blob("o_out");
  Blob* c_out_blob = BnInOp2Blob("c_out");
  // f_gate_out = in * i2h_f_weight + hidden * h2h_f_weight +
  //							bias * bias_f_mulipler
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, static_cast<T>(1),
      static_cast<T>(0), BnInOp2Blob("in"), BnInOp2Blob("i2h_f_weight"),
      f_gate_out_blob);
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(1),
                                       hidden_blob, BnInOp2Blob("h2h_f_weight"),
                                       f_gate_out_blob);
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), BnInOp2Blob("bias_f_multiplier"),
      BnInOp2Blob("bias_f"), f_gate_out_blob);
  // f_out = sigmoid(f_gate_out)
  KernelUtil<device_type, T>::Sigmoid(
      ctx.device_ctx, out_blob->shape().elem_cnt(), f_gate_out_blob->dptr<T>(),
      f_out_blob->mut_dptr<T>());

  // i_gate_out = in * i2h_i_weight + hidden * h2h_i_weight +
  //							bias_i * bias_i_mulipler
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, static_cast<T>(1),
      static_cast<T>(0), BnInOp2Blob("in"), BnInOp2Blob("i2h_i_weight"),
      i_gate_out_blob);
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(1),
                                       hidden_blob, BnInOp2Blob("h2h_i_weight"),
                                       i_gate_out_blob);
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), BnInOp2Blob("bias_i_multiplier"),
      BnInOp2Blob("bias_i"), i_gate_out_blob);
  // i_out = sigmoid(i_gate_out)
  KernelUtil<device_type, T>::Sigmoid(
      ctx.device_ctx, out_blob->shape().elem_cnt(), i_gate_out_blob->dptr<T>(),
      i_out_blob->mut_dptr<T>());

  // c_gate_out = in * i2h_c_weight + hidden * h2h_c_weight +
  //							bias_c * bias_c_mulipler
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, static_cast<T>(1),
      static_cast<T>(0), BnInOp2Blob("in"), BnInOp2Blob("i2h_c_weight"),
      c_gate_out_blob);
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(1),
                                       hidden_blob, BnInOp2Blob("h2h_c_weight"),
                                       c_gate_out_blob);
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), BnInOp2Blob("bias_c_multiplier"),
      BnInOp2Blob("bias_c"), c_gate_out_blob);

  // c_out = tanh(c_gate_out)
  KernelUtil<device_type, T>::TanH(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                   c_gate_out_blob->dptr<T>(),
                                   c_out_blob->mut_dptr<T>());

  // o_gate_out = in * i2h_o_weight + hidden * h2h_o_weight +
  //							bias_o * bias_o_mulipler
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasTrans, static_cast<T>(1),
      static_cast<T>(0), BnInOp2Blob("in"), BnInOp2Blob("i2h_o_weight"),
      o_gate_out_blob);
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasNoTrans, CblasTrans,
                                       static_cast<T>(1), static_cast<T>(1),
                                       hidden_blob, BnInOp2Blob("h2h_o_weight"),
                                       o_gate_out_blob);
  KernelUtil<device_type, T>::BlobGemm(
      ctx.device_ctx, CblasNoTrans, CblasNoTrans, static_cast<T>(1),
      static_cast<T>(1), BnInOp2Blob("bias_o_multiplier"),
      BnInOp2Blob("bias_o"), o_gate_out_blob);

  // o_out = sigmoid(o_gate_out)
  KernelUtil<device_type, T>::Sigmoid(
      ctx.device_ctx, out_blob->shape().elem_cnt(), o_gate_out_blob->dptr<T>(),
      o_out_blob->mut_dptr<T>());

  // out = o_out *  tanh(c_out)
  KernelUtil<device_type, T>::TanH(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                   c_gate_out_blob->dptr<T>(),
                                   out_blob->mut_dptr<T>());
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                  o_out_blob->dptr<T>(), out_blob->dptr<T>(),
                                  out_blob->mut_dptr<T>());

  // rec_out = out
  BnInOp2Blob("rec_out")->CopyDataContentFrom(ctx.device_ctx, out_blob);

  // cell_out = f_out * cell_in + i_out *  c_out
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, out_blob->shape().elem_cnt(), f_out_blob->dptr<T>(),
      cell_in_blob->dptr<T>(), cell_out_blob->mut_dptr<T>());
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, out_blob->shape().elem_cnt(), i_out_blob->dptr<T>(),
      cell_out_blob->dptr<T>(), update_blob->mut_dptr<T>());
  KernelUtil<device_type, T>::Axpy(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                   static_cast<T>(1), cell_out_blob->dptr<T>(),
                                   static_cast<T>(1), update_blob->dptr<T>(),
                                   static_cast<T>(1));
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_blob = BnInOp2Blob("out");
  Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* hidden_blob = this->GetHiddenBlob(BnInOp2Blob);

  const Blob* rec_out_diff_blob = BnInOp2Blob("rec_out_diff");
  const Blob* f_gate_out_blob = BnInOp2Blob("f_gate_out");
  const Blob* f_out_blob = BnInOp2Blob("f_out");
  Blob* f_gate_out_diff_blob = BnInOp2Blob("f_gate_out_diff");
  Blob* f_out_diff_blob = BnInOp2Blob("f_out_diff");

  const Blob* i_gate_out_blob = BnInOp2Blob("i_gate_out");
  const Blob* i_out_blob = BnInOp2Blob("i_out");
  Blob* i_gate_out_diff_blob = BnInOp2Blob("i_gate_out_diff");
  Blob* i_out_diff_blob = BnInOp2Blob("i_out_diff");

  const Blob* c_gate_out_blob = BnInOp2Blob("c_gate_out");
  const Blob* c_out_blob = BnInOp2Blob("c_out");
  Blob* c_gate_out_diff_blob = BnInOp2Blob("c_gate_out_diff");
  Blob* c_out_diff_blob = BnInOp2Blob("c_out_diff");

  const Blob* o_gate_out_blob = BnInOp2Blob("o_gate_out");
  const Blob* o_out_blob = BnInOp2Blob("o_out");
  Blob* o_gate_out_diff_blob = BnInOp2Blob("o_gate_out_diff");
  Blob* o_out_diff_blob = BnInOp2Blob("o_out_diff");

  const Blob* cell_in_blob = BnInOp2Blob("cell_in");
  const Blob* cell_out_blob = BnInOp2Blob("cell_out");
  Blob* cell_out_diff_blob = BnInOp2Blob("cell_out_diff");
  Blob* cell_in_diff_blob = BnInOp2Blob("cell_in_diff");

  if (in_blob->col_id() != in_blob->max_col_id()) {
    // cell_out_diff = rec_out_diff * o_out * [1 -
    // tanh(cell_out)*tanh(cell_out)]
    //								+ cell_out_diff
    KernelUtil<device_type, T>::TanH(
        ctx.device_ctx, out_blob->shape().elem_cnt(), cell_out_blob->dptr<T>(),
        cell_out_blob->mut_dptr<T>());
    KernelUtil<device_type, T>::Mul(
        ctx.device_ctx, out_blob->shape().elem_cnt(), cell_out_blob->dptr<T>(),
        cell_out_blob->dptr<T>(), cell_out_blob->mut_dptr<T>());
    KernelUtil<device_type, T>::Mul(
        ctx.device_ctx, out_blob->shape().elem_cnt(), cell_out_blob->dptr<T>(),
        o_out_blob->dptr<T>(), cell_out_blob->mut_dptr<T>());
    KernelUtil<device_type, T>::Mul(
        ctx.device_ctx, out_blob->shape().elem_cnt(),
        rec_out_diff_blob->dptr<T>(), cell_out_blob->dptr<T>(),
        cell_out_blob->mut_dptr<T>());
    KernelUtil<device_type, T>::Axpy(
        ctx.device_ctx, out_blob->shape().elem_cnt(), static_cast<T>(1),
        cell_out_blob->dptr<T>(), static_cast<T>(1),
        cell_out_diff_blob->dptr<T>(), static_cast<T>(1),
        cell_out_diff_blob->mut_dptr<T>());
  } else {
    cell_out_diff_blob->CopyDataContentFrom(ctx.device_ctx, out_diff_blob);
  }
  // i_out_diff = cell_out_diff * c_out
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                  BnInOp2Blob("rec_out_diff")->dptr<T>(),
                                  c_out_blob->dptr<T>(),
                                  i_out_diff_blob->mut_dptr<T>());

  //	i_gate_out_diff = ComputeSigmoidDiff(i_out_diff)
  KernelUtil<device_type, T>::SigmoidBackward(
      ctx.device_ctx, out_blob->shape().elem_cnt(), i_gate_out_blob->dptr<T>(),
      i_out_blob->dptr<T>(), i_out_diff_blob->dptr<T>(),
      i_gate_out_diff_blob->dptr<T>());
  //	h2h_i_weight_diff = i_gate_out_diff * hidden
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       i_gate_out_diff_blob, hidden_blob,
                                       BnInOp2Blob("h2h_i_weight_diff"));
  // 	i2h_i_weight_diff = i_gate_out_diff * in
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       i_gate_out_diff_blob, BnInOp2Blob("in"),
                                       BnInOp2Blob("i2h_i_weight_diff"));
  // f_out_diff = cell_out_diff * cell_in
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                  cell_out_diff_blob->dptr<T>(),
                                  cell_in_blob->dptr<T>(),
                                  f_out_diff_blob->mut_dptr<T>());
  //	f_gate_out_diff = f_out_diff * (1 - f_out_diff)
  KernelUtil<device_type, T>::SigmoidBackward(
      ctx.device_ctx, out_blob->shape().elem_cnt(), f_gate_out_blob->dptr<T>(),
      f_out_blob->dptr<T>(), f_out_diff_blob->dptr<T>(),
      f_gate_out_diff_blob->mut_dptr<T>());
  //	h2h_f_weight_diff = f_gate_out_diff * hidden
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       f_gate_out_diff_blob, hidden_blob,
                                       BnInOp2Blob("h2h_f_weight_diff"));
  // 	i2h_f_weight_diff = f_gate_out_diff * in
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       f_gate_out_diff_blob, BnInOp2Blob("in"),
                                       BnInOp2Blob("i2h_f_weight_diff"));

  // c_out_diff = cell_out_diff * i_out
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                  cell_out_diff_blob->dptr<T>(),
                                  i_out_blob->dptr<T>(),
                                  c_out_diff_blob->mut_dptr<T>());

  KernelUtil<device_type, T>::TanHBackward(
      ctx.device_ctx, out_blob->shape().elem_cnt(), c_out_blob->dptr<T>(),
      c_gate_out_blob->dptr<T>(), c_out_diff_blob->dptr<T>(),
      c_gate_out_diff_blob->mut_dptr<T>());
  //	h2h_c_weight_diff = c_gate_out_diff * hidden
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       c_gate_out_diff_blob, hidden_blob,
                                       BnInOp2Blob("h2h_c_weight_diff"));
  // 	i2h_c_weight_diff = c_gate_out_diff * in
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       c_gate_out_diff_blob, BnInOp2Blob("in"),
                                       BnInOp2Blob("i2h_c_weight_diff"));

  // o_out_diff = rec_out_diff * tanh(cell_out)
  KernelUtil<device_type, T>::TanH(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                   cell_out_blob->dptr<T>(),
                                   cell_out_blob->mut_dptr<T>());
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                  rec_out_diff_blob->dptr<T>(),
                                  cell_out_blob->dptr<T>(),
                                  BnInOp2Blob("o_out_diff")->mut_dptr<T>());

  //	o_gate_out_diff = o_out_diff * (1 - o_out_diff)
  KernelUtil<device_type, T>::SigmoidBackward(
      ctx.device_ctx, out_blob->shape().elem_cnt(), o_out_blob->dptr<T>(),
      o_gate_out_blob->dptr<T>(), o_out_diff_blob->dptr<T>(),
      BnInOp2Blob("o_gate_out_diff")->mut_dptr<T>());
  //	h2h_o_weight_diff = o_gate_out_diff * hidden
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       o_gate_out_diff_blob, hidden_blob,
                                       BnInOp2Blob("h2h_o_weight_diff"));
  // 	i2h_o_weight_diff = o_gate_out_diff * in
  KernelUtil<device_type, T>::BlobGemm(ctx.device_ctx, CblasTrans, CblasNoTrans,
                                       static_cast<T>(1), static_cast<T>(0),
                                       o_gate_out_diff_blob, BnInOp2Blob("in"),
                                       BnInOp2Blob("i2h_o_weight_diff"));
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::VirtualInitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_lstm_conf(),
                        i2h_f_weight_initializer),
      (*random_seed_gen)(), BnInOp2Blob("i2h_f_weight"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_lstm_conf(),
                        i2h_i_weight_initializer),
      (*random_seed_gen)(), BnInOp2Blob("i2h_i_weight"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_lstm_conf(),
                        i2h_o_weight_initializer),
      (*random_seed_gen)(), BnInOp2Blob("i2h_o_weight"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_lstm_conf(),
                        i2h_c_weight_initializer),
      (*random_seed_gen)(), BnInOp2Blob("i2h_c_weight"));

  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_lstm_conf(),
                        h2h_f_weight_initializer),
      (*random_seed_gen)(), BnInOp2Blob("h2h_f_weight"));

  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_lstm_conf(),
                        h2h_i_weight_initializer),
      (*random_seed_gen)(), BnInOp2Blob("h2h_i_weight"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_lstm_conf(),
                        h2h_c_weight_initializer),
      (*random_seed_gen)(), BnInOp2Blob("h2h_c_weight"));

  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_lstm_conf(),
                        h2h_o_weight_initializer),
      (*random_seed_gen)(), BnInOp2Blob("h2h_o_weight"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_lstm_conf(), bias_f_initializer),
      (*random_seed_gen)(), BnInOp2Blob("bais_f"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_lstm_conf(), bias_i_initializer),
      (*random_seed_gen)(), BnInOp2Blob("bais_i"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_lstm_conf(), bias_o_initializer),
      (*random_seed_gen)(), BnInOp2Blob("bais_o"));
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      OF_PB_POINTER_GET(this->op_conf().basic_lstm_conf(), bias_c_initializer),
      (*random_seed_gen)(), BnInOp2Blob("bais_c"));
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::VirtualInitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* i2h_f_weight_blob = BnInOp2Blob("i2h_f_weight");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, i2h_f_weight_blob, "i2h_f_weight",
      i2h_f_weight_blob->shape().At(0), i2h_f_weight_blob->shape().Count(1));
  Blob* i2h_i_weight_blob = BnInOp2Blob("i2h_i_weight");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, i2h_i_weight_blob, "i2h_i_weight",
      i2h_i_weight_blob->shape().At(0), i2h_i_weight_blob->shape().Count(1));
  Blob* i2h_c_weight_blob = BnInOp2Blob("i2h_c_weight");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, i2h_c_weight_blob, "i2h_c_weight",
      i2h_c_weight_blob->shape().At(0), i2h_c_weight_blob->shape().Count(1));
  Blob* i2h_o_weight_blob = BnInOp2Blob("i2h_o_weight");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, i2h_o_weight_blob, "i2h_o_weight",
      i2h_o_weight_blob->shape().At(0), i2h_o_weight_blob->shape().Count(1));

  Blob* h2h_f_weight_blob = BnInOp2Blob("h2h_f_weight");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, h2h_f_weight_blob, "h2h_f_weight",
      h2h_f_weight_blob->shape().At(0), h2h_f_weight_blob->shape().Count(1));
  Blob* h2h_i_weight_blob = BnInOp2Blob("h2h_i_weight");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, h2h_i_weight_blob, "h2h_i_weight",
      h2h_i_weight_blob->shape().At(0), h2h_i_weight_blob->shape().Count(1));
  Blob* h2h_c_weight_blob = BnInOp2Blob("h2h_c_weight");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, h2h_c_weight_blob, "h2h_c_weight",
      h2h_c_weight_blob->shape().At(0), h2h_c_weight_blob->shape().Count(1));
  Blob* h2h_o_weight_blob = BnInOp2Blob("h2h_o_weight");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, h2h_c_weight_blob, "h2h_c_weight",
      h2h_o_weight_blob->shape().At(0), h2h_o_weight_blob->shape().Count(1));

  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias_f"), "bias_f",
      BnInOp2Blob("bias_f")->shape().At(0), 1);
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias_i"), "bias_i",
      BnInOp2Blob("bias_i")->shape().At(0), 1);
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias_o"), "bias_o",
      BnInOp2Blob("bias_o")->shape().At(0), 1);
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias_c"), "bias_c",
      BnInOp2Blob("bias_c")->shape().At(0), 1);

  if (NeedExternalH0()) {
    KernelUtil<device_type, T>::InitialzeWithModelDir(
        ctx, part_id, part_num, model_load_dir, BnInOp2Blob("h0"), "h0",
        BnInOp2Blob("h0")->shape().At(0), 1);
  }
  if (NeedExternalC0()) {
    KernelUtil<device_type, T>::InitializeWithModelDir(
        ctx, part_id, part_num, model_load_dir, BnInOp2Blob("c0"), "c0",
        BnInOp2Blob("c0")->shape().At(0), 1);
  }
}

template<DeviceType device_type, typename T>
void BasicLstmKernel<device_type, T>::InitPureModelTmpBlobs(
    DeviceCtx* ctx,
    std::function<Blob*(const std ::string&)> BnInOp2Blob) const {
  InitializerConf bias_f_multiplier_fill_conf;
  bias_f_multiplier_fill_conf.mutable_constant_conf()->set_value(1.f);
  KernelUtil<device_type, T>::Initialize(ctx, bias_f_multiplier_fill_conf, 0,
                                         BnInOp2Blob("bias_f_multiplier"));

  InitializerConf bias_i_multiplier_fill_conf;
  bias_i_multiplier_fill_conf.mutable_constant_conf()->set_value(1.f);
  KernelUtil<device_type, T>::Initialize(ctx, bias_i_multiplier_fill_conf, 0,
                                         BnInOp2Blob("bias_i_multiplier"));

  InitializerConf bias_o_multiplier_fill_conf;
  bias_o_multiplier_fill_conf.mutable_constant_conf()->set_value(1.f);
  KernelUtil<device_type, T>::Initialize(ctx, bias_o_multiplier_fill_conf, 0,
                                         BnInOp2Blob("bias_o_multiplier"));

  InitializerConf bias_c_multiplier_fill_conf;
  bias_c_multiplier_fill_conf.mutable_constant_conf()->set_value(1.f);
  KernelUtil<device_type, T>::Initialize(ctx, bias_c_multiplier_fill_conf, 0,
                                         BnInOp2Blob("bias_c_multiplier"));
}

}  // namespace oneflow
