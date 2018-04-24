#include "oneflow/core/kernel/normalization_kernel.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/kernel/transpose_kernel.h"
#include "oneflow/core/operator/normalization_op.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
void Rsqrt(DeviceCtx* ctx, const int64_t n, const T* x, const float epsilon,
           T* y) {
  KernelUtil<device_type, T>::Copy(ctx, n, x, 1, y, 1);
  KernelUtil<device_type, T>::Rsqrt(ctx, n, y, epsilon);
}

template<DeviceType device_type, typename T>
void ScalarSub(DeviceCtx* ctx, const int64_t n, const T* x, const T* scalar_ptr,
               T* y) {
  KernelUtil<device_type, T>::Copy(ctx, n, x, 1, y, 1);
  KernelUtil<device_type, T>::Axpy(ctx, n, static_cast<T>(-1), scalar_ptr, 0, y,
                                   1);
}

template<typename T>
T* GetTmpForSumDptr(Blob* tmp_storage_blob, T* nop_addr) {
  if (tmp_storage_blob != nullptr) {
    return tmp_storage_blob->mut_dptr<T>();
  } else {
    return nop_addr;
  }
}
size_t GetTmpForSumByteSize(Blob* tmp_storage_blob) {
  if (tmp_storage_blob != nullptr) {
    return tmp_storage_blob->ByteSizeOfDataContentField();
  } else {
    return 0;
  }
}

template<DeviceType device_type, typename T>
void ComputeAxisSum(const KernelCtx& ctx,
                    const std::function<Blob*(const std::string&)>& BnInOp2Blob,
                    const Blob* x_blob, Blob* y_blob, int64_t axis,
                    const T alpha) {
  int num_before_axis_dim = x_blob->shape().CountBeforeAxis(axis);
  int num_axis_dim = x_blob->shape().At(axis);
  int num_after_axis_dim = x_blob->shape().CountAfterAxis(axis);
  CHECK_EQ(y_blob->shape().elem_cnt(), num_axis_dim);
  Blob* before_by_axis_blob = BnInOp2Blob("before_by_axis_matrix");
  const Blob* after_axis_sum_multiplier =
      BnInOp2Blob("after_axis_sum_multiplier");
  KernelUtil<device_type, T>::Gemv(
      ctx.device_ctx, CblasTrans, num_before_axis_dim * num_axis_dim,
      num_after_axis_dim, alpha, x_blob->dptr<T>(), num_after_axis_dim,
      after_axis_sum_multiplier->dptr<T>(), 1, 0.0,
      before_by_axis_blob->mut_dptr<T>(), 1);
  const Blob* before_axis_sum_multiplier =
      BnInOp2Blob("before_axis_sum_multiplier");
  KernelUtil<device_type, T>::Gemv(
      ctx.device_ctx, CblasNoTrans, num_before_axis_dim, num_axis_dim, 1.0,
      before_by_axis_blob->dptr<T>(), num_axis_dim,
      before_axis_sum_multiplier->dptr<T>(), 1, 0.0, y_blob->mut_dptr<T>(), 1);
}

template<DeviceType device_type, typename T>
void ComputeAxisSum(const KernelCtx& ctx,
                    const std::function<Blob*(const std::string&)>& BnInOp2Blob,
                    const Blob* x_blob, Blob* y_blob, int64_t axis) {
  return ComputeAxisSum<device_type, T>(ctx, BnInOp2Blob, x_blob, y_blob, axis,
                                        1.f);
}

template<DeviceType device_type, typename T>
void ComputeAxisMean(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob,
    const Blob* x_blob, Blob* y_blob, int64_t axis) {
  const T alpha = 1.f / (x_blob->shape().elem_cnt() / x_blob->shape().At(axis));
  return ComputeAxisSum<device_type, T>(ctx, BnInOp2Blob, x_blob, y_blob, axis,
                                        alpha);
}

}  // namespace

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& normalization_conf = this->op_conf().normalization_conf();
  if (normalization_conf.scale()) {
    InitializerConf gamma_init_conf;
    float gamma_init = normalization_conf.gamma_init();
    gamma_init_conf.mutable_constant_conf()->set_value(gamma_init);
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx, &gamma_init_conf, 0, BnInOp2Blob("gamma"));
  }
  if (normalization_conf.center()) {
    InitializerConf beta_init_conf;
    float beta_init = normalization_conf.beta_init();
    beta_init_conf.mutable_constant_conf()->set_value(beta_init);
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx, &beta_init_conf, 0, BnInOp2Blob("beta"));
  }
  float mean_init = normalization_conf.mean_init();
  InitializerConf moving_mean_init_conf;
  moving_mean_init_conf.mutable_constant_conf()->set_value(mean_init);
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx, &moving_mean_init_conf, 0, BnInOp2Blob("moving_mean"));
  float variance_init = normalization_conf.variance_init();
  InitializerConf moving_variance_init_conf;
  moving_variance_init_conf.mutable_constant_conf()->set_value(variance_init);
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx, &moving_variance_init_conf, 0, BnInOp2Blob("moving_variance"));
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = this->op_conf().normalization_conf();
  int32_t dim_num = this->kernel_conf().normalization_conf().transpose_cols();
  if (conf.scale()) {
    Blob* gamma_blob = BnInOp2Blob("gamma");
    KernelUtil<device_type, T>::InitializeWithDir(
        ctx, 0, part_num, model_load_dir, gamma_blob, "gamma", dim_num,
        gamma_blob->shape().Count(1));
  }
  if (conf.center()) {
    Blob* beta_blob = BnInOp2Blob("beta");
    KernelUtil<device_type, T>::InitializeWithDir(
        ctx, 0, part_num, model_load_dir, beta_blob, "beta", dim_num,
        beta_blob->shape().Count(1));
  }
  Blob* mean_blob = BnInOp2Blob("moving_mean");
  KernelUtil<device_type, T>::InitializeWithDir(
      ctx, 0, part_num, model_load_dir, mean_blob, "moving_mean", dim_num,
      mean_blob->shape().Count(1));
  Blob* variance_blob = BnInOp2Blob("moving_variance");
  KernelUtil<device_type, T>::InitializeWithDir(
      ctx, 0, part_num, model_load_dir, variance_blob, "moving_variance",
      dim_num, variance_blob->shape().Count(1));
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::InitPureModelTmpBlobs(
    DeviceCtx* ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitializerConf sum_multiplier_initializer_conf;
  sum_multiplier_initializer_conf.mutable_constant_conf()->set_value(1.0f);
  KernelUtil<device_type, T>::InitializeWithConf(
      ctx, sum_multiplier_initializer_conf, 0,
      BnInOp2Blob("after_axis_sum_multiplier"));
  KernelUtil<device_type, T>::InitializeWithConf(
      ctx, sum_multiplier_initializer_conf, 0,
      BnInOp2Blob("before_axis_sum_multiplier"));
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = this->kernel_conf().normalization_conf();
  const Blob* mean_blob = nullptr;
  const Blob* variance_blob = nullptr;
  const Blob* comp_in_blob = nullptr;
  Blob* comp_out_blob = nullptr;
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* trans_in_blob = BnInOp2Blob("trans_in");
  Blob* trans_out_blob = BnInOp2Blob("trans_out");
  if (conf.need_transpose()) {
    Transpose<device_type, T>(ctx.device_ctx, in_blob, trans_in_blob,
                              conf.perm());
    comp_in_blob = trans_in_blob;
    comp_out_blob = trans_out_blob;
  } else {
    comp_in_blob = in_blob;
    comp_out_blob = out_blob;
  }
  if (Global<JobDesc>::Get()->IsTrain()) {
    CalcMeanAndVariance(ctx, BnInOp2Blob);
    UpdateMovingMeanAndMovingVariance(ctx, BnInOp2Blob);
    mean_blob = BnInOp2Blob("new_mean");
    variance_blob = BnInOp2Blob("new_variance");
  } else {
    mean_blob = BnInOp2Blob("moving_mean");
    variance_blob = BnInOp2Blob("moving_variance");
  }
  Normalize(ctx, BnInOp2Blob, mean_blob, variance_blob, comp_in_blob,
            comp_out_blob);
  if (conf.need_transpose()) {
    Transpose<device_type, T>(ctx.device_ctx, trans_out_blob, out_blob,
                              conf.perm());
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& normalization_op_conf = this->op_conf().normalization_conf();
  const auto& normalization_kernel_conf =
      this->kernel_conf().normalization_conf();
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* comp_out_diff_blob = nullptr;
  Blob* comp_in_diff_blob = nullptr;
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  Blob* trans_in_blob = BnInOp2Blob("trans_in");
  Blob* trans_out_blob = BnInOp2Blob("trans_out");
  bool need_transpose = normalization_kernel_conf.need_transpose();
  bool need_comp_in_diff = (in_diff_blob != nullptr);
  if (need_transpose) {
    Transpose<device_type, T>(ctx.device_ctx, out_diff_blob, trans_out_blob,
                              normalization_kernel_conf.perm());
    comp_in_diff_blob = trans_in_blob;
    comp_out_diff_blob = trans_out_blob;
  } else {
    comp_in_diff_blob = in_diff_blob;
    comp_out_diff_blob = out_diff_blob;
  }
  if (need_comp_in_diff || normalization_op_conf.scale()) {
    CalcAboutGammaDiff(ctx, BnInOp2Blob, comp_out_diff_blob, need_comp_in_diff);
  }
  if (need_comp_in_diff || normalization_op_conf.center()) {
    CalcAboutBetaDiff(ctx, BnInOp2Blob, need_comp_in_diff);
  }
  if (need_comp_in_diff) {
    CalcInDiff(ctx, BnInOp2Blob, comp_out_diff_blob, comp_in_diff_blob);
    if (need_transpose)
      Transpose<device_type, T>(ctx.device_ctx, trans_in_blob, in_diff_blob,
                                normalization_kernel_conf.perm());
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::CalcAboutGammaDiff(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)> BnInOp2Blob,
    const Blob* out_diff_blob, bool need_comp_in_diff) const {
  const auto& conf = this->kernel_conf().normalization_conf();
  const int32_t norm_part_num = conf.transpose_cols();
  const int64_t norm_elem_num = conf.transpose_rows();
  Blob* normalized_blob = BnInOp2Blob("normalized_in");
  Blob* gamma_diff_blob = BnInOp2Blob("gamma_diff");
  Blob* inv_var_blob = BnInOp2Blob("inv_var");
  // it's safe to use in_diff as tmp_blob
  Blob* tmp_blob = BnInOp2Blob("in_diff");
  Blob* raw_out_diff_blob = BnInOp2Blob("out_diff");
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, raw_out_diff_blob->shape().elem_cnt(),
      raw_out_diff_blob->dptr<T>(), normalized_blob->dptr<T>(),
      tmp_blob->mut_dptr<T>());
  ComputeAxisSum<device_type, T>(ctx, BnInOp2Blob, tmp_blob, gamma_diff_blob,
                                 conf.axis());
  FOR_RANGE(int32_t, i, 0, norm_part_num) {
    //    KernelUtil<device_type, T>::Dot(
    //        ctx.device_ctx, norm_elem_num,
    //        out_diff_blob->dptr<T>() + i * norm_elem_num, 1,
    //        normalized_blob->dptr<T>() + i * norm_elem_num, 1,
    //        gamma_diff_blob->mut_dptr<T>() + i);
    if (need_comp_in_diff) {
      KernelUtil<device_type, T>::Scal(
          ctx.device_ctx, norm_elem_num, gamma_diff_blob->dptr<T>() + i,
          normalized_blob->mut_dptr<T>() + i * norm_elem_num, 1);
    }
  }
  const Blob* gamma_blob = BnInOp2Blob("gamma");
  if (gamma_blob != nullptr) {
    KernelUtil<device_type, T>::Mul(
        ctx.device_ctx, norm_part_num, gamma_blob->dptr<T>(),
        inv_var_blob->dptr<T>(), inv_var_blob->mut_dptr<T>());
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::CalcAboutBetaDiff(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)> BnInOp2Blob,
    bool need_comp_in_diff) const {
  const auto& conf = this->kernel_conf().normalization_conf();
  const auto& op_conf = this->kernel_conf().op_conf().normalization_conf();
  const int32_t norm_part_num = conf.transpose_cols();
  const int64_t norm_elem_num = conf.transpose_rows();
  Blob* normalized_blob = BnInOp2Blob("normalized_in");
  Blob* beta_diff_blob = BnInOp2Blob("beta_diff");
  ComputeAxisSum<device_type, T>(ctx, BnInOp2Blob, BnInOp2Blob("out_diff"),
                                 beta_diff_blob, op_conf.axis());
  FOR_RANGE(int32_t, i, 0, norm_part_num) {
    if (need_comp_in_diff) {
      KernelUtil<device_type, T>::Axpy(
          ctx.device_ctx, norm_elem_num, static_cast<T>(1),
          beta_diff_blob->dptr<T>() + i, 0,
          normalized_blob->mut_dptr<T>() + i * norm_elem_num, 1);
    }
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::CalcInDiff(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)> BnInOp2Blob,
    const Blob* out_diff_blob, Blob* in_diff_blob) const {
  const auto& normalization_kernel_conf =
      this->kernel_conf().normalization_conf();
  const int32_t norm_part_num = normalization_kernel_conf.transpose_cols();
  const int64_t norm_elem_num = normalization_kernel_conf.transpose_rows();
  Blob* normalized_blob = BnInOp2Blob("normalized_in");
  Blob* inv_var_blob = BnInOp2Blob("inv_var");
  KernelUtil<device_type, T>::Scal(
      ctx.device_ctx, normalized_blob->shape().elem_cnt(),
      static_cast<T>(-1.0 / norm_elem_num), normalized_blob->mut_dptr<T>(), 1);
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, normalized_blob->shape().elem_cnt(), static_cast<T>(1),
      out_diff_blob->dptr<T>(), 1, normalized_blob->mut_dptr<T>(), 1);
  FOR_RANGE(int32_t, i, 0, norm_part_num) {
    KernelUtil<device_type, T>::Scal(
        ctx.device_ctx, norm_elem_num, inv_var_blob->dptr<T>() + i,
        normalized_blob->mut_dptr<T>() + i * norm_elem_num, 1);
  }
  in_diff_blob->CopyDataContentFrom(ctx.device_ctx, normalized_blob);
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::Normalize(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob,
    const Blob* mean_blob, const Blob* variance_blob, const Blob* in_blob,
    Blob* out_blob) const {
  const auto& normalization_op_conf = this->op_conf().normalization_conf();
  const auto& normalization_kernel_conf =
      this->kernel_conf().normalization_conf();
  const int32_t norm_part_num = normalization_kernel_conf.transpose_cols();
  const int64_t norm_elem_num = normalization_kernel_conf.transpose_rows();
  const bool scale = normalization_op_conf.scale();
  const bool center = normalization_op_conf.center();
  Blob* inv_var_blob = BnInOp2Blob("inv_var");
  Blob* normalized_blob = BnInOp2Blob("normalized_in");
  Rsqrt<device_type, T>(ctx.device_ctx, norm_part_num, variance_blob->dptr<T>(),
                        normalization_op_conf.epsilon(),
                        inv_var_blob->mut_dptr<T>());
  FOR_RANGE(int32_t, i, 0, norm_part_num) {
    ScalarSub<device_type, T>(
        ctx.device_ctx, norm_elem_num, in_blob->dptr<T>() + i * norm_elem_num,
        mean_blob->dptr<T>() + i,
        normalized_blob->mut_dptr<T>() + i * norm_elem_num);
    KernelUtil<device_type, T>::Scal(
        ctx.device_ctx, norm_elem_num, inv_var_blob->dptr<T>() + i,
        normalized_blob->mut_dptr<T>() + i * norm_elem_num, 1);
  }
  out_blob->CopyDataContentFrom(ctx.device_ctx, normalized_blob);
  if (scale) {
    const Blob* gamma_blob = BnInOp2Blob("gamma");
    FOR_RANGE(int32_t, i, 0, norm_part_num) {
      KernelUtil<device_type, T>::Scal(
          ctx.device_ctx, norm_elem_num, gamma_blob->dptr<T>() + i,
          out_blob->mut_dptr<T>() + i * norm_elem_num, 1);
    }
  }
  if (center) {
    const Blob* beta_blob = BnInOp2Blob("beta");
    FOR_RANGE(int32_t, i, 0, norm_part_num) {
      KernelUtil<device_type, T>::Axpy(
          ctx.device_ctx, norm_elem_num, static_cast<T>(1),
          beta_blob->dptr<T>() + i, 0,
          out_blob->mut_dptr<T>() + i * norm_elem_num, 1);
    }
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::CalcMeanAndVariance(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const auto& op_conf = this->kernel_conf().op_conf().normalization_conf();
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* mean_blob = BnInOp2Blob("new_mean");
  ComputeAxisMean<device_type, T>(ctx, BnInOp2Blob, in_blob, mean_blob,
                                  op_conf.axis());
  //  It's safe to use `out' as tmp blob
  Blob* tmp_blob = BnInOp2Blob("out");
  Blob* variance_blob = BnInOp2Blob("new_variance");
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, in_blob->shape().elem_cnt(),
                                  in_blob->dptr<T>(), in_blob->dptr<T>(),
                                  tmp_blob->mut_dptr<T>());
  ComputeAxisMean<device_type, T>(ctx, BnInOp2Blob, tmp_blob, variance_blob,
                                  op_conf.axis());
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, mean_blob->shape().elem_cnt(),
                                  mean_blob->dptr<T>(), mean_blob->dptr<T>(),
                                  tmp_blob->mut_dptr<T>());
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, variance_blob->shape().elem_cnt(), -1.f,
      tmp_blob->dptr<T>(), 1, variance_blob->mut_dptr<T>(), 1);
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::UpdateMovingMeanAndMovingVariance(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const auto& conf = this->op_conf().normalization_conf();
  auto tpl = reinterpret_cast<
      std::tuple<int64_t, std::function<const Blob*(const std::string&)>>*>(
      ctx.other);
  int64_t piece_id = std::get<0>(*tpl);
  std::function<const Blob*(const std::string&)> lbn2preblob =
      std::get<1>(*tpl);
  const Blob* mean_blob = BnInOp2Blob("new_mean");
  const Blob* variance_blob = BnInOp2Blob("new_variance");
  Blob* moving_mean_blob = BnInOp2Blob("moving_mean");
  Blob* moving_variance_blob = BnInOp2Blob("moving_variance");
  if (conf.use_first_piece_init_moving() && piece_id == 0) {
    moving_mean_blob->CopyDataContentFrom(ctx.device_ctx, mean_blob);
    moving_variance_blob->CopyDataContentFrom(ctx.device_ctx, variance_blob);
    return;
  }
  const Blob* pre_moving_mean_blob =
      lbn2preblob(this->Lbn4BnInOp("moving_mean"));
  if (pre_moving_mean_blob != moving_mean_blob) {
    moving_mean_blob->CopyDataContentFrom(ctx.device_ctx, pre_moving_mean_blob);
  }
  const Blob* pre_moving_variance_blob =
      lbn2preblob(this->Lbn4BnInOp("moving_variance"));
  if (pre_moving_variance_blob != moving_variance_blob) {
    moving_variance_blob->CopyDataContentFrom(ctx.device_ctx,
                                              pre_moving_variance_blob);
  }
  const T momentum = conf.momentum();
  const T one_minus_momentum = 1 - momentum;
  KernelUtil<device_type, T>::Scal(
      ctx.device_ctx, moving_mean_blob->shape().elem_cnt(), momentum,
      moving_mean_blob->mut_dptr<T>(), 1);
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, mean_blob->shape().elem_cnt(), one_minus_momentum,
      mean_blob->dptr<T>(), 1, moving_mean_blob->mut_dptr<T>(), 1);
  KernelUtil<device_type, T>::Scal(
      ctx.device_ctx, moving_variance_blob->shape().elem_cnt(), momentum,
      moving_variance_blob->mut_dptr<T>(), 1);
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, variance_blob->shape().elem_cnt(), one_minus_momentum,
      variance_blob->dptr<T>(), 1, moving_variance_blob->mut_dptr<T>(), 1);
}

template<DeviceType device_type, typename T>
const PbMessage& NormalizationKernel<device_type, T>::GetCustomizedOpConf()
    const {
  return this->op_conf().normalization_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kNormalizationConf,
                           NormalizationKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
