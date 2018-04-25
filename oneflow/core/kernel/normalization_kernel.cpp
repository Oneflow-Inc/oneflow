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

}  // namespace

NormalizationCtx::NormalizationCtx(const KernelConf& kernel_conf,
                                   DataType type) {
#ifdef WITH_CUDA
  const NormalizationKernelConf& conf = kernel_conf.normalization_conf();
  mode_ = static_cast<cudnnBatchNormMode_t>(conf.cudnn_bn_mode());
  std::vector<int64_t> in_shape(conf.in().dim().begin(), conf.in().dim().end());
  CHECK(4 <= in_shape.size() && in_shape.size() <= 5) << in_shape.size();
  int32_t axis = kernel_conf.op_conf().normalization_conf().axis();
  param_desc_.reset(new CudnnTensorDesc());
  int N = in_shape[0];
  int C = in_shape[axis];
  int H = in_shape[axis == 1 ? 2 : 1];
  int W = in_shape[axis == 1 ? 3 : 2];
  int D = in_shape.size() > 4 ? in_shape[axis == 1 ? 4 : 3] : 1;
  std::vector<int> dims = {N, C, H, W, D};
  std::vector<int> strides;
  if (axis == 1) {
    strides = {C * H * W * D, H * W * D, W * D, D, 1};
  } else {
    strides = {H * W * D * C, 1, W * D * C, D * C, C};
  }
  in_desc_.reset(
      new CudnnTensorDesc(type, in_shape.size(), dims.data(), strides.data()));
  CudaCheck(cudnnDeriveBNTensorDescriptor(param_desc_->Get(), in_desc_->Get(),
                                          mode_));
#endif  // WITH_CUDA
}
#ifdef WITH_CUDA
const cudnnBatchNormMode_t& NormalizationCtx::cudnn_batch_norm_mode() const {
  return mode_;
}
const cudnnTensorDescriptor_t& NormalizationCtx::cudnn_in_tensor_desc() const {
  return in_desc_->Get();
}
const cudnnTensorDescriptor_t& NormalizationCtx::cudnn_param_tensor_desc()
    const {
  return param_desc_->Get();
}
#endif  // WITH_CUDA

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
  if (conf.scale()) {
    Blob* gamma_blob = BnInOp2Blob("gamma");
    KernelUtil<device_type, T>::InitializeWithDir(
        ctx, 0, part_num, model_load_dir, gamma_blob, "gamma",
        gamma_blob->shape().At(0), gamma_blob->shape().Count(1));
  }
  if (conf.center()) {
    Blob* beta_blob = BnInOp2Blob("beta");
    KernelUtil<device_type, T>::InitializeWithDir(
        ctx, 0, part_num, model_load_dir, beta_blob, "beta",
        beta_blob->shape().At(0), beta_blob->shape().Count(1));
  }
  Blob* mean_blob = BnInOp2Blob("moving_mean");
  KernelUtil<device_type, T>::InitializeWithDir(
      ctx, 0, part_num, model_load_dir, mean_blob, "moving_mean",
      mean_blob->shape().At(0), mean_blob->shape().Count(1));
  Blob* variance_blob = BnInOp2Blob("moving_variance");
  KernelUtil<device_type, T>::InitializeWithDir(
      ctx, 0, part_num, model_load_dir, variance_blob, "moving_variance",
      variance_blob->shape().At(0), variance_blob->shape().Count(1));
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::InitPureModelTmpBlobs(
    DeviceCtx* ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = this->kernel_conf().normalization_conf();
  if (!conf.use_cudnn()) {
    InitializerConf sum_multiplier_initializer_conf;
    sum_multiplier_initializer_conf.mutable_constant_conf()->set_value(1.0f);
    KernelUtil<device_type, T>::InitializeWithConf(
        ctx, sum_multiplier_initializer_conf, 0,
        BnInOp2Blob("after_axis_sum_multiplier"));
    KernelUtil<device_type, T>::InitializeWithConf(
        ctx, sum_multiplier_initializer_conf, 0,
        BnInOp2Blob("before_axis_sum_multiplier"));
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = this->kernel_conf().normalization_conf();
#ifdef WITH_CUDA
  if (conf.use_cudnn()) {
    NormalizationCudnnForward(ctx, BnInOp2Blob);
    return;
  }
#endif
  const Blob* mean_blob = nullptr;
  const Blob* variance_blob = nullptr;
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  if (Global<JobDesc>::Get()->IsTrain()) {
    CalcMeanAndVariance(ctx, BnInOp2Blob);
    UpdateMovingMeanAndMovingVariance(ctx, BnInOp2Blob);
    mean_blob = BnInOp2Blob("new_mean");
    variance_blob = BnInOp2Blob("new_variance");
  } else {
    mean_blob = BnInOp2Blob("moving_mean");
    variance_blob = BnInOp2Blob("moving_variance");
  }
  Normalize(ctx, BnInOp2Blob, mean_blob, variance_blob, in_blob, out_blob);
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& normalization_kernel_conf =
      this->kernel_conf().normalization_conf();
#ifdef WITH_CUDA
  if (normalization_kernel_conf.use_cudnn()) {
    NormalizationCudnnBackward(ctx, BnInOp2Blob);
    return;
  }
#endif
  const auto& normalization_op_conf = this->op_conf().normalization_conf();
  bool need_comp_in_diff = (BnInOp2Blob("in_diff") != nullptr);
  if (need_comp_in_diff || normalization_op_conf.scale()) {
    CalcAboutGammaDiff(ctx, BnInOp2Blob, need_comp_in_diff);
  }
  if (need_comp_in_diff || normalization_op_conf.center()) {
    CalcAboutBetaDiff(ctx, BnInOp2Blob, need_comp_in_diff);
  }
  if (need_comp_in_diff) { CalcInDiff(ctx, BnInOp2Blob); }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::CalcAboutGammaDiff(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob,
    bool need_comp_in_diff) const {
  Blob* normalized_blob = BnInOp2Blob("normalized_in");
  Blob* gamma_diff_blob = BnInOp2Blob("gamma_diff");
  Blob* inv_var_blob = BnInOp2Blob("inv_var");
  // it's safe to use in_diff as tmp blob
  Blob* tmp_blob = BnInOp2Blob("in_diff");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  KernelUtil<device_type, T>::Mul(
      ctx.device_ctx, out_diff_blob->shape().elem_cnt(),
      out_diff_blob->dptr<T>(), normalized_blob->dptr<T>(),
      tmp_blob->mut_dptr<T>());
  ComputeAxisSum(ctx, BnInOp2Blob, tmp_blob, gamma_diff_blob);
  if (need_comp_in_diff) {
    AxisSliceMul(ctx, normalized_blob, gamma_diff_blob, normalized_blob);
  }
  const Blob* gamma_blob = BnInOp2Blob("gamma");
  if (gamma_blob != nullptr) {
    KernelUtil<device_type, T>::Mul(
        ctx.device_ctx, inv_var_blob->shape().elem_cnt(), gamma_blob->dptr<T>(),
        inv_var_blob->dptr<T>(), inv_var_blob->mut_dptr<T>());
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::CalcAboutBetaDiff(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob,
    bool need_comp_in_diff) const {
  Blob* normalized_blob = BnInOp2Blob("normalized_in");
  Blob* beta_diff_blob = BnInOp2Blob("beta_diff");
  ComputeAxisSum(ctx, BnInOp2Blob, BnInOp2Blob("out_diff"), beta_diff_blob);
  if (need_comp_in_diff) {
    AxisSliceAdd(ctx, normalized_blob, beta_diff_blob, normalized_blob);
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::CalcInDiff(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  int64_t axis = this->op_conf().normalization_conf().axis();
  Blob* normalized_blob = BnInOp2Blob("normalized_in");
  Blob* inv_var_blob = BnInOp2Blob("inv_var");
  KernelUtil<device_type, T>::Scal(
      ctx.device_ctx, normalized_blob->shape().elem_cnt(),
      static_cast<T>(-1.0 * normalized_blob->shape().At(axis)
                     / normalized_blob->shape().elem_cnt()),
      normalized_blob->mut_dptr<T>(), 1);
  KernelUtil<device_type, T>::Axpy(
      ctx.device_ctx, normalized_blob->shape().elem_cnt(), static_cast<T>(1),
      BnInOp2Blob("out_diff")->dptr<T>(), 1, normalized_blob->mut_dptr<T>(), 1);
  AxisSliceMul(ctx, normalized_blob, inv_var_blob, normalized_blob);
  BnInOp2Blob("in_diff")->CopyDataContentFrom(ctx.device_ctx, normalized_blob);
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::Normalize(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob,
    const Blob* mean_blob, const Blob* variance_blob, const Blob* in_blob,
    Blob* out_blob) const {
  const auto& normalization_op_conf = this->op_conf().normalization_conf();
  const bool scale = normalization_op_conf.scale();
  const bool center = normalization_op_conf.center();
  Blob* inv_var_blob = BnInOp2Blob("inv_var");
  Blob* normalized_blob = BnInOp2Blob("normalized_in");
  Rsqrt<device_type, T>(ctx.device_ctx, variance_blob->shape().elem_cnt(),
                        variance_blob->dptr<T>(),
                        normalization_op_conf.epsilon(),
                        inv_var_blob->mut_dptr<T>());
  AxisSliceSub(ctx, in_blob, mean_blob, normalized_blob);
  AxisSliceMul(ctx, normalized_blob, inv_var_blob, normalized_blob);
  out_blob->CopyDataContentFrom(ctx.device_ctx, normalized_blob);
  if (scale) {
    const Blob* gamma_blob = BnInOp2Blob("gamma");
    AxisSliceMul(ctx, out_blob, gamma_blob, out_blob);
  }
  if (center) {
    const Blob* beta_blob = BnInOp2Blob("beta");
    AxisSliceAdd(ctx, out_blob, beta_blob, out_blob);
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::CalcMeanAndVariance(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* mean_blob = BnInOp2Blob("new_mean");
  ComputeAxisMean(ctx, BnInOp2Blob, in_blob, mean_blob);
  //  It's safe to use `out' as tmp blob
  Blob* tmp_blob = BnInOp2Blob("out");
  Blob* variance_blob = BnInOp2Blob("new_variance");
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, in_blob->shape().elem_cnt(),
                                  in_blob->dptr<T>(), in_blob->dptr<T>(),
                                  tmp_blob->mut_dptr<T>());
  ComputeAxisMean(ctx, BnInOp2Blob, tmp_blob, variance_blob);
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
  InitMovingMeanAndMovingVariance(ctx, BnInOp2Blob, true);
  const auto& conf = this->op_conf().normalization_conf();
  const Blob* mean_blob = BnInOp2Blob("new_mean");
  const Blob* variance_blob = BnInOp2Blob("new_variance");
  Blob* moving_mean_blob = BnInOp2Blob("moving_mean");
  Blob* moving_variance_blob = BnInOp2Blob("moving_variance");
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
void NormalizationKernel<device_type, T>::InitMovingMeanAndMovingVariance(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob,
    bool use_new) const {
  const auto& conf = this->op_conf().normalization_conf();
  auto tpl = reinterpret_cast<
      std::tuple<int64_t, std::function<const Blob*(const std::string&)>>*>(
      ctx.other);
  int64_t piece_id = std::get<0>(*tpl);
  std::function<const Blob*(const std::string&)> lbn2preblob =
      std::get<1>(*tpl);
  Blob* moving_mean_blob = BnInOp2Blob("moving_mean");
  Blob* moving_variance_blob = BnInOp2Blob("moving_variance");
  if (this->op_conf().model_load_dir() == ""
      && (Global<SnapshotMgr>::Get() == nullptr
          || Global<SnapshotMgr>::Get()->GetReadableSnapshot() == nullptr)
      && piece_id == 0) {
    if (use_new) {
      if (conf.use_first_piece_init_moving()) {
        const Blob* mean_blob = BnInOp2Blob("new_mean");
        const Blob* variance_blob = BnInOp2Blob("new_variance");
        moving_mean_blob->CopyDataContentFrom(ctx.device_ctx, mean_blob);
        moving_variance_blob->CopyDataContentFrom(ctx.device_ctx,
                                                  variance_blob);
        return;
      }
    } else {
      Memset<device_type>(ctx.device_ctx, moving_mean_blob->mut_dptr<T>(), 0,
                          moving_mean_blob->ByteSizeOfDataContentField());
      Memset<device_type>(ctx.device_ctx, moving_variance_blob->mut_dptr<T>(),
                          0, moving_mean_blob->ByteSizeOfDataContentField());
      return;
    }
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
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::ComputeAxisSum(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob,
    const Blob* x_blob, Blob* y_blob, const T alpha) const {
  int64_t axis = this->op_conf().normalization_conf().axis();
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
void NormalizationKernel<device_type, T>::ComputeAxisSum(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob,
    const Blob* x_blob, Blob* y_blob) const {
  return ComputeAxisSum(ctx, BnInOp2Blob, x_blob, y_blob, 1.f);
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::ComputeAxisMean(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob,
    const Blob* x_blob, Blob* y_blob) const {
  int64_t axis = this->op_conf().normalization_conf().axis();
  const T alpha = 1.f / (x_blob->shape().elem_cnt() / x_blob->shape().At(axis));
  return ComputeAxisSum(ctx, BnInOp2Blob, x_blob, y_blob, alpha);
}

template<DeviceType device_type, typename T>
template<void (*handler)(DeviceCtx*, const size_t, const size_t, const size_t,
                         const T*, const T*, T*)>
void NormalizationKernel<device_type, T>::AxisSliceDo(const KernelCtx& ctx,
                                                      const Blob* x_blob,
                                                      const Blob* y_blob,
                                                      Blob* z_blob) const {
  int64_t axis = this->op_conf().normalization_conf().axis();
  size_t before_axis_dim_size = x_blob->shape().CountBeforeAxis(axis);
  size_t axis_dim_size = x_blob->shape().At(axis);
  size_t after_axis_dim_size = x_blob->shape().CountAfterAxis(axis);
  CHECK_EQ(axis_dim_size, y_blob->shape().elem_cnt());
  CHECK_EQ(x_blob->shape(), z_blob->shape());
  handler(ctx.device_ctx, before_axis_dim_size, axis_dim_size,
          after_axis_dim_size, x_blob->dptr<T>(), y_blob->dptr<T>(),
          z_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::AxisSliceAdd(const KernelCtx& ctx,
                                                       const Blob* x_blob,
                                                       const Blob* y_blob,
                                                       Blob* z_blob) const {
  AxisSliceDo<&KernelUtil<device_type, T>::AxisSliceAdd>(ctx, x_blob, y_blob,
                                                         z_blob);
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::AxisSliceSub(const KernelCtx& ctx,
                                                       const Blob* x_blob,
                                                       const Blob* y_blob,
                                                       Blob* z_blob) const {
  AxisSliceDo<&KernelUtil<device_type, T>::AxisSliceSub>(ctx, x_blob, y_blob,
                                                         z_blob);
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::AxisSliceMul(const KernelCtx& ctx,
                                                       const Blob* x_blob,
                                                       const Blob* y_blob,
                                                       Blob* z_blob) const {
  AxisSliceDo<&KernelUtil<device_type, T>::AxisSliceMul>(ctx, x_blob, y_blob,
                                                         z_blob);
}

template<DeviceType device_type, typename T>
const PbMessage& NormalizationKernel<device_type, T>::GetCustomizedOpConf()
    const {
  return this->op_conf().normalization_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kNormalizationConf,
                           NormalizationKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
