#include "oneflow/core/kernel/local_response_normalization_kernel.h"
#include "oneflow/core/common/eigen_util.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

#ifdef WITH_CUDA
CudnnLRNDesc::CudnnLRNDesc(unsigned depth_radius, double alpha, double beta, double bias) {
  CudaCheck(cudnnCreateLRNDescriptor(&val_));
  CudaCheck(cudnnSetLRNDescriptor(val_, depth_radius, alpha, beta, bias));
}

CudnnLRNDesc::~CudnnLRNDesc() { CudaCheck(cudnnDestroyLRNDescriptor(val_)); }
#endif  // WITH_CUDA

template<typename T>
void LocalResponseNormalizationKernel<DeviceType::kCPU, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = kernel_conf().op_attribute().op_conf().local_response_normalization_conf();
  if (conf.data_format() == "channels_first") {
    NCHWForward(ctx, BnInOp2Blob);
  } else if (conf.data_format() == "channels_last") {
    NHWCForward(ctx, BnInOp2Blob);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void LocalResponseNormalizationKernel<DeviceType::kCPU, T>::NCHWForward(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const LocalResponseNormalizationOpConf& lrn_conf =
      this->op_conf().local_response_normalization_conf();
  const Blob* in_blob = BnInOp2Blob("in");
  const ShapeView& in_shape = in_blob->shape();
  Blob* out_blob = BnInOp2Blob("out");
  Blob* ps_blob = BnInOp2Blob("padded_square");
  Blob* nc_blob = BnInOp2Blob("normalize_coef");
  Memset<DeviceType::kCPU>(ctx.device_ctx, ps_blob->mut_dptr(), 0, ps_blob->ByteSizeOfBlobBody());

  T* ps_dptr = ps_blob->mut_dptr<T>();
  T* nc_dptr = nc_blob->mut_dptr<T>();
  FOR_RANGE(int64_t, i, 0, in_shape.elem_cnt()) { nc_dptr[i] = lrn_conf.bias(); }
  int64_t image_size = in_shape.Count(1);
  int64_t channel_size = in_shape.Count(2);
  int64_t size = 2 * lrn_conf.depth_radius() + 1;
  T alpha_over_n = lrn_conf.alpha() / size;

  FOR_RANGE(int64_t, n, 0, in_shape.At(0)) {
    const T* in_dptr = in_blob->dptr<T>() + image_size * n;
    KernelUtil<DeviceType::kCPU, T>::Mul(ctx.device_ctx, image_size, in_dptr, in_dptr,
                                         ps_dptr + lrn_conf.depth_radius() * channel_size);
    FOR_RANGE(int64_t, c, 0, size) {
      KernelUtil<DeviceType::kCPU, T>::Axpy(ctx.device_ctx, channel_size, alpha_over_n,
                                            ps_dptr + c * channel_size, 1, nc_dptr, 1);
    }
    FOR_RANGE(int64_t, c, 1, in_shape.At(1)) {
      KernelUtil<DeviceType::kCPU, T>::Copy(ctx.device_ctx, channel_size,
                                            nc_dptr + (c - 1) * channel_size, 1,
                                            nc_dptr + c * channel_size, 1);
      KernelUtil<DeviceType::kCPU, T>::Axpy(ctx.device_ctx, channel_size, alpha_over_n,
                                            ps_dptr + (c + size - 1) * channel_size, 1,
                                            nc_dptr + c * channel_size, 1);
      KernelUtil<DeviceType::kCPU, T>::Axpy(ctx.device_ctx, channel_size, -alpha_over_n,
                                            ps_dptr + (c - 1) * channel_size, 1,
                                            nc_dptr + c * channel_size, 1);
    }
    nc_dptr += image_size;
  }
  KernelUtil<DeviceType::kCPU, T>::Powx(ctx.device_ctx, in_shape.elem_cnt(), nc_blob->dptr<T>(),
                                        -lrn_conf.beta(), out_blob->mut_dptr<T>());
  KernelUtil<DeviceType::kCPU, T>::Mul(ctx.device_ctx, in_shape.elem_cnt(), in_blob->dptr<T>(),
                                       out_blob->dptr<T>(), out_blob->mut_dptr<T>());
}

template<typename T>
void LocalResponseNormalizationKernel<DeviceType::kCPU, T>::NCHWBackward(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const LocalResponseNormalizationOpConf& lrn_conf =
      this->op_conf().local_response_normalization_conf();
  const Blob* in_blob = BnInOp2Blob("in");
  const ShapeView& in_shape = in_blob->shape();
  const Blob* out_blob = BnInOp2Blob("out");
  Blob* ps_blob = BnInOp2Blob("padded_square");
  Memset<DeviceType::kCPU>(ctx.device_ctx, ps_blob->mut_dptr(), 0, ps_blob->ByteSizeOfBlobBody());

  const T* out_diff_dptr = BnInOp2Blob("out_diff")->dptr<T>();
  T* in_diff_dptr = BnInOp2Blob("in_diff")->mut_dptr<T>();
  T* nc_dptr = BnInOp2Blob("normalize_coef")->mut_dptr<T>();
  T* ps_dptr = ps_blob->mut_dptr<T>();

  int64_t image_size = in_shape.Count(1);
  int64_t channel_size = in_shape.Count(2);
  int64_t size = 2 * lrn_conf.depth_radius() + 1;
  T cache_ratio_value = 2. * lrn_conf.beta() * lrn_conf.alpha() / size;

  KernelUtil<DeviceType::kCPU, T>::Powx(ctx.device_ctx, in_shape.elem_cnt(), nc_dptr,
                                        -lrn_conf.beta(), in_diff_dptr);
  KernelUtil<DeviceType::kCPU, T>::Mul(ctx.device_ctx, in_shape.elem_cnt(), out_diff_dptr,
                                       in_diff_dptr, in_diff_dptr);

  FOR_RANGE(int64_t, n, 0, in_shape.At(0)) {
    KernelUtil<DeviceType::kCPU, T>::Mul(ctx.device_ctx, image_size, out_diff_dptr + n * image_size,
                                         out_blob->dptr<T>() + n * image_size,
                                         ps_dptr + lrn_conf.depth_radius() * channel_size);
    KernelUtil<DeviceType::kCPU, T>::Div(
        ctx.device_ctx, image_size, ps_dptr + lrn_conf.depth_radius() * channel_size,
        nc_dptr + n * image_size, ps_dptr + lrn_conf.depth_radius() * channel_size);
    T* accum_ratio_data = nc_dptr + n * image_size;
    T* accum_ratio_times_bottom = accum_ratio_data + channel_size;
    Memset<DeviceType::kCPU>(ctx.device_ctx, accum_ratio_data, 0, channel_size * sizeof(T));
    FOR_RANGE(int64_t, c, 0, size - 1) {
      KernelUtil<DeviceType::kCPU, T>::Axpy(ctx.device_ctx, channel_size, 1.,
                                            ps_dptr + c * channel_size, 1, accum_ratio_data, 1);
    }
    FOR_RANGE(int64_t, c, 0, in_shape.At(1)) {
      KernelUtil<DeviceType::kCPU, T>::Axpy(ctx.device_ctx, channel_size, 1.,
                                            ps_dptr + (c + size - 1) * channel_size, 1,
                                            accum_ratio_data, 1);
      KernelUtil<DeviceType::kCPU, T>::Mul(ctx.device_ctx, channel_size,
                                           in_blob->dptr<T>() + n * image_size + c * channel_size,
                                           accum_ratio_data, accum_ratio_times_bottom);
      KernelUtil<DeviceType::kCPU, T>::Axpy(ctx.device_ctx, channel_size, -cache_ratio_value,
                                            accum_ratio_times_bottom, 1,
                                            in_diff_dptr + n * image_size + c * channel_size, 1);
      KernelUtil<DeviceType::kCPU, T>::Axpy(ctx.device_ctx, channel_size, -1.,
                                            ps_dptr + c * channel_size, 1, accum_ratio_data, 1);
    }
  }
}

template<typename T>
void LocalResponseNormalizationKernel<DeviceType::kCPU, T>::NHWCForward(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const ShapeView& in_shape = in_blob->shape();
  Blob* out_blob = BnInOp2Blob("out");
  Blob* padded_square_blob = BnInOp2Blob("padded_square");
  Blob* normalize_coef_blob = BnInOp2Blob("normalize_coef");
  Memset<DeviceType::kCPU>(ctx.device_ctx, out_blob->mut_dptr(), 0, out_blob->ByteSizeOfBlobBody());
  Memset<DeviceType::kCPU>(ctx.device_ctx, padded_square_blob->mut_dptr(), 0,
                           padded_square_blob->ByteSizeOfBlobBody());
  Memset<DeviceType::kCPU>(ctx.device_ctx, normalize_coef_blob->mut_dptr(), 0,
                           normalize_coef_blob->ByteSizeOfBlobBody());
  ConstEigenMatrixMap<T> in_mat(in_blob->dptr<T>(), in_shape.At(3),
                                in_shape.elem_cnt() / in_shape.At(3));
  EigenMatrixMap<T> out_mat(out_blob->mut_dptr<T>(), in_shape.At(3),
                            in_shape.elem_cnt() / in_shape.At(3));
  EigenMatrixMap<T> padded_square_mat(padded_square_blob->mut_dptr<T>(),
                                      padded_square_blob->shape().At(0), 1);
  EigenMatrixMap<T> normalize_coef_mat(normalize_coef_blob->mut_dptr<T>(), in_shape.At(3),
                                       in_shape.elem_cnt() / in_shape.At(3));

  const LocalResponseNormalizationOpConf& lrn_conf =
      this->op_conf().local_response_normalization_conf();
  const int32_t double_depth_radius = lrn_conf.depth_radius() * 2;
  FOR_RANGE(int32_t, r, 0, in_mat.cols()) {
    padded_square_mat.block(lrn_conf.depth_radius(), 0, out_mat.rows(), 1) =
        in_mat.col(r).cwiseProduct(in_mat.col(r)) * static_cast<T>(lrn_conf.alpha());
    T accumulated_scale(0);
    FOR_RANGE(int32_t, i, 0, double_depth_radius) { accumulated_scale += padded_square_mat(i); }
    FOR_RANGE(int32_t, i, 0, in_mat.rows()) {
      accumulated_scale += padded_square_mat(i + double_depth_radius);
      normalize_coef_mat(i, r) = static_cast<T>(lrn_conf.bias()) + accumulated_scale;
      accumulated_scale -= padded_square_mat(i);
    }
  }

  if (lrn_conf.beta() == 1) {
    out_mat.array() = in_mat.array() * normalize_coef_mat.array().inverse();
  } else if (lrn_conf.beta() == 0.5) {
    out_mat.array() = in_mat.array() * normalize_coef_mat.array().rsqrt();
  } else {
    out_mat.array() = in_mat.array()
                      * (normalize_coef_mat.array().log() * -static_cast<T>(lrn_conf.beta())).exp();
  }
}

template<typename T>
void LocalResponseNormalizationKernel<DeviceType::kCPU, T>::NHWCBackward(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const ShapeView& in_shape = in_blob->shape();
  const Blob* out_blob = BnInOp2Blob("out");
  const Blob* normalize_coef_blob = BnInOp2Blob("normalize_coef");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<DeviceType::kCPU>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                           in_diff_blob->ByteSizeOfBlobBody());
  ConstEigenMatrixMap<T> in_mat(in_blob->dptr<T>(), in_shape.At(3),
                                in_shape.elem_cnt() / in_shape.At(3));
  ConstEigenMatrixMap<T> out_mat(out_blob->dptr<T>(), in_shape.At(3),
                                 in_shape.elem_cnt() / in_shape.At(3));
  ConstEigenMatrixMap<T> normalize_coef_mat(normalize_coef_blob->dptr<T>(), in_shape.At(3),
                                            in_shape.elem_cnt() / in_shape.At(3));
  ConstEigenMatrixMap<T> out_diff_mat(out_diff_blob->dptr<T>(), in_shape.At(3),
                                      in_shape.elem_cnt() / in_shape.At(3));
  EigenMatrixMap<T> in_diff_mat(in_diff_blob->mut_dptr<T>(), in_shape.At(3),
                                in_shape.elem_cnt() / in_shape.At(3));

  const LocalResponseNormalizationOpConf& lrn_conf =
      this->op_conf().local_response_normalization_conf();
  FOR_RANGE(int32_t, i, 0, in_diff_mat.rows()) {
    FOR_RANGE(int32_t, j, 0, in_diff_mat.cols()) {
      int32_t depth_begin = std::max(0, j - lrn_conf.depth_radius());
      int32_t depth_end =
          std::min(static_cast<int32_t>(in_shape.At(3)), j + lrn_conf.depth_radius() + 1);
      FOR_RANGE(int32_t, k, depth_begin, depth_end) {
        T dyi = T(-2) * lrn_conf.alpha() * lrn_conf.beta() * in_mat(i, k) * out_mat(i, j)
                / normalize_coef_mat(i, j);
        if (k == j) {
          dyi += Eigen::numext::pow(normalize_coef_mat(i, j), static_cast<T>(-lrn_conf.beta()));
        }
        dyi *= out_diff_mat(i, j);
        in_diff_mat(i, k) += dyi;
      }
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLocalResponseNormalizationConf,
                           LocalResponseNormalizationKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
