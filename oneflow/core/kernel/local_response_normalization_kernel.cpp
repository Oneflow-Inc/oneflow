#include "oneflow/core/kernel/local_response_normalization_kernel.h"
#include "oneflow/core/common/eigen_util.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

#ifdef WITH_CUDA
CudnnLRNDesc::CudnnLRNDesc(unsigned depth_radius, double alpha, double beta,
                           double bias) {
  CudaCheck(cudnnCreateLRNDescriptor(&val_));
  CudaCheck(cudnnSetLRNDescriptor(val_, depth_radius, alpha, beta, bias));
}

CudnnLRNDesc::~CudnnLRNDesc() { CudaCheck(cudnnDestroyLRNDescriptor(val_)); }
#endif  // WITH_CUDA

template<typename T>
void LocalResponseNormalizationKernel<DeviceType::kCPU, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Shape& in_shape = in_blob->shape();
  Blob* out_blob = BnInOp2Blob("out");
  Blob* padded_square_blob = BnInOp2Blob("padded_square");
  Blob* normalize_coef_blob = BnInOp2Blob("normalize_coef");
  Memset<DeviceType::kCPU>(ctx.device_ctx, out_blob->mut_dptr(), 0,
                           out_blob->ByteSizeOfDataContentField());
  Memset<DeviceType::kCPU>(ctx.device_ctx, padded_square_blob->mut_dptr(), 0,
                           padded_square_blob->ByteSizeOfDataContentField());
  Memset<DeviceType::kCPU>(ctx.device_ctx, normalize_coef_blob->mut_dptr(), 0,
                           normalize_coef_blob->ByteSizeOfDataContentField());
  ConstEigenMatrixMap<T> in_mat(in_blob->dptr<T>(), in_shape.At(3),
                                in_shape.elem_cnt() / in_shape.At(3));
  EigenMatrixMap<T> out_mat(out_blob->mut_dptr<T>(), in_shape.At(3),
                            in_shape.elem_cnt() / in_shape.At(3));
  EigenMatrixMap<T> padded_square_mat(padded_square_blob->mut_dptr<T>(),
                                      padded_square_blob->shape().At(0), 1);
  EigenMatrixMap<T> normalize_coef_mat(normalize_coef_blob->mut_dptr<T>(),
                                       in_shape.At(3),
                                       in_shape.elem_cnt() / in_shape.At(3));

  const LocalResponseNormalizationOpConf& lrn_conf =
      this->op_conf().local_response_normalization_conf();
  const int32_t double_depth_radius = lrn_conf.depth_radius() * 2;
  FOR_RANGE(int32_t, r, 0, in_mat.cols()) {
    padded_square_mat.block(lrn_conf.depth_radius(), 0, out_mat.rows(), 1) =
        in_mat.col(r).cwiseProduct(in_mat.col(r))
        * static_cast<T>(lrn_conf.alpha());
    T accumulated_scale(0);
    FOR_RANGE(int32_t, i, 0, double_depth_radius) {
      accumulated_scale += padded_square_mat(i);
    }
    FOR_RANGE(int32_t, i, 0, in_mat.rows()) {
      accumulated_scale += padded_square_mat(i + double_depth_radius);
      normalize_coef_mat(i, r) =
          static_cast<T>(lrn_conf.bias()) + accumulated_scale;
      accumulated_scale -= padded_square_mat(i);
    }
  }

  if (lrn_conf.beta() == 1) {
    out_mat.array() = in_mat.array() * normalize_coef_mat.array().inverse();
  } else if (lrn_conf.beta() == 0.5) {
    out_mat.array() = in_mat.array() * normalize_coef_mat.array().rsqrt();
  } else {
    out_mat.array() =
        in_mat.array()
        * (normalize_coef_mat.array().log() * -static_cast<T>(lrn_conf.beta()))
              .exp();
  }
}

template<typename T>
void LocalResponseNormalizationKernel<DeviceType::kCPU, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Shape& in_shape = in_blob->shape();
  const Blob* out_blob = BnInOp2Blob("out");
  const Blob* normalize_coef_blob = BnInOp2Blob("normalize_coef");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<DeviceType::kCPU>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                           in_diff_blob->ByteSizeOfDataContentField());
  ConstEigenMatrixMap<T> in_mat(in_blob->dptr<T>(), in_shape.At(3),
                                in_shape.elem_cnt() / in_shape.At(3));
  ConstEigenMatrixMap<T> out_mat(out_blob->dptr<T>(), in_shape.At(3),
                                 in_shape.elem_cnt() / in_shape.At(3));
  ConstEigenMatrixMap<T> normalize_coef_mat(
      normalize_coef_blob->dptr<T>(), in_shape.At(3),
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
      int32_t depth_end = std::min(static_cast<int32_t>(in_shape.At(3)),
                                   j + lrn_conf.depth_radius() + 1);
      FOR_RANGE(int32_t, k, depth_begin, depth_end) {
        T dyi = T(-2) * lrn_conf.alpha() * lrn_conf.beta() * in_mat(i, k)
                * out_mat(i, j) / normalize_coef_mat(i, j);
        if (k == j) {
          dyi += Eigen::numext::pow(normalize_coef_mat(i, j),
                                    static_cast<T>(-lrn_conf.beta()));
        }
        dyi *= out_diff_mat(i, j);
        in_diff_mat(i, k) += dyi;
      }
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLocalResponseNormalizationConf,
                           LocalResponseNormalizationKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
