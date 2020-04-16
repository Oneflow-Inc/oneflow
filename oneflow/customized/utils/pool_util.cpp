#include "oneflow/customized/utils/pool_util.h"
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

namespace {

std::vector<int32_t> Get3DVec(const std::vector<int32_t>& original_vec, int32_t NDims) {
  std::vector<int32_t> vec;
  FOR_RANGE(uint8_t, dim, 0, 3) {
    int64_t index = static_cast<int64_t>(dim) - (3 - NDims);
    if (index < 0) {
      vec.push_back(1);
    } else {
      vec.push_back(original_vec.at(index));
    }
  }
  return vec;
}

}  // namespace

Params3D::Params3D(const int32_t dim, const Shape& x_shape, const std::string& data_format,
                   const std::string& padding, const std::vector<int32_t>& pool_size,
                   const std::vector<int32_t>& strides) {
  dim_ = dim;
  data_format_ = data_format;
  x_3d_ = {GetInDim(x_shape, data_format, 0, dim), GetInDim(x_shape, data_format, 1, dim),
           GetInDim(x_shape, data_format, 2, dim)};
  pool_size_3d_ = Get3DVec(pool_size, dim);
  strides_3d_ = Get3DVec(strides, dim);
  Get3DOutputSize(x_3d_, pool_size_3d_, strides_3d_, padding, &y_3d_, &padding_before_3d_,
                  &padding_after_3d_);
  if (data_format == "channels_first") {
    channel_num_ = x_shape.At(1);
  } else if (data_format == "channels_last") {
    channel_num_ = x_shape.At(x_shape.NumAxes() - 1);
  } else {
    UNIMPLEMENTED();
  }
  batch_num_ = x_shape.At(0);
}

Shape Params3D::GetYShape() const {
  DimVector y_dim_vec;
  if (dim_ == 1) {
    y_dim_vec = {y_3d_.at(2)};
  } else if (dim_ == 2) {
    y_dim_vec = {y_3d_.at(1), y_3d_.at(2)};
  } else if (dim_ == 3) {
    y_dim_vec = {y_3d_.at(0), y_3d_.at(1), y_3d_.at(2)};
  } else {
    UNIMPLEMENTED();
  }
  if (data_format_ == "channels_first") {
    y_dim_vec.insert(y_dim_vec.begin(), channel_num_);
  } else if (data_format_ == "channels_last") {
    y_dim_vec.insert(y_dim_vec.end(), channel_num_);
  } else {
    UNIMPLEMENTED();
  }
  y_dim_vec.insert(y_dim_vec.begin(), batch_num_);
  return Shape(y_dim_vec);
}

Shape Params3D::GetXShape5D() const {
  if (data_format_ == "channels_first") {
    return Shape({batch_num_, channel_num_, x_3d_.at(0), x_3d_.at(1), x_3d_.at(2)});
  } else if (data_format_ == "channels_last") {
    return Shape({batch_num_, channel_num_, x_3d_.at(0), x_3d_.at(1), x_3d_.at(2)});
  }
  UNIMPLEMENTED();
}

Shape Params3D::GetYShape5D() const {
  if (data_format_ == "channels_first") {
    Shape({batch_num_, channel_num_, y_3d_.at(0), y_3d_.at(1), y_3d_.at(2)});
  } else if (data_format_ == "channels_last") {
    Shape({batch_num_, channel_num_, y_3d_.at(0), y_3d_.at(1), y_3d_.at(2)});
  }
  UNIMPLEMENTED();
}

CudnnPoolDesc::CudnnPoolDesc(cudnnPoolingMode_t pooling_mode, int dims, const int* window,
                             const int* padding, const int* stride) {
  CudaCheck(cudnnCreatePoolingDescriptor(&val_));
  CudaCheck(cudnnSetPoolingNdDescriptor(val_, pooling_mode, CUDNN_NOT_PROPAGATE_NAN, dims, window,
                                        padding, stride));
}

CudnnPoolDesc::~CudnnPoolDesc() { CudaCheck(cudnnDestroyPoolingDescriptor(val_)); }

GPUPoolOpKernelState::GPUPoolOpKernelState(const int32_t dim, const std::string& pooling_type,
                                           const Shape& x_shape, const Shape& y_shape,
                                           const std::string& data_format, const DataType& dtype,
                                           const Params3D& params_3d) {
  cudnnPoolingMode_t pooling_mode_;
  if (pooling_type == "AVG") {
    pooling_mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  } else if (pooling_type == "MAX") {
    pooling_mode_ = CUDNN_POOLING_MAX;
  } else {
    UNIMPLEMENTED();
  }

  FixedVector pool_size(dim);
  FixedVector padding(dim);
  FixedVector strides(dim);
  FOR_RANGE(int, i, 0, dim) {
    int32_t index_in_3d = i + 3 - dim;
    pool_size[i] = params_3d.pool_size_3d().at(index_in_3d);
    padding[i] = std::max<int>(params_3d.padding_before_3d().at(index_in_3d),
                               params_3d.padding_after_3d().at(index_in_3d));
    strides[i] = params_3d.strides_3d().at(index_in_3d);
  }

  x_desc_.reset(new CudnnTensorDesc(dtype, x_shape, data_format));
  y_desc_.reset(new CudnnTensorDesc(dtype, y_shape, data_format));
  pooling_desc_.reset(
      new CudnnPoolDesc(pooling_mode_, dim, pool_size.data(), padding.data(), strides.data()));
}

std::shared_ptr<user_op::OpKernelState> GPUPoolOpKernelState::FromKernelInitContext(
    const int32_t& dim, const std::string& pooling_type, user_op::KernelInitContext* ctx) {
  const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
  const std::string data_format = ctx->GetAttr<std::string>("data_format");
  const std::string padding = ctx->GetAttr<std::string>("padding");
  const std::vector<int32_t>& pool_size = ctx->GetAttr<std::vector<int32_t>>("pool_size");
  const std::vector<int32_t>& strides = ctx->GetAttr<std::vector<int32_t>>("strides");
  const Params3D params_3d(dim, x_shape, data_format, padding, pool_size, strides);
  const Shape y_shape = ctx->TensorDesc4ArgNameAndIndex("y", 0)->shape();
  const DataType dtype = ctx->TensorDesc4ArgNameAndIndex("x", 0)->data_type();
  return std::make_shared<OpKernelStateWrapper<GPUPoolOpKernelState>>(
      dim, pooling_type, x_shape, y_shape, data_format, dtype, params_3d);
}

template<typename T>
void PoolKernelUtil<T>::CFirstForward(const Params3D& params_3d, const user_op::Tensor* in_blob,
                                      user_op::Tensor* out_blob,
                                      const ForwardInitialize& initialize,
                                      const CFirstProcess& process,
                                      const CFirstFinalize& finalize) {
  const Shape& in = params_3d.GetXShape5D();
  const Shape& out = params_3d.GetYShape5D();
  const std::vector<int32_t>& pool_size = params_3d.pool_size_3d();
  const std::vector<int32_t>& strides = params_3d.strides_3d();
  const std::vector<int32_t>& padding_before = params_3d.padding_before_3d();

  const T* input = in_blob->dptr<T>();
  T* output = out_blob->mut_dptr<T>();
  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, c, 0, in.At(1)) {
      FOR_RANGE(int64_t, pd, 0, out.At(2)) {
        int64_t dstart = pd * strides.at(0) - padding_before.at(0);
        int64_t dend = std::min(dstart + pool_size.at(0), in.At(2));
        dstart = std::max(dstart, static_cast<int64_t>(0));
        FOR_RANGE(int64_t, ph, 0, out.At(3)) {
          int64_t hstart = ph * strides.at(1) - padding_before.at(1);
          int64_t hend = std::min(hstart + pool_size.at(1), in.At(3));
          hstart = std::max(hstart, static_cast<int64_t>(0));
          FOR_RANGE(int64_t, pw, 0, out.At(4)) {
            int64_t wstart = pw * strides.at(2) - padding_before.at(2);
            int64_t wend = std::min(wstart + pool_size.at(2), in.At(4));
            wstart = std::max(wstart, static_cast<int64_t>(0));

            const int64_t pool_index = pd * out.Count(3) + ph * out.At(4) + pw;
            T res = initialize();
            FOR_RANGE(int64_t, d, dstart, dend) {
              FOR_RANGE(int64_t, h, hstart, hend) {
                FOR_RANGE(int64_t, w, wstart, wend) {
                  const int64_t input_index = d * in.Count(3) + h * in.At(4) + w;
                  process(input[input_index], res);
                }
              }
            }
            finalize((dend - dstart) * (hend - hstart) * (wend - wstart), res);
            output[pool_index] = res;
          }
        }
      }
      input += in.Count(2);
      output += out.Count(2);
    }
  }
}

template<typename T>
void PoolKernelUtil<T>::CFirstBackward(const Params3D& params_3d,
                                       const user_op::Tensor* out_diff_blob,
                                       const user_op::Tensor* out_blob,
                                       const user_op::Tensor* in_blob,
                                       user_op::Tensor* in_diff_blob,
                                       const CFirstProcessGrad& process) {
  const Shape& in = params_3d.GetXShape5D();
  const Shape& out = params_3d.GetYShape5D();
  const std::vector<int32_t>& pool_size = params_3d.pool_size_3d();
  const std::vector<int32_t>& strides = params_3d.strides_3d();
  const std::vector<int32_t>& padding_before = params_3d.padding_before_3d();

  const T* output_diff = out_diff_blob->dptr<T>();
  const T* output = out_blob->dptr<T>();
  const T* input = in_blob->dptr<T>();
  T* input_diff = in_diff_blob->mut_dptr<T>();
  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, c, 0, in.At(1)) {
      FOR_RANGE(int64_t, pd, 0, out.At(2)) {
        int64_t dstart = pd * strides.at(0) - padding_before.at(0);
        int64_t dend = std::min(dstart + pool_size.at(0), in.At(2));
        dstart = std::max(dstart, static_cast<int64_t>(0));
        FOR_RANGE(int64_t, ph, 0, out.At(3)) {
          int64_t hstart = ph * strides.at(1) - padding_before.at(1);
          int64_t hend = std::min(hstart + pool_size.at(1), in.At(3));
          hstart = std::max(hstart, static_cast<int64_t>(0));
          FOR_RANGE(int64_t, pw, 0, out.At(4)) {
            int64_t wstart = pw * strides.at(2) - padding_before.at(2);
            int64_t wend = std::min(wstart + pool_size.at(2), in.At(4));
            wstart = std::max(wstart, static_cast<int64_t>(0));

            const int64_t size = (dend - dstart) * (hend - hstart) * (wend - wstart);
            const int64_t pool_index = pd * out.Count(3) + ph * out.At(4) + pw;
            FOR_RANGE(int64_t, d, dstart, dend) {
              FOR_RANGE(int64_t, h, hstart, hend) {
                FOR_RANGE(int64_t, w, wstart, wend) {
                  const int64_t index = d * in.Count(3) + h * in.At(4) + w;
                  process(input[index], output[pool_index], output_diff[pool_index], size,
                          input_diff[index]);
                }
              }
            }
          }
        }
      }
      // offset
      input += in.Count(2);
      input_diff += in.Count(2);
      output += out.Count(2);
      output_diff += out.Count(2);
    }
  }
}

template<typename T>
void PoolKernelUtil<T>::CLastForward(const Params3D& params_3d, const user_op::Tensor* in_blob,
                                     user_op::Tensor* out_blob,
                                     const ForwardInitialize& forward_initialize,
                                     const CLastProcess& process, const CLastFinalize& finalize) {
  const Shape& in = params_3d.GetXShape5D();
  const Shape& out = params_3d.GetYShape5D();
  const std::vector<int32_t>& pool_size = params_3d.pool_size_3d();
  const std::vector<int32_t>& strides = params_3d.strides_3d();
  const std::vector<int32_t>& padding_before = params_3d.padding_before_3d();

  ConstEigenMatrixMap<T> in_mat(in_blob->dptr<T>(), in.At(1), in.elem_cnt() / in.At(1));
  EigenMatrixMap<T> out_mat(out_blob->mut_dptr<T>(), out.At(1), out.elem_cnt() / out.At(1));
  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, pd, 0, out.At(2)) {
      int64_t dstart = pd * strides.at(0) - padding_before.at(0);
      int64_t dend = std::min(dstart + pool_size.at(0), in.At(2));
      dstart = std::max(dstart, static_cast<int64_t>(0));
      FOR_RANGE(int64_t, ph, 0, out.At(3)) {
        int64_t hstart = ph * strides.at(1) - padding_before.at(1);
        int64_t hend = std::min(hstart + pool_size.at(1), in.At(3));
        hstart = std::max(hstart, static_cast<int64_t>(0));
        FOR_RANGE(int64_t, pw, 0, out.At(4)) {
          int64_t wstart = pw * strides.at(2) - padding_before.at(2);
          int64_t wend = std::min(wstart + pool_size.at(2), in.At(4));
          wstart = std::max(wstart, static_cast<int64_t>(0));
          const int out_col = ((n * out.At(2) + pd) * out.At(3) + ph) * out.At(4) + pw;
          out_mat.col(out_col).setConstant(forward_initialize());
          FOR_RANGE(int64_t, d, dstart, dend) {
            FOR_RANGE(int64_t, h, hstart, hend) {
              FOR_RANGE(int64_t, w, wstart, wend) {
                const int in_col = ((n * in.At(2) + d) * in.At(3) + h) * in.At(4) + w;
                process(in_col, out_col, in_mat, out_mat);
              }
            }
          }
          finalize((hend - hstart) * (wend - wstart) * (dend - dstart), out_col, out_mat);
        }
      }
    }
  }
}

template<typename T>
void PoolKernelUtil<T>::CLastBackward(const Params3D& params_3d,
                                      const user_op::Tensor* out_diff_blob,
                                      const user_op::Tensor* out_blob,
                                      const user_op::Tensor* in_blob, user_op::Tensor* in_diff_blob,
                                      const CLastProcessGrad& process) {
  const Shape& in = params_3d.GetXShape5D();
  const Shape& out = params_3d.GetYShape5D();
  const std::vector<int32_t>& pool_size = params_3d.pool_size_3d();
  const std::vector<int32_t>& strides = params_3d.strides_3d();
  const std::vector<int32_t>& padding_before = params_3d.padding_before_3d();

  // caffe2 implementation: need check
  ConstEigenArrayMap<T> out_mat(out_blob->dptr<T>(), out.At(1), out.elem_cnt() / out.At(1));
  ConstEigenArrayMap<T> in_mat(in_blob->dptr<T>(), in.At(1), in.elem_cnt() / in.At(1));
  ConstEigenArrayMap<T> out_diff_mat(out_diff_blob->dptr<T>(), out.At(1),
                                     out.elem_cnt() / out.At(1));
  EigenArrayMap<T> in_diff_mat(in_diff_blob->mut_dptr<T>(), in.At(1), in.elem_cnt() / in.At(1));
  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, pd, 0, out.At(2)) {
      int64_t dstart = pd * strides.at(0) - padding_before.at(0);
      int64_t dend = std::min(dstart + pool_size.at(0), in.At(2));
      dstart = std::max(dstart, static_cast<int64_t>(0));
      FOR_RANGE(int64_t, ph, 0, out.At(3)) {
        int64_t hstart = ph * strides.at(1) - padding_before.at(1);
        int64_t hend = std::min(hstart + pool_size.at(1), in.At(3));
        hstart = std::max(hstart, static_cast<int64_t>(0));
        FOR_RANGE(int64_t, pw, 0, out.At(4)) {
          int64_t wstart = pw * strides.at(2) - padding_before.at(2);
          int64_t wend = std::min(wstart + pool_size.at(2), in.At(4));
          wstart = std::max(wstart, static_cast<int64_t>(0));
          const int64_t pool_index = ((n * out.At(2) + pd) * out.At(3) + ph) * out.At(4) + pw;
          const int64_t size = (dend - dstart) * (hend - hstart) * (wend - wstart);
          FOR_RANGE(int64_t, d, dstart, dend) {
            FOR_RANGE(int64_t, h, hstart, hend) {
              FOR_RANGE(int64_t, w, wstart, wend) {
                const int64_t input_index = ((n * in.At(2) + d) * in.At(3) + h) * in.At(4) + w;
                process(pool_index, input_index, size, out_mat, in_mat, out_diff_mat, in_diff_mat);
              }
            }
          }
        }
      }
    }
  }
}

// TODO: tsai: initilize template of definition in interfaces
template struct PoolKernelUtil<float>;
template struct PoolKernelUtil<double>;

}  // namespace oneflow
