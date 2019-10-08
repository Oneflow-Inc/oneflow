#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/common/eigen_util.h"

namespace oneflow {

class PoolingCpuCtx final {
 public:
  PoolingCpuCtx(const PoolingKernelConf& kernel_conf) : kernel_conf_(kernel_conf) {}
  ~PoolingCpuCtx() = default;

  const PoolingKernelConf& kernel_conf() const { return kernel_conf_; }

 private:
  PoolingKernelConf kernel_conf_;
};

template<DeviceType device_type, typename T>
class PoolingCpuKernelIf : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingCpuKernelIf);
  PoolingCpuKernelIf() = default;
  virtual ~PoolingCpuKernelIf() = default;

 protected:
  const PoolingCpuCtx& pooling_ctx() const { return *pooling_ctx_; }
  void VirtualKernelInit() override {
    pooling_ctx_.reset(new PoolingCpuCtx(GetPoolingKernelConf()));
  }
  virtual const PoolingKernelConf& GetPoolingKernelConf() const = 0;
  void ForwardDataContent(const KernelCtx& kernel_ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    PoolingForward(kernel_ctx, this->pooling_ctx(), in_blob, out_blob);
  }
  virtual void PoolingForward(const KernelCtx& kernel_ctx, const PoolingCpuCtx& pooling_ctx,
                              const Blob* in_blob, Blob* out_blob) const = 0;
  virtual void PoolingBackward(const KernelCtx& kernel_ctx, const PoolingCpuCtx& pooling_ctx,
                               const Blob* out_diff_blob, const Blob* out_blob, const Blob* in_blob,
                               Blob* in_diff_blob) const = 0;

  std::unique_ptr<PoolingCpuCtx> pooling_ctx_;
};

template<typename T>
class PoolingCpuKernel : public PoolingCpuKernelIf<DeviceType::kCPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingCpuKernel);
  PoolingCpuKernel() = default;
  virtual ~PoolingCpuKernel() = default;

 protected:
  void PoolingForward(const KernelCtx& kernel_ctx, const PoolingCpuCtx& pooling_ctx,
                      const Blob* in_blob, Blob* out_blob) const override;
  void PoolingBackward(const KernelCtx& kernel_ctx, const PoolingCpuCtx& pooling_ctx,
                       const Blob* out_diff_blob, const Blob* out_blob, const Blob* in_blob,
                       Blob* in_diff_blob) const override;
  virtual T ForwardInitialize() const = 0;
  virtual void NCDHWProcess(const T& lhs, T& rhs) const = 0;
  virtual void NDHWCProcess(const int64_t in_col, const int64_t out_col,
                            ConstEigenMatrixMap<T>& in_mat, EigenMatrixMap<T>& out_mat) const = 0;
  virtual void NCDHWFinalize(const int64_t size, T& out) const = 0;
  virtual void NDHWCFinalize(const int64_t size, const int64_t col,
                             EigenMatrixMap<T>& out_mat) const = 0;
  virtual void NCDHWProcessGrad(const T& in, const T& out, const T& out_diff, const int64_t size,
                                T& in_diff) const = 0;
  virtual void NDHWCProcessGrad(const int64_t out_col, const int64_t in_col, const int64_t size,
                                ConstEigenArrayMap<T>& out_arr, ConstEigenArrayMap<T>& in_arr,
                                ConstEigenArrayMap<T>& out_diff_arr,
                                EigenArrayMap<T>& in_diff_arr) const = 0;
  void ForwardNCDHW(const PoolingCpuCtx& pooling_ctx, const Blob* in_blob, Blob* out_blob) const;
  void BackwardNCDHW(const PoolingCpuCtx& pooling_ctx, const Blob* out_diff_blob,
                     const Blob* out_blob, const Blob* in_blob, Blob* in_diff_blob) const;
  void ForwardNDHWC(const PoolingCpuCtx& pooling_ctx, const Blob* in_blob, Blob* out_blob) const;
  void BackwardNDHWC(const PoolingCpuCtx& pooling_ctx, const Blob* out_diff_blob,
                     const Blob* out_blob, const Blob* in_blob, Blob* in_diff_blob) const;
};

template<typename T>
void PoolingCpuKernel<T>::PoolingForward(const KernelCtx& kernel_ctx,
                                         const PoolingCpuCtx& pooling_ctx, const Blob* in_blob,
                                         Blob* out_blob) const {
  const std::string& data_format = pooling_ctx.kernel_conf().data_format();
  if (data_format == "channels_first") {
    this->ForwardNCDHW(pooling_ctx, in_blob, out_blob);
  } else if (data_format == "channels_last") {
    this->ForwardNDHWC(pooling_ctx, in_blob, out_blob);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void PoolingCpuKernel<T>::PoolingBackward(const KernelCtx& kernel_ctx,
                                          const PoolingCpuCtx& pooling_ctx,
                                          const Blob* out_diff_blob, const Blob* out_blob,
                                          const Blob* in_blob, Blob* in_diff_blob) const {
  const std::string& data_format = pooling_ctx.kernel_conf().data_format();
  if (data_format == "channels_first") {
    this->BackwardNCDHW(pooling_ctx, out_diff_blob, out_blob, in_blob, in_diff_blob);
  } else if (data_format == "channels_last") {
    this->BackwardNDHWC(pooling_ctx, out_diff_blob, out_blob, in_blob, in_diff_blob);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void PoolingCpuKernel<T>::ForwardNCDHW(const PoolingCpuCtx& ctx, const Blob* in_blob,
                                       Blob* out_blob) const {
  Shape in(ctx.kernel_conf().in());
  Shape out(ctx.kernel_conf().out());
  const PbRf<int32_t>& pool_size = ctx.kernel_conf().pool_size();
  const PbRf<int32_t>& strides = ctx.kernel_conf().strides();
  const PbRf<int32_t>& padding_before = ctx.kernel_conf().padding_before();

  const T* input = in_blob->dptr<T>();
  T* output = out_blob->mut_dptr<T>();
  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, c, 0, in.At(1)) {
      FOR_RANGE(int64_t, pd, 0, out.At(2)) {
        int64_t dstart = pd * strides.Get(0) - padding_before.Get(0);
        int64_t dend = std::min(dstart + pool_size.Get(0), in.At(2));
        dstart = std::max(dstart, static_cast<int64_t>(0));
        FOR_RANGE(int64_t, ph, 0, out.At(3)) {
          int64_t hstart = ph * strides.Get(1) - padding_before.Get(1);
          int64_t hend = std::min(hstart + pool_size.Get(1), in.At(3));
          hstart = std::max(hstart, static_cast<int64_t>(0));
          FOR_RANGE(int64_t, pw, 0, out.At(4)) {
            int64_t wstart = pw * strides.Get(2) - padding_before.Get(2);
            int64_t wend = std::min(wstart + pool_size.Get(2), in.At(4));
            wstart = std::max(wstart, static_cast<int64_t>(0));

            const int64_t pool_index = pd * out.Count(3) + ph * out.At(4) + pw;
            T res = ForwardInitialize();
            FOR_RANGE(int64_t, d, dstart, dend) {
              FOR_RANGE(int64_t, h, hstart, hend) {
                FOR_RANGE(int64_t, w, wstart, wend) {
                  const int64_t input_index = d * in.Count(3) + h * in.At(4) + w;
                  NCDHWProcess(input[input_index], res);
                }
              }
            }
            NCDHWFinalize((dend - dstart) * (hend - hstart) * (wend - wstart), res);
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
void PoolingCpuKernel<T>::BackwardNCDHW(const PoolingCpuCtx& ctx, const Blob* out_diff_blob,
                                        const Blob* out_blob, const Blob* in_blob,
                                        Blob* in_diff_blob) const {
  Shape in(ctx.kernel_conf().in());
  Shape out(ctx.kernel_conf().out());
  const PbRf<int32_t>& pool_size = ctx.kernel_conf().pool_size();
  const PbRf<int32_t>& strides = ctx.kernel_conf().strides();
  const PbRf<int32_t>& padding_before = ctx.kernel_conf().padding_before();

  const T* output_diff = out_diff_blob->dptr<T>();
  const T* output = out_blob->dptr<T>();
  const T* input = in_blob->dptr<T>();
  T* input_diff = in_diff_blob->mut_dptr<T>();
  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, c, 0, in.At(1)) {
      FOR_RANGE(int64_t, pd, 0, out.At(2)) {
        int64_t dstart = pd * strides.Get(0) - padding_before.Get(0);
        int64_t dend = std::min(dstart + pool_size.Get(0), in.At(2));
        dstart = std::max(dstart, static_cast<int64_t>(0));
        FOR_RANGE(int64_t, ph, 0, out.At(3)) {
          int64_t hstart = ph * strides.Get(1) - padding_before.Get(1);
          int64_t hend = std::min(hstart + pool_size.Get(1), in.At(3));
          hstart = std::max(hstart, static_cast<int64_t>(0));
          FOR_RANGE(int64_t, pw, 0, out.At(4)) {
            int64_t wstart = pw * strides.Get(2) - padding_before.Get(2);
            int64_t wend = std::min(wstart + pool_size.Get(2), in.At(4));
            wstart = std::max(wstart, static_cast<int64_t>(0));

            const int64_t size = (dend - dstart) * (hend - hstart) * (wend - wstart);
            const int64_t pool_index = pd * out.Count(3) + ph * out.At(4) + pw;
            FOR_RANGE(int64_t, d, dstart, dend) {
              FOR_RANGE(int64_t, h, hstart, hend) {
                FOR_RANGE(int64_t, w, wstart, wend) {
                  const int64_t index = d * in.Count(3) + h * in.At(4) + w;
                  NCDHWProcessGrad(input[index], output[pool_index], output_diff[pool_index], size,
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
void PoolingCpuKernel<T>::ForwardNDHWC(const PoolingCpuCtx& ctx, const Blob* in_blob,
                                       Blob* out_blob) const {
  Shape in(ctx.kernel_conf().in());
  Shape out(ctx.kernel_conf().out());
  const PbRf<int32_t>& pool_size = ctx.kernel_conf().pool_size();
  const PbRf<int32_t>& strides = ctx.kernel_conf().strides();
  const PbRf<int32_t>& padding_before = ctx.kernel_conf().padding_before();

  ConstEigenMatrixMap<T> in_mat(in_blob->dptr<T>(), in.At(1), in.elem_cnt() / in.At(1));
  EigenMatrixMap<T> out_mat(out_blob->mut_dptr<T>(), out.At(1), out.elem_cnt() / out.At(1));
  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, pd, 0, out.At(2)) {
      int64_t dstart = pd * strides.Get(0) - padding_before.Get(0);
      int64_t dend = std::min(dstart + pool_size.Get(0), in.At(2));
      dstart = std::max(dstart, static_cast<int64_t>(0));
      FOR_RANGE(int64_t, ph, 0, out.At(3)) {
        int64_t hstart = ph * strides.Get(1) - padding_before.Get(1);
        int64_t hend = std::min(hstart + pool_size.Get(1), in.At(3));
        hstart = std::max(hstart, static_cast<int64_t>(0));
        FOR_RANGE(int64_t, pw, 0, out.At(4)) {
          int64_t wstart = pw * strides.Get(2) - padding_before.Get(2);
          int64_t wend = std::min(wstart + pool_size.Get(2), in.At(4));
          wstart = std::max(wstart, static_cast<int64_t>(0));
          const int out_col = ((n * out.At(2) + pd) * out.At(3) + ph) * out.At(4) + pw;
          out_mat.col(out_col).setConstant(ForwardInitialize());
          FOR_RANGE(int64_t, d, dstart, dend) {
            FOR_RANGE(int64_t, h, hstart, hend) {
              FOR_RANGE(int64_t, w, wstart, wend) {
                const int in_col = ((n * in.At(2) + d) * in.At(3) + h) * in.At(4) + w;
                NDHWCProcess(in_col, out_col, in_mat, out_mat);
              }
            }
          }
          NDHWCFinalize((hend - hstart) * (wend - wstart) * (dend - dstart), out_col, out_mat);
        }
      }
    }
  }
}

template<typename T>
void PoolingCpuKernel<T>::BackwardNDHWC(const PoolingCpuCtx& ctx, const Blob* out_diff_blob,
                                        const Blob* out_blob, const Blob* in_blob,
                                        Blob* in_diff_blob) const {
  Shape in(ctx.kernel_conf().in());
  Shape out(ctx.kernel_conf().out());
  const PbRf<int32_t>& pool_size = ctx.kernel_conf().pool_size();
  const PbRf<int32_t>& strides = ctx.kernel_conf().strides();
  const PbRf<int32_t>& padding_before = ctx.kernel_conf().padding_before();

  // caffe2 implementation: need check
  ConstEigenArrayMap<T> out_mat(out_blob->dptr<T>(), out.At(1), out.elem_cnt() / out.At(1));
  ConstEigenArrayMap<T> in_mat(in_blob->dptr<T>(), in.At(1), in.elem_cnt() / in.At(1));
  ConstEigenArrayMap<T> out_diff_mat(out_diff_blob->dptr<T>(), out.At(1),
                                     out.elem_cnt() / out.At(1));
  EigenArrayMap<T> in_diff_mat(in_diff_blob->mut_dptr<T>(), in.At(1), in.elem_cnt() / in.At(1));
  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, pd, 0, out.At(2)) {
      int64_t dstart = pd * strides.Get(0) - padding_before.Get(0);
      int64_t dend = std::min(dstart + pool_size.Get(0), in.At(2));
      dstart = std::max(dstart, static_cast<int64_t>(0));
      FOR_RANGE(int64_t, ph, 0, out.At(3)) {
        int64_t hstart = ph * strides.Get(1) - padding_before.Get(1);
        int64_t hend = std::min(hstart + pool_size.Get(1), in.At(3));
        hstart = std::max(hstart, static_cast<int64_t>(0));
        FOR_RANGE(int64_t, pw, 0, out.At(4)) {
          int64_t wstart = pw * strides.Get(2) - padding_before.Get(2);
          int64_t wend = std::min(wstart + pool_size.Get(2), in.At(4));
          wstart = std::max(wstart, static_cast<int64_t>(0));
          const int64_t pool_index = ((n * out.At(2) + pd) * out.At(3) + ph) * out.At(4) + pw;
          const int64_t size = (dend - dstart) * (hend - hstart) * (wend - wstart);
          FOR_RANGE(int64_t, d, dstart, dend) {
            FOR_RANGE(int64_t, h, hstart, hend) {
              FOR_RANGE(int64_t, w, wstart, wend) {
                const int64_t input_index = ((n * in.At(2) + d) * in.At(3) + h) * in.At(4) + w;
                NDHWCProcessGrad(pool_index, input_index, size, out_mat, in_mat, out_diff_mat,
                                 in_diff_mat);
              }
            }
          }
        }
      }
    }
  }
}

template<typename T>
class MaxPoolingCpuKernel final : public PoolingCpuKernel<T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPoolingCpuKernel);
  MaxPoolingCpuKernel() = default;
  virtual ~MaxPoolingCpuKernel() = default;

 private:
  const PoolingKernelConf& GetPoolingKernelConf() const override {
    return this->kernel_conf().max_pooling_conf().pooling_conf();
  }
  T ForwardInitialize() const override { return GetMinVal<T>(); }
  void NCDHWProcess(const T& lhs, T& rhs) const override {
    if (lhs > rhs) { rhs = lhs; }
  }
  void NDHWCProcess(const int64_t in_col, const int64_t out_col, ConstEigenMatrixMap<T>& in_mat,
                    EigenMatrixMap<T>& out_mat) const override {
    out_mat.col(out_col) = out_mat.col(out_col).cwiseMax(in_mat.col(in_col));
  }
  void NCDHWFinalize(const int64_t size, T& out) const override {}
  void NDHWCFinalize(const int64_t size, const int64_t col,
                     EigenMatrixMap<T>& out_mat) const override {}
  void NCDHWProcessGrad(const T& in, const T& out, const T& out_diff, const int64_t size,
                        T& in_diff) const override {
    if (in == out) { in_diff += out_diff; }
  }
  void NDHWCProcessGrad(const int64_t out_col, const int64_t in_col, const int64_t size,
                        ConstEigenArrayMap<T>& out_arr, ConstEigenArrayMap<T>& in_arr,
                        ConstEigenArrayMap<T>& out_diff_arr,
                        EigenArrayMap<T>& in_diff_arr) const override {
    in_diff_arr.col(in_col) +=
        out_diff_arr.col(out_col)
        * (in_arr.col(in_col).cwiseEqual(out_arr.col(out_col)).template cast<T>());
  }
};

#define REGISTER_POOLING_KERNEL(op_type, dtype, kernel) \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(op_type, DeviceType::kCPU, dtype, kernel<dtype>);

#define REGISTER_MAX_POOLING_KERNEL(dim)                                                     \
  REGISTER_POOLING_KERNEL(OperatorConf::kMaxPooling##dim##Conf, float, MaxPoolingCpuKernel); \
  REGISTER_POOLING_KERNEL(OperatorConf::kMaxPooling##dim##Conf, double, MaxPoolingCpuKernel);

REGISTER_MAX_POOLING_KERNEL(1D);
REGISTER_MAX_POOLING_KERNEL(2D);
REGISTER_MAX_POOLING_KERNEL(3D);

template<typename T>
class AveragePoolingCpuKernel final : public PoolingCpuKernel<T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePoolingCpuKernel);
  AveragePoolingCpuKernel() = default;
  virtual ~AveragePoolingCpuKernel() = default;

 private:
  const PoolingKernelConf& GetPoolingKernelConf() const override {
    return this->kernel_conf().average_pooling_conf().pooling_conf();
  }
  T ForwardInitialize() const override { return GetZeroVal<T>(); }
  void NCDHWProcess(const T& lhs, T& rhs) const override { rhs += lhs; }
  void NDHWCProcess(const int64_t in_col, const int64_t out_col, ConstEigenMatrixMap<T>& in_mat,
                    EigenMatrixMap<T>& out_mat) const override {
    out_mat.col(out_col) += in_mat.col(in_col);
  }
  void NCDHWFinalize(const int64_t size, T& out) const override { out /= size; }
  void NDHWCFinalize(const int64_t size, const int64_t col,
                     EigenMatrixMap<T>& out_mat) const override {
    out_mat.col(col) /= size;
  }
  void NCDHWProcessGrad(const T& in, const T& out, const T& out_diff, const int64_t size,
                        T& in_diff) const override {
    in_diff += (out_diff / static_cast<T>(size));
  }
  void NDHWCProcessGrad(const int64_t out_col, const int64_t in_col, const int64_t size,
                        ConstEigenArrayMap<T>& out_arr, ConstEigenArrayMap<T>& in_arr,
                        ConstEigenArrayMap<T>& out_diff_arr,
                        EigenArrayMap<T>& in_diff_arr) const override {
    in_diff_arr.col(in_col) += out_diff_arr.col(out_col) / static_cast<T>(size);
  }
};

#define REGISTER_AVERAGE_POOLING_KERNEL(dim)                                \
  REGISTER_POOLING_KERNEL(OperatorConf::kAveragePooling##dim##Conf, float,  \
                          AveragePoolingCpuKernel);                         \
  REGISTER_POOLING_KERNEL(OperatorConf::kAveragePooling##dim##Conf, double, \
                          AveragePoolingCpuKernel);

REGISTER_AVERAGE_POOLING_KERNEL(1D);
REGISTER_AVERAGE_POOLING_KERNEL(2D);
REGISTER_AVERAGE_POOLING_KERNEL(3D);

}  // namespace oneflow
