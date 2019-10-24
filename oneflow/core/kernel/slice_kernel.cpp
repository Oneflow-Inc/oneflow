#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/ndarray/cpu_ndarray_builder.h"

namespace oneflow {

namespace {

int64_t GetStart(const DimSliceConf& conf) {
  CHECK_GT(conf.stride(), 0);
  return conf.has_start() ? conf.start() : Slice::kStart;
}

int64_t GetEnd(const DimSliceConf& conf) {
  CHECK_GT(conf.stride(), 0);
  return conf.has_end() ? conf.end() : Slice::kEnd;
}

int64_t GetStride(const DimSliceConf& conf) { return conf.stride(); }

}  // namespace

template<typename T, size_t NDIMS>
struct NdarraySliceUtil;

template<typename T>
struct NdarraySliceUtil<T, 2> final {
  static void Forward(DeviceCtx* device_ctx, const PbRpf<DimSliceConf>& rep_dim_slice,
                      const Blob* in_blob, Blob* out_blob) {
    CpuNdarrayBuilder<T, 2> ndarray;
    auto&& in_ndarray = ndarray.Var(in_blob->shape(), const_cast<T*>(in_blob->dptr<T>()));
    auto&& out_ndarray = ndarray.Var(out_blob->shape(), out_blob->mut_dptr<T>());
    out_ndarray.CopyFrom(in_ndarray({GetStart(rep_dim_slice.Get(0)), GetEnd(rep_dim_slice.Get(0)),
                                     GetStride(rep_dim_slice.Get(0))},
                                    {GetStart(rep_dim_slice.Get(1)), GetEnd(rep_dim_slice.Get(1)),
                                     GetStride(rep_dim_slice.Get(1))}));
  }

  static void Backward(DeviceCtx* device_ctx, const PbRpf<DimSliceConf>& rep_dim_slice,
                       const Blob* out_diff_blob, Blob* in_diff_blob) {
    CpuNdarrayBuilder<T, 2> ndarray;
    auto&& out_diff_ndarray =
        ndarray.Var(out_diff_blob->shape(), const_cast<T*>(out_diff_blob->dptr<T>()));
    auto&& in_diff_ndarray = ndarray.Var(in_diff_blob->shape(), in_diff_blob->mut_dptr<T>());
    in_diff_ndarray({GetStart(rep_dim_slice.Get(0)), GetEnd(rep_dim_slice.Get(0)),
                     GetStride(rep_dim_slice.Get(0))},
                    {GetStart(rep_dim_slice.Get(1)), GetEnd(rep_dim_slice.Get(1)),
                     GetStride(rep_dim_slice.Get(1))})
        .CopyFrom(out_diff_ndarray({}, {}));
  }
};

template<typename T>
struct NdarraySliceUtil<T, 3> final {
  static void Forward(DeviceCtx* device_ctx, const PbRpf<DimSliceConf>& rep_dim_slice,
                      const Blob* in_blob, Blob* out_blob) {
    CpuNdarrayBuilder<T, 3> ndarray;
    auto&& in_ndarray = ndarray.Var(in_blob->shape(), const_cast<T*>(in_blob->dptr<T>()));
    auto&& out_ndarray = ndarray.Var(out_blob->shape(), out_blob->mut_dptr<T>());
    out_ndarray.CopyFrom(in_ndarray({GetStart(rep_dim_slice.Get(0)), GetEnd(rep_dim_slice.Get(0)),
                                     GetStride(rep_dim_slice.Get(0))},
                                    {GetStart(rep_dim_slice.Get(1)), GetEnd(rep_dim_slice.Get(1)),
                                     GetStride(rep_dim_slice.Get(1))},
                                    {GetStart(rep_dim_slice.Get(2)), GetEnd(rep_dim_slice.Get(2)),
                                     GetStride(rep_dim_slice.Get(2))}));
  }

  static void Backward(DeviceCtx* device_ctx, const PbRpf<DimSliceConf>& rep_dim_slice,
                       const Blob* out_diff_blob, Blob* in_diff_blob) {
    CpuNdarrayBuilder<T, 3> ndarray;
    auto&& out_diff_ndarray =
        ndarray.Var(out_diff_blob->shape(), const_cast<T*>(out_diff_blob->dptr<T>()));
    auto&& in_diff_ndarray = ndarray.Var(in_diff_blob->shape(), in_diff_blob->mut_dptr<T>());
    in_diff_ndarray({GetStart(rep_dim_slice.Get(0)), GetEnd(rep_dim_slice.Get(0)),
                     GetStride(rep_dim_slice.Get(0))},
                    {GetStart(rep_dim_slice.Get(1)), GetEnd(rep_dim_slice.Get(1)),
                     GetStride(rep_dim_slice.Get(1))},
                    {GetStart(rep_dim_slice.Get(2)), GetEnd(rep_dim_slice.Get(2)),
                     GetStride(rep_dim_slice.Get(2))})
        .CopyFrom(out_diff_ndarray({}, {}, {}));
  }
};

template<typename T>
class SliceCpuKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceCpuKernel);
  SliceCpuKernel() = default;
  ~SliceCpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const SliceOpConf& conf = this->op_conf().slice_conf();
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    CHECK_EQ(in_blob->shape().NumAxes(), out_blob->shape().NumAxes());

    switch (out_blob->shape().NumAxes()) {
// clang-format off
    #define MAKE_CASE(num_axes)                                                                 \
      case num_axes: {                                                                          \
        NdarraySliceUtil<T, num_axes>::Forward(ctx.device_ctx, conf.dim_slice_conf(), in_blob,  \
                                              out_blob);                                        \
        break;                                                                                  \
      }
      MAKE_CASE(2);
      MAKE_CASE(3);
    #undef MAKE_CASE
      // clang-format on
      default: { UNIMPLEMENTED(); }
    }
  }
};

template<typename T>
class SliceGradCpuKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceGradCpuKernel);
  SliceGradCpuKernel() = default;
  ~SliceGradCpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const SliceGradOpConf& conf = this->op_conf().slice_grad_conf();
    const Blob* dy_blob = BnInOp2Blob("dy");
    Blob* dx_blob = BnInOp2Blob("dx");
    CHECK_EQ(dy_blob->shape().NumAxes(), dx_blob->shape().NumAxes());

    Memset<DeviceType::kCPU>(ctx.device_ctx, dx_blob->mut_dptr<T>(), 0,
                             dx_blob->ByteSizeOfBlobBody());

    switch (dx_blob->shape().NumAxes()) {
// clang-format off
    #define MAKE_CASE(num_axes)                                                         \
      case num_axes: {                                                                  \
        NdarraySliceUtil<T, num_axes>::Backward(ctx.device_ctx, conf.dim_slice_conf(),  \
                                                dy_blob, dx_blob);                      \
        break;                                                                          \
      }
      MAKE_CASE(2);
      MAKE_CASE(3);
    #undef MAKE_CASE
      // clang-format on
      default: { UNIMPLEMENTED(); }
    }
  }
};

#define REGISTER_SLICE_CPU_KERNEL(dtype)                                                       \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSliceConf, DeviceType::kCPU, dtype,     \
                                        SliceCpuKernel<dtype>)                                 \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSliceGradConf, DeviceType::kCPU, dtype, \
                                        SliceGradCpuKernel<dtype>)

REGISTER_SLICE_CPU_KERNEL(float);
REGISTER_SLICE_CPU_KERNEL(double);
REGISTER_SLICE_CPU_KERNEL(int8_t);
REGISTER_SLICE_CPU_KERNEL(int32_t);
REGISTER_SLICE_CPU_KERNEL(int64_t);

}  // namespace oneflow
