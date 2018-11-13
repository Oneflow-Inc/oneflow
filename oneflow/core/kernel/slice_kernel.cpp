#include "oneflow/core/kernel/slice_kernel.h"
#include "oneflow/core/ndarray/ndarray_helper.h"

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

template<DeviceType device_type, typename T>
void SliceKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  SliceKernelUtil<device_type, T>::Forward(ctx.device_ctx,
                                           this->op_conf().slice_conf().dim_slice_conf(),
                                           BnInOp2Blob("in"), BnInOp2Blob("out"));
}

template<DeviceType device_type, typename T>
void SliceKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  SliceKernelUtil<device_type, T>::Backward(ctx.device_ctx,
                                            this->op_conf().slice_conf().dim_slice_conf(),
                                            BnInOp2Blob(GenDiffBn("out")), in_diff_blob);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSliceConf, SliceKernel, ARITHMETIC_DATA_TYPE_SEQ);

template<typename T>
struct SliceKernelUtil<DeviceType::kCPU, T> final {
  static void Forward(DeviceCtx* device_ctx, const PbRpf<DimSliceConf>& rep_dim_slice,
                      const Blob* in_blob, Blob* out_blob) {
    CHECK_EQ(in_blob->shape().NumAxes(), out_blob->shape().NumAxes());
    switch (out_blob->shape().NumAxes()) {
// clang-format off
#define MAKE_CASE(num_axes)                                                                   \
      case num_axes: {                                                                        \
        NdArraySliceUtil<T, num_axes>::Forward(device_ctx, rep_dim_slice, in_blob, out_blob); \
        break;                                                                                \
      }
      MAKE_CASE(2);
      MAKE_CASE(3);
#undef MAKE_CASE
      // clang-format on
      default: { UNIMPLEMENTED(); }
    }
  }

  static void Backward(DeviceCtx* device_ctx, const PbRpf<DimSliceConf>& rep_dim_slice,
                       const Blob* out_diff_blob, Blob* in_diff_blob) {
    CHECK_EQ(out_diff_blob->shape().NumAxes(), in_diff_blob->shape().NumAxes());
    size_t num_axes = in_diff_blob->shape().NumAxes();
    switch (num_axes) {
// clang-format off
#define MAKE_CASE(num_axes)                                                                   \
      case num_axes: {                                                                        \
        NdArraySliceUtil<T, num_axes>::Backward(device_ctx, rep_dim_slice, out_diff_blob,     \
                                                in_diff_blob);                                \
        break;                                                                                \
      }
      MAKE_CASE(2);
      MAKE_CASE(3);
#undef MAKE_CASE
      // clang-format on
      default: { UNIMPLEMENTED(); }
    }
  }
};

#define INSTANTIATE_SLICE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct SliceKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SLICE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ)

template<typename T>
struct NdArraySliceUtil<T, 2> final {
  static void Forward(DeviceCtx* device_ctx, const PbRpf<DimSliceConf>& rep_dim_slice,
                      const Blob* in_blob, Blob* out_blob) {
    NdArrayHelper<T, 2> ndarray;
    auto&& in_ndarray = ndarray.Var(in_blob->shape(), const_cast<T*>(in_blob->dptr<T>()));
    auto&& out_ndarray = ndarray.Var(out_blob->shape(), out_blob->mut_dptr<T>());
    out_ndarray.Assign(in_ndarray({}, {GetStart(rep_dim_slice.Get(0)), GetEnd(rep_dim_slice.Get(0)),
                                       GetStride(rep_dim_slice.Get(0))}));
  }

  static void Backward(DeviceCtx* device_ctx, const PbRpf<DimSliceConf>& rep_dim_slice,
                       const Blob* out_diff_blob, Blob* in_diff_blob) {
    NdArrayHelper<T, 2> ndarray;
    auto&& out_diff_ndarray =
        ndarray.Var(out_diff_blob->shape(), const_cast<T*>(out_diff_blob->dptr<T>()));
    auto&& in_diff_ndarray = ndarray.Var(in_diff_blob->shape(), in_diff_blob->mut_dptr<T>());
    in_diff_ndarray({}, {GetStart(rep_dim_slice.Get(0)), GetEnd(rep_dim_slice.Get(0)),
                         GetStride(rep_dim_slice.Get(0))})
        .Assign(out_diff_ndarray({}, {}));
  }
};

template<typename T>
struct NdArraySliceUtil<T, 3> final {
  static void Forward(DeviceCtx* device_ctx, const PbRpf<DimSliceConf>& rep_dim_slice,
                      const Blob* in_blob, Blob* out_blob) {
    NdArrayHelper<T, 3> ndarray;
    auto&& in_ndarray = ndarray.Var(in_blob->shape(), const_cast<T*>(in_blob->dptr<T>()));
    auto&& out_ndarray = ndarray.Var(out_blob->shape(), out_blob->mut_dptr<T>());
    out_ndarray.Assign(in_ndarray({},
                                  {GetStart(rep_dim_slice.Get(0)), GetEnd(rep_dim_slice.Get(0)),
                                   GetStride(rep_dim_slice.Get(0))},
                                  {GetStart(rep_dim_slice.Get(1)), GetEnd(rep_dim_slice.Get(1)),
                                   GetStride(rep_dim_slice.Get(1))}));
  }

  static void Backward(DeviceCtx* device_ctx, const PbRpf<DimSliceConf>& rep_dim_slice,
                       const Blob* out_diff_blob, Blob* in_diff_blob) {
    NdArrayHelper<T, 3> ndarray;
    auto&& out_diff_ndarray =
        ndarray.Var(out_diff_blob->shape(), const_cast<T*>(out_diff_blob->dptr<T>()));
    auto&& in_diff_ndarray = ndarray.Var(in_diff_blob->shape(), in_diff_blob->mut_dptr<T>());
    in_diff_ndarray({},
                    {GetStart(rep_dim_slice.Get(0)), GetEnd(rep_dim_slice.Get(0)),
                     GetStride(rep_dim_slice.Get(0))},
                    {GetStart(rep_dim_slice.Get(1)), GetEnd(rep_dim_slice.Get(1)),
                     GetStride(rep_dim_slice.Get(1))})
        .Assign(out_diff_ndarray({}, {}, {}));
  }
};

}  // namespace oneflow
