#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/arg_where_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

template<typename T, typename I, size_t NDims>
class ArgWhereCpuKernel : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ArgWhereCpuKernel);
  ArgWhereCpuKernel() = default;
  ~ArgWhereCpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in = BnInOp2Blob("in");
    Blob* out = BnInOp2Blob("out");
    Blob* out_size = BnInOp2Blob("out_size");

    I nonzero_cnt = 0;
    FOR_RANGE(int64_t, i, 0, in->shape().elem_cnt()) {
      if (static_cast<bool>(in->dptr<T>()[i])) {
        out->mut_dptr<I>()[nonzero_cnt * NDims] = i;
        nonzero_cnt += 1;
      }
    }
    CHECK_LE(nonzero_cnt, std::numeric_limits<I>::max());

    fixed_vector<I, NDims> dims(NDims);
    std::transform(in->shape().ptr(), in->shape().ptr() + in->shape().NumAxes(), dims.begin(),
                   [](int64_t dim) { return static_cast<I>(dim); });
    NdIndexOffsetHelper<I, NDims> index_converter(dims.data(), dims.size());
    FOR_RANGE(I, i, 0, nonzero_cnt) {
      I* nd_index_ptr = out->mut_dptr<I>() + i * NDims;
      // convert offset to nd_index inplace
      index_converter.OffsetToNdIndex(*nd_index_ptr, nd_index_ptr);
    }
    *out_size->mut_dptr<I>() = nonzero_cnt;
  }
};

#define REGISTER_ARG_WHERE_CPU_KERNELS(dtype_pair, itype_pair)             \
  REGISTER_ARG_WHERE_KERNELS_AT_NDIMS(ArgWhereCpuKernel, DeviceType::kCPU, \
                                      OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(itype_pair))

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ARG_WHERE_CPU_KERNELS, ARITHMETIC_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
