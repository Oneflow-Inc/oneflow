#include "oneflow/core/kernel/arg_where_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

template<typename T, typename I, size_t NDims>
struct ArgWhereKernelUtil<DeviceType::kCPU, T, I, NDims> {
  static void ArgWhere(DeviceCtx* ctx, const ShapeView& in_shape, const T* in_ptr, void* tmp,
                       size_t tmp_max_bytes, I* out_ptr, I* out_size_ptr) {
    CHECK_LE(in_shape.elem_cnt(), std::numeric_limits<I>::max());
    I true_cnt = 0;
    fixed_vector<I, NDims> dims(NDims);
    std::transform(in_shape.ptr(), in_shape.ptr() + in_shape.NumAxes(), dims.begin(),
                   [](int64_t dim) { return static_cast<I>(dim); });
    NdIndexOffsetHelper<I, NDims> index_converter(dims.data(), dims.size());
    FOR_RANGE(int64_t, i, 0, in_shape.elem_cnt()) {
      if (static_cast<bool>(in_ptr[i])) {
        index_converter.OffsetToNdIndex(i, out_ptr + true_cnt * NDims);
        true_cnt += 1;
      }
    }
    *out_size_ptr = true_cnt;
  }

  static size_t GetArgWhereWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n) { return 0; }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ARG_WHERE_KERNEL_UTIL, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
