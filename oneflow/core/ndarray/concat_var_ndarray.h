#ifndef ONEFLOW_CORE_NDARRAY_CONCAT_VAR_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_CONCAT_VAR_NDARRAY_H_

#include "oneflow/core/ndarray/ndarray.h"
#include "oneflow/core/ndarray/var_ndarray.h"
#include "oneflow/core/common/range.h"

namespace oneflow {

template<typename T, int NDIMS, int CONCAT_AXES>
class ConcatVarNdArray : public NdArray<T, NDIMS> {
 public:
  static const bool immutable = false;
  static_assert(CONCAT_AXES >= 0 && CONCAT_AXES < NDIMS, "CONCAT_AXES should be a valid dim");
  ConcatVarNdArray(const std::vector<VarNdArray<T, NDIMS>>& var_ndarrays)
      : NdArray<T, NDIMS>(CalcConcatenatedShape(var_ndarrays)),
        var_ndarrays_(var_ndarrays),
        dim_ranges_(CalcDimRanges(var_ndarrays)),
        contiguous_lens_(CalcContiguousLens(var_ndarrays)) {}
  ~ConcatVarNdArray() = default;

  template<typename XT>
  void CopyFrom(const XT& ndarray) {
    NdArrayAssign(this, ndarray);
  }
  void GetMutPtrAndContiguousSize(int64_t offset, T** ptr, size_t* size) const {
    int64_t dim[NDIMS] = {0};
    this->xpu_shape().template Offset2Coordinate<NDIMS>(offset, dim);
    int32_t var_index = 0;
    this->GetVarNdArrayIndexAndInputDim(dim[CONCAT_AXES], &var_index, &dim[CONCAT_AXES]);
    int64_t input_offset =
        this->var_ndarray(var_index).xpu_shape().template Coordinate2Offset<NDIMS>(dim);
    this->GetMutPtrAndMinContiguousSize(var_index, input_offset, ptr, size);
  }

 protected:
  ALWAYS_INLINE void GetVarNdArrayIndexAndInputDim(int64_t output_dim, int32_t* var_index,
                                                   int64_t* input_dim) const {
    *var_index = VarNdArrayIndex4OutputDim(output_dim);
    *input_dim = output_dim - dim_ranges_[*var_index].begin();
  }
  ALWAYS_INLINE const VarNdArray<T, NDIMS> var_ndarray(int32_t var_index) const {
    return var_ndarrays_[var_index];
  }
  ALWAYS_INLINE void GetMutPtrAndMinContiguousSize(int32_t var_index, int64_t var_offset, T** ptr,
                                                   size_t* size) const {
    size_t var_contiguous_size = 0;
    var_ndarray(var_index).GetMutPtrAndContiguousSize(var_offset, ptr, &var_contiguous_size);
    *size = std::min(var_contiguous_size,
                     contiguous_lens_[var_index] - var_offset % contiguous_lens_[var_index]);
  }

 private:
  ALWAYS_INLINE int32_t VarNdArrayIndex4OutputDim(int64_t output_dim) const {
    // TODO change to bianry search
    FOR_RANGE(int32_t, i, 0, dim_ranges_.size()) {
      if (output_dim >= dim_ranges_[i].begin() && output_dim < dim_ranges_[i].end()) { return i; }
    }
    UNIMPLEMENTED();
  }
  Shape CalcConcatenatedShape(const std::vector<VarNdArray<T, NDIMS>>& var_ndarrays) const {
    CheckInputShape(var_ndarrays);
    Shape shape(var_ndarrays[0].shape());
    int64_t axes_dim_num = 0;
    FOR_RANGE(int32_t, i, 0, var_ndarrays.size()) {
      axes_dim_num += var_ndarrays[i].shape().At(CONCAT_AXES);
    }
    shape.Set(CONCAT_AXES, axes_dim_num);
    return shape;
  }
  void CheckInputShape(const std::vector<VarNdArray<T, NDIMS>>& var_ndarrays) const {
    FOR_RANGE(int32_t, i, 1, var_ndarrays.size()) {
      FOR_RANGE(int32_t, j, 0, NDIMS) {
        if (j == CONCAT_AXES) { continue; }
        CHECK_EQ(var_ndarrays[0].shape().At(j), var_ndarrays[i].shape().At(j));
      }
    }
  }
  std::vector<Range> CalcDimRanges(const std::vector<VarNdArray<T, NDIMS>>& var_ndarrays) const {
    int64_t axes_dim_num = 0;
    std::vector<Range> ret;
    FOR_RANGE(int32_t, i, 0, var_ndarrays.size()) {
      ret.push_back(Range(axes_dim_num, axes_dim_num + var_ndarrays[i].shape().At(CONCAT_AXES)));
      axes_dim_num += var_ndarrays[i].shape().At(CONCAT_AXES);
    }
    return ret;
  }
  std::vector<size_t> CalcContiguousLens(
      const std::vector<VarNdArray<T, NDIMS>>& var_ndarrays) const {
    std::vector<size_t> ret(var_ndarrays.size(), 0);
    FOR_RANGE(int32_t, i, 0, var_ndarrays.size()) {
      ret[i] = var_ndarrays[i].shape().Count(CONCAT_AXES);
    }
    return ret;
  }
  const std::vector<VarNdArray<T, NDIMS>> var_ndarrays_;
  const std::vector<Range> dim_ranges_;
  const std::vector<size_t> contiguous_lens_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_CONCAT_VAR_NDARRAY_H_
