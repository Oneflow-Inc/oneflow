#ifndef ONEFLOW_CORE_NDARRAY_CONCAT_VAR_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_CONCAT_VAR_NDARRAY_H_

#include "oneflow/core/ndarray/ndarray.h"
#include "oneflow/core/ndarray/var_ndarray.h"
#include "oneflow/core/common/range.h"

namespace oneflow {

template<typename Derived, typename T, int NDIMS, int CONCAT_AXES>
class ConcatVarNdArrayBase final : public NdArray<T, NDIMS> {
 public:
  static const bool immutable = false;
  static_assert(CONCAT_AXES >= 0 && CONCAT_AXES < NDIMS, "CONCAT_AXES should be a valid dim");
  ConcatVarNdArrayBase(const std::vector<VarNdArray<T, NDIMS>>& var_ndarrays)
      : NdArray<T, NDIMS>(CalcConcatenatedShape(var_ndarrays)),
        var_ndarrays_(var_ndarrays),
        dim_ranges_(CalcDimRanges(var_ndarrays)),
        contiguous_lens_(CalcContiguousLengths(var_ndarrays)) {}
  virtual ~ConcatVarNdArrayBase() = default;

  template<typename XT>
  void Assign(const XT& ndarray) {
    NdArrayAssign(dynamic_cast<Derived*>(this), ndarray);
  }

 protected:
  ALWAYS_INLINE void GetVarNdArrayIndexAndInputDim(int64_t output_dim, int32_t* var_index,
                                                   int64_t* input_dim) const {
    TODO();
  }
  ALWAYS_INLINE const VarNdArray<T, NDIMS> var_ndarray(int32_t var_index) const {
    return var_ndarrays_[var_index];
  }
  ALWAYS_INLINE void GetMutPtrAndMinContiguousSize(int32_t var_index, int64_t offset,
                                                   int64_t input_offset, T** ptr, size_t* size) {
    TODO();
  }

 private:
  ALWAYS_INLINE int32_t VarNdArrayIndex4OutputDim(int64_t output_dim) const {
    TODO();
    return 0;
  }
  Shape CalcConcatenatedShape(const std::vector<VarNdArray<T, NDIMS>>& var_ndarrays) {
    CheckInputShape(var_ndarrays);
    TODO();
    return Shape();
  }
  void CheckInputShape(const std::vector<VarNdArray<T, NDIMS>>& var_ndarrays) { TODO(); }
  const std::vector<Range>& CalcDimRanges(const std::vector<VarNdArray<T, NDIMS>>& var_ndarrays) {
    TODO();
    std::vector<Range> ret;
    return ret;
  }
  const std::vector<VarNdArray<T, NDIMS>> var_ndarrays_;
  const std::vector<Range> dim_ranges_;
  const std::vector<size_t> contiguous_lens_;
};

#define DEFINE_CONCAT_VAR_NDARRAY_CTOR_DTOR(ndims, axes)                                        \
  ConcatVarNdArray(const std::vector<VarNdArray<T, ndims>>& var_ndarrays)                       \
      : ConcatVarNdArrayBase<ConcatVarNdArray<T, ndims, axes>, T, ndims, axes>(var_ndarrays) {} \
  ~ConcatVarNdArray() = default;

template<typename T, int NDIMS, int CONCAT_AXES>
class ConcatVarNdArray;

template<typename T>
class ConcatVarNdArray<T, 1, 0> final
    : public ConcatVarNdArrayBase<ConcatVarNdArray<T, 1, 0>, T, 1, 0> {
 public:
  DEFINE_CONCAT_VAR_NDARRAY_CTOR_DTOR(1, 0);

  ALWAYS_INLINE T Get(int64_t dim0) const {
    int32_t var_index = 0;
    this->GetVarNdArrayIndexAndInputDim(dim0, &var_index, &dim0);
    return this->var_ndarray(var_index).Get(dim0);
  }
  ALWAYS_INLINE T* Mut(int64_t dim0) const {
    int32_t var_index = 0;
    this->GetVarNdArrayIndexAndInputDim(dim0, &var_index, &dim0);
    return this->var_ndarray(var_index).Mut(dim0);
  }
  ALWAYS_INLINE void GetMutPtrAndContiguousSize(int64_t offset, T** ptr, size_t* size) const {
    int32_t var_index = 0;
    int64_t dim0 = 0;
    this->GetVarNdArrayIndexAndInputDim(offset, &var_index, &dim0);
    this->GetMutPtrAndMinContiguousSize(var_index, offset, dim0, ptr, size);
  }
};

#define DEFINE_ACCESS_FUNC_BODY(func, axes)                                 \
  func(int64_t dim0, int64_t dim1) const {                                  \
    int32_t var_index = 0;                                                  \
    this->GetVarNdArrayIndexAndInputDim(dim##axes, &var_index, &dim##axes); \
    return this->var_ndarray(var_index).func(dim0, dim1);                   \
  }
#define DEFINE_CONCAT_VAR_NDARRAY(axes)                                                \
  template<typename T>                                                                 \
  class ConcatVarNdArray<T, 2, axes> final                                             \
      : public ConcatVarNdArrayBase<ConcatVarNdArray<T, 2, axes>, T, 2, axes> {        \
   public:                                                                             \
    DEFINE_CONCAT_VAR_NDARRAY_CTOR_DTOR(2, axes);                                      \
    T DEFINE_ACCESS_FUNC_BODY(Get, axes);                                              \
    T* DEFINE_ACCESS_FUNC_BODY(Mut, axes);                                             \
    void GetMutPtrAndContiguousSize(int64_t offset, T** ptr, size_t* size) const {     \
      int64_t dim0 = 0;                                                                \
      int64_t dim1 = 0;                                                                \
      this->Offset2Dims(offset, &dim0, &dim1);                                         \
      int32_t var_index = 0;                                                           \
      this->GetVarNdArrayIndexAndInputDim(dim##axes, &var_index, &dim##axes);          \
      int64_t input_offset = this->var_ndarray(var_index).Dims2Offset(dim0, dim1);     \
      this->GetMutPtrAndMinContiguousSize(var_index, offset, input_offset, ptr, size); \
    }                                                                                  \
  }
DEFINE_CONCAT_VAR_NDARRAY(0);
DEFINE_CONCAT_VAR_NDARRAY(1);
#undef DEFINE_CONCAT_VAR_NDARRAY
#undef DEFINE_ACCESS_FUNC_BODY

#define DEFINE_ACCESS_FUNC_BODY(func, axes)                                 \
  func(int64_t dim0, int64_t dim1, int64_t dim2) const {                    \
    int32_t var_index = 0;                                                  \
    this->GetVarNdArrayIndexAndInputDim(dim##axes, &var_index, &dim##axes); \
    return this->var_ndarray(var_index).func(dim0, dim1, dim2);             \
  }
#define DEFINE_CONCAT_VAR_NDARRAY(axes)                                                  \
  template<typename T>                                                                   \
  class ConcatVarNdArray<T, 3, axes> final                                               \
      : public ConcatVarNdArrayBase<ConcatVarNdArray<T, 3, axes>, T, 3, axes> {          \
   public:                                                                               \
    DEFINE_CONCAT_VAR_NDARRAY_CTOR_DTOR(3, axes);                                        \
    T DEFINE_ACCESS_FUNC_BODY(Get, axes);                                                \
    T* DEFINE_ACCESS_FUNC_BODY(Mut, axes);                                               \
    void GetMutPtrAndContiguousSize(int64_t offset, T** ptr, size_t* size) const {       \
      int64_t dim0 = 0;                                                                  \
      int64_t dim1 = 0;                                                                  \
      int64_t dim2 = 0;                                                                  \
      this->Offset2Dims(offset, &dim0, &dim1, &dim2);                                    \
      int32_t var_index = 0;                                                             \
      this->GetVarNdArrayIndexAndInputDim(dim##axes, &var_index, &dim##axes);            \
      int64_t input_offset = this->var_ndarray(var_index).Dims2Offset(dim0, dim1, dim2); \
      this->GetMutPtrAndMinContiguousSize(var_index, offset, input_offset, ptr, size);   \
    }                                                                                    \
  }
DEFINE_CONCAT_VAR_NDARRAY(0);
DEFINE_CONCAT_VAR_NDARRAY(1);
DEFINE_CONCAT_VAR_NDARRAY(2);
#undef DEFINE_CONCAT_VAR_NDARRAY
#undef DEFINE_ACCESS_FUNC_BODY

#define DEFINE_ACCESS_FUNC_BODY(func, axes)                                 \
  func(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {      \
    int32_t var_index = 0;                                                  \
    this->GetVarNdArrayIndexAndInputDim(dim##axes, &var_index, &dim##axes); \
    return this->var_ndarray(var_index).func(dim0, dim1, dim2, dim3);       \
  }
#define DEFINE_CONCAT_VAR_NDARRAY(axes)                                                        \
  template<typename T>                                                                         \
  class ConcatVarNdArray<T, 4, axes> final                                                     \
      : public ConcatVarNdArrayBase<ConcatVarNdArray<T, 4, axes>, T, 4, axes> {                \
   public:                                                                                     \
    DEFINE_CONCAT_VAR_NDARRAY_CTOR_DTOR(4, axes);                                              \
    T DEFINE_ACCESS_FUNC_BODY(Get, axes);                                                      \
    T* DEFINE_ACCESS_FUNC_BODY(Mut, axes);                                                     \
    void GetMutPtrAndContiguousSize(int64_t offset, T** ptr, size_t* size) const {             \
      int64_t dim0 = 0;                                                                        \
      int64_t dim1 = 0;                                                                        \
      int64_t dim2 = 0;                                                                        \
      int64_t dim3 = 0;                                                                        \
      this->Offset2Dims(offset, &dim0, &dim1, &dim2, &dim3);                                   \
      int32_t var_index = 0;                                                                   \
      this->GetVarNdArrayIndexAndInputDim(dim##axes, &var_index, &dim##axes);                  \
      int64_t input_offset = this->var_ndarray(var_index).Dims2Offset(dim0, dim1, dim2, dim3); \
      this->GetMutPtrAndMinContiguousSize(var_index, offset, input_offset, ptr, size);         \
    }                                                                                          \
  }
DEFINE_CONCAT_VAR_NDARRAY(0);
DEFINE_CONCAT_VAR_NDARRAY(1);
DEFINE_CONCAT_VAR_NDARRAY(2);
DEFINE_CONCAT_VAR_NDARRAY(3);
#undef DEFINE_CONCAT_VAR_NDARRAY
#undef DEFINE_ACCESS_FUNC_BODY

#define DEFINE_ACCESS_FUNC_BODY(func, axes)                                          \
  func(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4) const { \
    int32_t var_index = 0;                                                           \
    this->GetVarNdArrayIndexAndInputDim(dim##axes, &var_index, &dim##axes);          \
    return this->var_ndarray(var_index).func(dim0, dim1, dim2, dim3, dim4);          \
  }
#define DEFINE_CONCAT_VAR_NDARRAY(axes)                                                \
  template<typename T>                                                                 \
  class ConcatVarNdArray<T, 5, axes> final                                             \
      : public ConcatVarNdArrayBase<ConcatVarNdArray<T, 5, axes>, T, 5, axes> {        \
   public:                                                                             \
    DEFINE_CONCAT_VAR_NDARRAY_CTOR_DTOR(5, axes);                                      \
    T DEFINE_ACCESS_FUNC_BODY(Get, axes);                                              \
    T* DEFINE_ACCESS_FUNC_BODY(Mut, axes);                                             \
    void GetMutPtrAndContiguousSize(int64_t offset, T** ptr, size_t* size) const {     \
      int64_t dim0 = 0;                                                                \
      int64_t dim1 = 0;                                                                \
      int64_t dim2 = 0;                                                                \
      int64_t dim3 = 0;                                                                \
      int64_t dim4 = 0;                                                                \
      this->Offset2Dims(offset, &dim0, &dim1, &dim2, &dim3, &dim4);                    \
      int32_t var_index = 0;                                                           \
      this->GetVarNdArrayIndexAndInputDim(dim##axes, &var_index, &dim##axes);          \
      int64_t input_offset =                                                           \
          this->var_ndarray(var_index).Dims2Offset(dim0, dim1, dim2, dim3, dim4);      \
      this->GetMutPtrAndMinContiguousSize(var_index, offset, input_offset, ptr, size); \
    }                                                                                  \
  }
DEFINE_CONCAT_VAR_NDARRAY(0);
DEFINE_CONCAT_VAR_NDARRAY(1);
DEFINE_CONCAT_VAR_NDARRAY(2);
DEFINE_CONCAT_VAR_NDARRAY(3);
DEFINE_CONCAT_VAR_NDARRAY(4);
#undef DEFINE_CONCAT_VAR_NDARRAY
#undef DEFINE_ACCESS_FUNC_BODY

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_CONCAT_VAR_NDARRAY_H_
