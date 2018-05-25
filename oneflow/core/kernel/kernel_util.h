#ifndef ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_

#include "oneflow/core/common/blas.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/range.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/switch_func.h"

namespace oneflow {

template<cudaMemcpyKind cpy_kind>
void Memcpy(DeviceCtx*, void* dst, const void* src, size_t sz);

template<DeviceType device_type>
struct GetCudaMemcpyKind;
template<>
struct GetCudaMemcpyKind<DeviceType::kCPU> {
  static const cudaMemcpyKind val = cudaMemcpyKind::cudaMemcpyHostToHost;
};
template<>
struct GetCudaMemcpyKind<DeviceType::kGPU> {
  static const cudaMemcpyKind val = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
};
size_t GetTmpSizeForReduceSum(DataType data_type, int64_t sum_elem_num);

template<DeviceType device_type>
void Memcpy(DeviceCtx*, void* dst, const void* src, size_t sz,
            cudaMemcpyKind kind = GetCudaMemcpyKind<device_type>::val);

void AutoMemcpy(DeviceCtx* ctx, void* dst, const void* src, size_t sz,
                const MemoryCase& src_mem_case, const MemoryCase& dst_mem_case);

template<DeviceType device_type>
void Memset(DeviceCtx*, void* dst, const char value, size_t sz);

#if defined(__CUDACC__)
#define OF_DEVICE_FUNC __device__
#else
#define OF_DEVICE_FUNC
#endif

template<typename T>
OF_DEVICE_FUNC T ReduceCoreAdd(const T x, const T y) {
  return x + y;
}

template<typename T>
OF_DEVICE_FUNC T ReduceCoreMax(const T x, const T y) {
  return x > y ? x : y;
}

// CPU, GPU, Integral, Floating
template<DeviceType device_type, typename T, typename Derived>
struct KernelUtilIf {
  static void OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const T alpha, const T* a, const T* b,
                     const T beta, T* c) {
    const int lda = (trans_a == CblasNoTrans) ? k : m;
    const int ldb = (trans_b == CblasNoTrans) ? n : k;
    const int ldc = n;

    Derived::Gemm(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c,
                  ldc);
  }

  static void OFGemmTrans(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                          enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                          const T alpha, const T* a, const T* b, const T beta, T* c) {
    trans_a = (trans_a == CblasNoTrans) ? CblasTrans : CblasNoTrans;
    trans_b = (trans_b == CblasNoTrans) ? CblasTrans : CblasNoTrans;
    OFGemm(ctx, trans_b, trans_a, n, m, k, alpha, b, a, beta, c);
  }

  static void BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       T alpha, T beta, const Blob* a, const Blob* b, Blob* c) {
    const int m = c->shape().At(0);
    const int n = c->shape().Count(1);
    const int k = (trans_a == CblasNoTrans) ? a->shape().Count(1) : a->shape().At(0);

    OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a->dptr<T>(), b->dptr<T>(), beta,
           c->mut_dptr<T>());
  }

  static void InitializeWithProperConf(DeviceCtx* ctx, const InitializerConf* initializer_conf,
                                       uint32_t random_seed, Blob* blob,
                                       const std::string& data_format = "") {
    if (initializer_conf == nullptr) {
      initializer_conf = Global<JobDesc>::Get()->DefaultInitializerConf();
    }
    Derived::InitializeWithConf(ctx, *initializer_conf, random_seed, blob, data_format);
  }
  static void InitializeWithProperConf(DeviceCtx* ctx, const PbMessage* initializer_conf,
                                       uint32_t random_seed, Blob* blob,
                                       const std::string& data_format = "") {
    InitializeWithProperConf(ctx, static_cast<const InitializerConf*>(initializer_conf),
                             random_seed, blob, data_format);
  }
};

template<DeviceType device_type, typename T, typename U = void>
struct KernelUtil;

// CPU, Integral, Floating
template<typename T, typename Derived>
struct CpuKernelUtilIf {
  static void Axpy(DeviceCtx* ctx, const int n, const T* alpha, const T* x, const int incx, T* y,
                   const int incy);
  static void Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr);
  static void Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr, T* temp_storage,
                  size_t temp_storage_bytes);
  static void Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr);
  static void Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr, T* temp_storage,
                  size_t temp_storage_bytes);
  static void CopyColsRegion(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num,
                             const T* x, const int64_t x_col_offset, const int64_t x_lda, T* y,
                             const int64_t y_col_offset, const int64_t y_lda);
  static void RowMax(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x,
                     T* y);
  static void RowMax(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x, T* y,
                     void* temp_storage, const size_t temp_storage_bytes) {
    RowMax(ctx, row_num, col_num, x, y);
  }
  static void RowSum(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x,
                     T* y);
  static void RowSum(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x, T* y,
                     void* temp_storage, const size_t temp_storage_bytes) {
    RowSum(ctx, row_num, col_num, x, y);
  }
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape,
                        const Shape& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const T* x, T* y);
  static void InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                                const std::string& model_dir, Blob* blob,
                                const std::string& bn_in_op, int32_t dim_num,
                                int64_t num_in_each_dim);
};

// CPU, Floating
template<typename T>
struct KernelUtil<DeviceType::kCPU, T, typename std::enable_if<IsFloating<T>::value>::type>
    : public KernelUtilIf<DeviceType::kCPU, T, KernelUtil<DeviceType::kCPU, T>>,
      public CpuKernelUtilIf<T, KernelUtil<DeviceType::kCPU, T>> {
  static void Dot(DeviceCtx* ctx, const int n, const T* x, const int incx, const T* y,
                  const int incy, T* result);
  static void Copy(DeviceCtx* ctx, const int n, const T* x, const int incx, T* y, const int incy);
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                   const int incy);
  static void Scal(DeviceCtx* ctx, const int n, const T alpha, T* x, const int incx);
  static void Scal(DeviceCtx* ctx, const int n, const T* alpha, T* x, const int incx);
  static void Gemv(DeviceCtx* ctx, const enum CBLAS_TRANSPOSE trans, int m, int n, const T alpha,
                   const T* a, int lda, const T* x, const int incx, const T beta, T* y,
                   const int incy);
  static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a,
                   const enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                   const T alpha, const T* a, const int lda, const T* b, const int ldb,
                   const T beta, T* c, const int ldc);

  static void Exp(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
  static void Div(DeviceCtx* ctx, const int64_t n, T* x, const T* alpha);
  static void Div(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z);
  static void Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z);
  static void Rsqrt(DeviceCtx* ctx, const int64_t n, T* x, const float epsilon);
  static void Powx(DeviceCtx* ctx, const int64_t n, const T* x, const float power, T* y);

  static void Sigmoid(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
  static void SigmoidBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                              T* dx);
  static void TanH(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
  static void TanHBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx);
  static void Relu(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx);

  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob);
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob, const std::string& data_format);
};

// CPU, Integral
template<typename T>
struct KernelUtil<DeviceType::kCPU, T, typename std::enable_if<IsIntegral<T>::value>::type>
    : public KernelUtilIf<DeviceType::kCPU, T, KernelUtil<DeviceType::kCPU, T>>,
      public CpuKernelUtilIf<T, KernelUtil<DeviceType::kCPU, T>> {
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                   const int incy);
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob);
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob, const std::string& data_format);
};

// GPU, Integral, Floating
template<typename T, typename Derived>
struct GpuKernelUtilIf {
  static void Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr, T* temp_storage,
                  size_t temp_storage_bytes);
  static void Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr, T* temp_storage,
                  size_t temp_storage_bytes);
  static void CopyColsRegion(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num,
                             const T* x, const int64_t x_col_offset, const int64_t x_lda, T* y,
                             const int64_t y_col_offset, const int64_t y_lda);
  static void RowMax(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x, T* y,
                     void* temp_storage, const size_t temp_storage_bytes);
  static void RowSum(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x, T* y,
                     void* temp_storage, const size_t temp_storage_bytes);
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape,
                        const Shape& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const T* x, T* y);
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob);
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob, const std::string& data_format);
  static void InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                                const std::string& model_dir, Blob* blob,
                                const std::string& bn_in_op, int32_t dim_num,
                                int64_t num_in_each_dim);
};

// GPU, Floating
template<typename T>
struct KernelUtil<DeviceType::kGPU, T, typename std::enable_if<IsFloating<T>::value>::type>
    : public KernelUtilIf<DeviceType::kGPU, T, KernelUtil<DeviceType::kGPU, T>>,
      public GpuKernelUtilIf<T, KernelUtil<DeviceType::kGPU, T>> {
  static void Dot(DeviceCtx* ctx, const int n, const T* x, const int incx, const T* y,
                  const int incy, T* result);
  static void Copy(DeviceCtx* ctx, const int n, const T* x, const int incx, T* y, const int incy);
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                   const int incy);
  static void Axpy(DeviceCtx* ctx, const int n, const T* alpha, const T* x, const int incx, T* y,
                   const int incy);
  static void Scal(DeviceCtx* ctx, const int n, const T alpha, T* x, const int incx);
  static void Scal(DeviceCtx* ctx, const int n, const T* alpha, T* x, const int incx);
  static void Gemv(DeviceCtx* ctx, const enum CBLAS_TRANSPOSE trans, int m, int n, const T alpha,
                   const T* a, int lda, const T* x, const int incx, const T beta, T* y,
                   const int incy);
  static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a,
                   const enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                   const T alpha, const T* a, const int lda, const T* b, const int ldb,
                   const T beta, T* c, const int ldc);

  static void Exp(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
  static void Div(DeviceCtx* ctx, const int64_t n, T* x, const T* alpha);
  static void Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z);
  static void Rsqrt(DeviceCtx* ctx, const int64_t n, T* x, const float epsilon);

  static void Sigmoid(DeviceCtx* ctx, int64_t n, const T* x, T* y);
  static void SigmoidBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                              T* dx);
  static void TanH(DeviceCtx* ctx, int64_t n, const T* x, T* y);
  static void TanHBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx);
  static void Relu(DeviceCtx* ctx, int64_t n, const T* x, T* y);
  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx);
};

// GPU, Integral
template<typename T>
struct KernelUtil<DeviceType::kGPU, T, typename std::enable_if<IsIntegral<T>::value>::type>
    : public KernelUtilIf<DeviceType::kGPU, T, KernelUtil<DeviceType::kGPU, T>>,
      public GpuKernelUtilIf<T, KernelUtil<DeviceType::kGPU, T>> {
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                   const int incy);
};

using CopyBlobFieldMthd = void (Blob::*)(DeviceCtx*, const Blob*);

template<DeviceType device_type, typename Iter>
void CopyFromIterToIter(DeviceCtx* ctx, Iter& src_it, Iter& dst_it) {
  const char* src_ptr = nullptr;
  size_t src_size = 0;
  char* dst_ptr = nullptr;
  size_t dst_size = 0;
  while (true) {
    if (src_size == 0) { std::tie(src_ptr, src_size) = src_it.GetNext(); }
    if (dst_size == 0) { std::tie(dst_ptr, dst_size) = dst_it.GetNext(); }
    if (src_size == 0) {
      CHECK_EQ(src_size, dst_size);
      break;
    }
    size_t cp_size = std::min(src_size, dst_size);
    if (dst_ptr != nullptr) {
      if (src_ptr != nullptr) {
        Memcpy<device_type>(ctx, dst_ptr, src_ptr, cp_size);
      } else {
        Memset<device_type>(ctx, dst_ptr, 0, cp_size);
      }
      dst_ptr += cp_size;
    }
    if (src_ptr != nullptr) { src_ptr += cp_size; }
    src_size -= cp_size;
    dst_size -= cp_size;
  }
}

class DataContentIterator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataContentIterator);
  DataContentIterator() = delete;
  ~DataContentIterator() = default;

  DataContentIterator(std::function<Blob*(const std::string&)> BnInOp2Blob,
                      const PbRpf<std::string>* bns, int32_t axis) {
    BnInOp2Blob_ = BnInOp2Blob;
    seg_num_ = BnInOp2Blob(bns->Get(0))->shape().Count(0, axis);
    seg_idx_ = 0;
    bns_ = bns;
    bn_idx_ = 0;
    axis_ = axis;
  }

  std::tuple<char*, size_t> GetNext() {
    std::tuple<char*, size_t> ret(nullptr, 0);
    if (seg_idx_ == seg_num_) { return ret; }
    Blob* blob = BnInOp2Blob_(bns_->Get(bn_idx_));
    int64_t elem_num = blob->shape().Count(axis_);
    std::get<1>(ret) = elem_num * GetSizeOfDataType(blob->data_type());
    if (blob->IsColValid()) {
      std::get<0>(ret) = blob->mut_dptr<char>() + seg_idx_ * std::get<1>(ret);
    }
    bn_idx_ += 1;
    if (bn_idx_ == bns_->size()) {
      bn_idx_ = 0;
      seg_idx_ += 1;
    }
    return ret;
  }

  static CopyBlobFieldMthd GetCopyBlobFieldMthd() { return &Blob::CopyDataContentFrom; }

 private:
  std::function<Blob*(const std::string&)> BnInOp2Blob_;
  int64_t seg_num_;
  int64_t seg_idx_;
  const PbRpf<std::string>* bns_;
  int32_t bn_idx_;
  int32_t axis_;
};

class ParallelConcatSplitHelper {
 public:
  ParallelConcatSplitHelper(std::function<Blob*(const std::string&)> BnInOp2Blob,
                            const PbRpf<std::string>* bns, int32_t axis, int32_t thr_num)
      : BnInOp2Blob_(BnInOp2Blob), bns_(bns), axis_(axis), thr_num_(thr_num) {
    int64_t bn_num = bns_->size();
    int64_t axis_zero_total_size = calc_axis_zero_total_size();

    std::vector<int64_t> thr_range_begin(thr_num_);
    std::vector<int64_t> thr_range_end(thr_num_);
    BalancedSplitter balanced_splitter(axis_zero_total_size, thr_num_);
    FOR_RANGE(size_t, thr_id, 0, thr_num_) {
      thr_range_begin[thr_id] = balanced_splitter.At(thr_id).begin();
      thr_range_end[thr_id] = balanced_splitter.At(thr_id).end();
    }

    if (0 == axis_) {
      seg_bottom_.resize(thr_num_, 0);
      seg_top_.resize(thr_num_, 1);

      std::vector<int64_t> blob_begin(bn_num);
      std::vector<int64_t> blob_end(bn_num);
      blob_begin[0] = 0;
      blob_end[0] = BnInOp2Blob_(bns_->Get(0))->shape().At(0);
      FOR_RANGE(size_t, i, 1, bn_num) {
        blob_begin[i] = blob_end[i - 1];
        blob_end[i] = blob_begin[i] + BnInOp2Blob_(bns_->Get(i))->shape().At(0);
      }

      bn_bottom_.resize(thr_num_);
      bn_top_.resize(thr_num_);
      std::vector<std::vector<int64_t>> thr_blob_begin(thr_num_);
      std::vector<std::vector<int64_t>> thr_blob_end(thr_num_);
      FOR_RANGE(int64_t, thr_id, 0, thr_num_) {
        thr_blob_begin[thr_id].resize(bn_num);
        thr_blob_end[thr_id].resize(bn_num);
        FOR_RANGE(int64_t, bn_id, 0, bn_num) {
          thr_blob_begin[thr_id][bn_id] = 0;
          thr_blob_end[thr_id][bn_id] = blob_end[bn_id] - blob_begin[bn_id];
          if (blob_begin[bn_id] <= thr_range_begin[thr_id]
              && thr_range_begin[thr_id] < blob_end[bn_id]) {
            bn_bottom_[thr_id] = bn_id;
            thr_blob_begin[thr_id][bn_id] = thr_range_begin[thr_id] - blob_begin[bn_id];
          }
          if (blob_begin[bn_id] <= thr_range_end[thr_id] - 1
              && thr_range_end[thr_id] - 1 < blob_end[bn_id]) {
            bn_top_[thr_id] = bn_id + 1;
            thr_blob_end[thr_id][bn_id] = thr_range_end[thr_id] - blob_begin[bn_id];
          }
        }
      }

      thr_bn_offset_.resize(thr_num_);
      thr_bn_elem_size_.resize(thr_num_);
      FOR_RANGE(int64_t, thr_id, 0, thr_num_) {
        thr_bn_offset_[thr_id].resize(bn_num, 0);
        thr_bn_elem_size_[thr_id].resize(bn_num, 0);
        FOR_RANGE(int64_t, bn_id, 0, bn_num) {
          if (bn_id == bn_bottom_[thr_id]) {
            thr_bn_offset_[thr_id][bn_id] =
                thr_blob_begin[thr_id][bn_id] * BnInOp2Blob_(bns_->Get(bn_id))->shape().Count(1)
                * GetSizeOfDataType(BnInOp2Blob_(bns_->Get(0))->data_type());
          } else {
            thr_bn_offset_[thr_id][bn_id] = 0;
          }
          thr_bn_elem_size_[thr_id][bn_id] =
              (thr_blob_end[thr_id][bn_id] - thr_blob_begin[thr_id][bn_id])
              * BnInOp2Blob(bns_->Get(bn_id))->shape().Count(1)
              * GetSizeOfDataType(BnInOp2Blob_(bns_->Get(0))->data_type());
        }
      }
    } else {
      bn_bottom_.resize(thr_num, 0);
      bn_top_.resize(thr_num, bn_num);

      seg_bottom_.resize(thr_num_);
      seg_top_.resize(thr_num_);

      int64_t seg_num = BnInOp2Blob_(bns_->Get(0))->shape().Count(0, axis_);
      CHECK_EQ(seg_num % axis_zero_total_size, 0);
      FOR_RANGE(size_t, thr_id, 0, thr_num_) {
        seg_bottom_[thr_id] = seg_num / axis_zero_total_size * thr_range_begin[thr_id];
        seg_top_[thr_id] = seg_num / axis_zero_total_size * thr_range_end[thr_id];
      }
      // The thr_bn_offset_ and thr_bn_elem_size_ will be set in ParallelConcatSplitIterator
    }
  }

 public:
  int64_t seg_bottom(int64_t thr_id) const {
    CHECK_LT(thr_id, thr_num_);
    return seg_bottom_[thr_id];
  }
  int64_t seg_top(int64_t thr_id) const {
    CHECK_LT(thr_id, thr_num_);
    return seg_top_[thr_id];
  }
  int64_t bn_bottom(int64_t thr_id) const {
    CHECK_LT(thr_id, thr_num_);
    return bn_bottom_[thr_id];
  }
  int64_t bn_top(int64_t thr_id) const {
    CHECK_LT(thr_id, thr_num_);
    return bn_top_[thr_id];
  }
  int64_t thr_bn_offset(int64_t thr_id, int64_t bn_id) const {
    CHECK_LT(thr_id, thr_num_);
    CHECK_LT(bn_id, bns_->size());
    return thr_bn_offset_[thr_id][bn_id];
  }
  int64_t thr_bn_elem_size(int64_t thr_id, int64_t bn_id) const {
    CHECK_LT(thr_id, thr_num_);
    CHECK_LT(bn_id, bns_->size());
    return thr_bn_elem_size_[thr_id][bn_id];
  }

 private:
  std::function<Blob*(const std::string&)> BnInOp2Blob_;
  const PbRpf<std::string>* bns_;
  int32_t axis_;
  int32_t thr_num_;

  std::vector<int64_t> seg_bottom_;
  std::vector<int64_t> seg_top_;
  std::vector<int64_t> bn_bottom_;
  std::vector<int64_t> bn_top_;
  std::vector<std::vector<int64_t>> thr_bn_offset_;
  std::vector<std::vector<int64_t>> thr_bn_elem_size_;

  int64_t calc_axis_zero_total_size() {
    int64_t axis_zero_total_size = 0;
    if (0 == axis_) {
      FOR_RANGE(size_t, i, 0, bns_->size()) {
        axis_zero_total_size += BnInOp2Blob_(bns_->Get(i))->shape().At(0);
      }
    } else {
      axis_zero_total_size = BnInOp2Blob_(bns_->Get(0))->shape().At(0);
    }
    return axis_zero_total_size;
  }
};

class ParallelDataContentIterator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ParallelDataContentIterator);
  ParallelDataContentIterator() = delete;
  ~ParallelDataContentIterator() = default;

  ParallelDataContentIterator(std::function<Blob*(const std::string&)> BnInOp2Blob,
                              const PbRpf<std::string>* bns, int32_t axis, int32_t thr_id,
                              const ParallelConcatSplitHelper& helper)
      : helper_(helper) {
    BnInOp2Blob_ = BnInOp2Blob;
    bns_ = bns;
    axis_ = axis;
    thr_id_ = thr_id;

    seg_idx_ = helper_.seg_bottom(thr_id);
    bn_idx_ = helper_.bn_bottom(thr_id);
  }

  std::tuple<char*, size_t> GetNext() {
    std::tuple<char*, size_t> ret(nullptr, 0);
    if (seg_idx_ == helper_.seg_top(thr_id_)) { return ret; }

    Blob* blob = BnInOp2Blob_(bns_->Get(bn_idx_));
    std::get<1>(ret) = elem_size(blob);
    if (blob->IsColValid()) { std::get<0>(ret) = blob->mut_dptr<char>() + offset(blob); }

    bn_idx_ += 1;
    if (bn_idx_ == helper_.bn_top(thr_id_)) {
      bn_idx_ = 0;  // if axis == 0, will not come here, if axis != 0, bn_idx_ returns to 0
      seg_idx_ += 1;
    }
    return ret;
  }

 private:
  std::function<Blob*(const std::string&)> BnInOp2Blob_;
  const PbRpf<std::string>* bns_;
  int32_t axis_;
  int32_t thr_id_;
  const ParallelConcatSplitHelper& helper_;

  int64_t seg_idx_;
  int64_t bn_idx_;

  int64_t offset(const Blob* blob) {
    if (0 == axis_) {
      return helper_.thr_bn_offset(thr_id_, bn_idx_);
    } else {
      return seg_idx_ * elem_size(blob);
    }
  }
  int64_t elem_size(const Blob* blob) {
    if (0 == axis_) {
      return helper_.thr_bn_elem_size(thr_id_, bn_idx_);
    } else {
      return blob->shape().Count(axis_) * GetSizeOfDataType(blob->data_type());
    }
  }
};

class FieldIterator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FieldIterator);
  FieldIterator() = delete;
  virtual ~FieldIterator() = default;

  FieldIterator(std::function<Blob*(const std::string&)> BnInOp2Blob, const PbRpf<std::string>* bns,
                int32_t axis) {
    BnInOp2Blob_ = BnInOp2Blob;
    bns_ = bns;
    bn_idx_ = 0;
    if (axis == 0) {
      bn_num_ = bns_->size();
    } else {
      bn_num_ = 1;
    }
  }

  std::tuple<char*, size_t> GetNext() {
    std::tuple<char*, size_t> ret(nullptr, 0);
    if (bn_idx_ == bn_num_) { return ret; }
    Blob* blob = BnInOp2Blob_(bns_->Get(bn_idx_++));
    std::get<0>(ret) = GetMutPtr(blob);
    std::get<1>(ret) = GetSizeOfField(blob);
    return ret;
  }

 protected:
  virtual char* GetMutPtr(Blob* blob) = 0;
  virtual size_t GetSizeOfField(Blob* blob) const = 0;

 private:
  std::function<Blob*(const std::string&)> BnInOp2Blob_;
  const PbRpf<std::string>* bns_;
  int32_t bn_idx_;
  int32_t bn_num_;
};

class DataIdIterator final : public FieldIterator {
 public:
  DataIdIterator(std::function<Blob*(const std::string&)> BnInOp2Blob,
                 const PbRpf<std::string>* bns, int32_t axis)
      : FieldIterator(BnInOp2Blob, bns, axis) {}
  static CopyBlobFieldMthd GetCopyBlobFieldMthd() { return &Blob::CopyDataIdFrom; }

 private:
  char* GetMutPtr(Blob* blob) override { return blob->mut_data_id(); }

  size_t GetSizeOfField(Blob* blob) const override { return blob->ByteSizeOfDataIdField(); }
};

class ColNumIterator final : public FieldIterator {
 public:
  ColNumIterator(std::function<Blob*(const std::string&)> BnInOp2Blob,
                 const PbRpf<std::string>* bns, int32_t axis)
      : FieldIterator(BnInOp2Blob, bns, axis) {}
  static CopyBlobFieldMthd GetCopyBlobFieldMthd() { return &Blob::CopyColNumFrom; }

 private:
  char* GetMutPtr(Blob* blob) override { return reinterpret_cast<char*>(blob->mut_col_num()); }

  size_t GetSizeOfField(Blob* blob) const override { return blob->ByteSizeOfColNumField(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_
