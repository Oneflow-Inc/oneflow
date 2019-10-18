#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/nonzero_kernel_util.cuh"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

template<size_t NDims>
struct OutputIterByNDims {
  // Required iterator traits
  typedef OutputIterByNDims self_type;
  typedef std::ptrdiff_t difference_type;
  typedef void value_type;
  typedef void pointer;
  typedef int32_t& reference;
  typedef std::random_access_iterator_tag iterator_category;

  OutputIterByNDims(int32_t* ptr, int32_t index_max_size)
      : ptr_(ptr), index_max_size_(index_max_size) {
    CHECK_GT(NDims, 1);
  }

  OF_DEVICE_FUNC int32_t& operator[](int i) {
    assert(0 <= i && i < index_max_size_);
    return *(ptr_ + (i * NDims));
  }

 private:
  int32_t* ptr_;
  int32_t index_max_size_;
};

template<size_t NDims>
struct Strides {
  int32_t val[NDims];
};

template<size_t NDims>
__global__ void CalcOutIndexFromFlatIndex(const int32_t* nnz, Strides<NDims> strides,
                                          int32_t* output) {
  int32_t num_nonzero = __ldg(nnz);
  CUDA_1D_KERNEL_LOOP(i, num_nonzero) {
    int32_t flat_index_val = __ldg(output + i * NDims);
    for (int j = 0; j < NDims; ++j) {
      *(output + i * NDims + j) = flat_index_val / strides.val[j];
      flat_index_val %= strides.val[j];
    }
  }
}

template<typename T>
struct IsNonZero {
  OF_DEVICE_FUNC bool operator()(const T& val) const { return (val != static_cast<T>(0)); }
};

}  // namespace

template<typename T, typename OutputIter>
cudaError_t CubSelectFlagged(cudaStream_t stream, int num_items, void* tmp, size_t& tmp_bytes,
                             const T* flags, OutputIter out, int32_t* num_selected) {
  IsNonZero<T> is_nonzero;
  cub::TransformInputIterator<bool, IsNonZero<T>, const T*> flag_iter(flags, is_nonzero);
  cub::CountingInputIterator<int32_t> flat_index_counter(0);
  return cub::DeviceSelect::Flagged(tmp, tmp_bytes, flat_index_counter, flag_iter, out,
                                    num_selected, num_items, stream, false);
}

template<typename T, size_t NDims>
class NonzeroGpuKernel : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NonzeroGpuKernel);
  NonzeroGpuKernel() = default;
  ~NonzeroGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* num_nonzero_blob = BnInOp2Blob("num_nonzero");
    Blob* out_blob = BnInOp2Blob("out");
    Blob* out_tmp_blob = BnInOp2Blob("out_tmp");
    int32_t elem_cnt = in_blob->shape().elem_cnt();
    CHECK_LE(elem_cnt, out_blob->shape().At(0));

    size_t tmp_bytes = 0;
    CudaCheck(CubSelectFlagged<T, int32_t*>(ctx.device_ctx->cuda_stream(), elem_cnt, nullptr,
                                            tmp_bytes, nullptr, nullptr, nullptr));
    CHECK_LE(tmp_bytes, out_tmp_blob->static_shape().elem_cnt());

    if (NDims == 1) {
      CudaCheck(CubSelectFlagged<T, int32_t*>(ctx.device_ctx->cuda_stream(), elem_cnt,
                                              out_tmp_blob->mut_dptr(), tmp_bytes,
                                              in_blob->dptr<T>(), out_blob->mut_dptr<int32_t>(),
                                              num_nonzero_blob->mut_dptr<int32_t>()));
    } else {
      OutputIterByNDims<NDims> out_iter(out_blob->mut_dptr<int32_t>(), elem_cnt);
      CudaCheck(CubSelectFlagged<T, OutputIterByNDims<NDims>>(
          ctx.device_ctx->cuda_stream(), elem_cnt, out_tmp_blob->mut_dptr(), tmp_bytes,
          in_blob->dptr<T>(), out_iter, num_nonzero_blob->mut_dptr<int32_t>()));

      Strides<NDims> strides;
      strides.val[NDims - 1] = 1;
      for (int32_t i = NDims - 2; i >= 0; i--) {
        strides.val[i] = strides.val[i + 1] * in_blob->shape().At(i + 1);
      }
      // TODO(niuchong): BlockNum can be changed to improve perf
      CalcOutIndexFromFlatIndex<NDims>
          <<<128, kCudaThreadsNumPerBlock, 0, ctx.device_ctx->cuda_stream()>>>(
              num_nonzero_blob->dptr<int32_t>(), strides, out_blob->mut_dptr<int32_t>());
    }
  }
};

#define REGISTER_NONZERO_GPU_KERNEL_WITH_NDIMS(dtype, ndims)                           \
  NEW_REGISTER_KERNEL(OperatorConf::kLocalNonzeroConf, NonzeroGpuKernel<dtype, ndims>) \
      .SetIsMatchedPred([](const KernelConf& conf) {                                   \
        return (DeviceType::kGPU == conf.op_attribute().op_conf().device_type())       \
               && (GetDataType<dtype>::value == conf.data_type())                      \
               && (ndims == conf.nonzero_gpu_kernel_conf().num_axes());                \
      });

#define REGISTER_NONZERO_GPU_KERNEL(dtype)          \
  REGISTER_NONZERO_GPU_KERNEL_WITH_NDIMS(dtype, 1); \
  REGISTER_NONZERO_GPU_KERNEL_WITH_NDIMS(dtype, 2); \
  REGISTER_NONZERO_GPU_KERNEL_WITH_NDIMS(dtype, 3);

REGISTER_NONZERO_GPU_KERNEL(float);
REGISTER_NONZERO_GPU_KERNEL(int8_t);
REGISTER_NONZERO_GPU_KERNEL(int32_t);

}  // namespace oneflow
