#include "oneflow/core/kernel/kernel.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

class SegmentOffsetCreator final {
 public:
  SegmentOffsetCreator(int32_t num_col) : num_col_(num_col) {}
  __device__ int32_t operator()(int32_t idx) const { return idx * num_col_; }

 private:
  int32_t num_col_;
};

}  // namespace

template<typename T>
__global__ void WriteToOutputKernel(const int32_t elem_cnt,
                                    const cub::KeyValuePair<int32_t, T>* key_value_out_ptr,
                                    int32_t* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { out_ptr[i] = key_value_out_ptr[i].key; }
}

template<typename T>
class ArgmaxGpuKernel : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ArgmaxGpuKernel);
  ArgmaxGpuKernel() = default;
  ~ArgmaxGpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* temp_storage_blob = BnInOp2Blob("temp_storage");
    Blob* key_value_out_blob = BnInOp2Blob("key_value_out");
    Blob* out_blob = BnInOp2Blob("out");
    const int32_t num_col = in_blob->shape().At(in_blob->shape().NumAxes() - 1);
    const int32_t num_row = in_blob->shape().elem_cnt() / num_col;

    cub::CountingInputIterator<int32_t> counting_iter(0);
    cub::TransformInputIterator<int32_t, SegmentOffsetCreator, cub::CountingInputIterator<int32_t>>
        segment_offsets_t(counting_iter, SegmentOffsetCreator(num_col));

    cub::KeyValuePair<int32_t, T>* key_value_out_ptr =
        reinterpret_cast<cub::KeyValuePair<int32_t, T>*>(key_value_out_blob->mut_dptr<char>());
    size_t temp_storage_bytes = temp_storage_blob->shape().elem_cnt();
    auto err = cub::DeviceSegmentedReduce::ArgMax(
        /* d_temp_storage */ temp_storage_blob->mut_dptr<char>(),
        /* temp_storage_bytes */ temp_storage_bytes,
        /* d_in */ in_blob->dptr<T>(),
        /* d_out */ key_value_out_ptr,
        /* num_segments */ num_row,
        /* d_begin_offsets */ segment_offsets_t,
        /* d_end_offsets */ segment_offsets_t + 1,
        /* stream */ ctx.device_ctx->cuda_stream());
    CudaCheck(err);

    const int32_t n = out_blob->shape().elem_cnt();
    WriteToOutputKernel<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx.device_ctx->cuda_stream()>>>(
            n, key_value_out_ptr, out_blob->mut_dptr<int32_t>());
  }
};

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kArgmaxConf, DeviceType::kGPU, float,
                                      ArgmaxGpuKernel<float>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kArgmaxConf, DeviceType::kGPU, double,
                                      ArgmaxGpuKernel<double>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kArgmaxConf, DeviceType::kGPU, int8_t,
                                      ArgmaxGpuKernel<int8_t>)
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kArgmaxConf, DeviceType::kGPU, int32_t,
                                      ArgmaxGpuKernel<int32_t>)

}  // namespace oneflow
