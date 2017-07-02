#include "oneflow/core/kernel/data_loader_kernel.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

template<typename FloatingPointType>
class DataLoaderKernel<DeviceType::kCPU, FloatingPointType> final
    : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataLoaderKernel);
  DataLoaderKernel() = default;
  ~DataLoaderKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override {
    UNEXPECTED_RUN();
  }
};

template<typename FloatingPointType>
void DataLoaderKernel<DeviceType::kCPU, FloatingPointType>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  PersistentCircularLineReader* reader =
      RuntimeCtx::Singleton()->GetDataReader();
  if (reader == nullptr) {
    std::string data_dir = op()->GetStringFromSpecialConf("data_dir");
    int64_t parallel_id = reinterpret_cast<int64_t>(kernel_ctx.other);
    std::string file_path = data_dir + "part-" + std::to_string(parallel_id);
    RuntimeCtx::Singleton()->InitDataReader(file_path);
    reader = RuntimeCtx::Singleton()->GetDataReader();
  }
  Blob* label_blob = BnInOp2BlobPtr("label");
  Blob* feature_blob = BnInOp2BlobPtr("feature");
  size_t piece_size = label_blob->shape().elem_cnt();

  kernel_ctx.device_ctx->cpu_stream()->SendWork([=]() {
    FloatingPointType* label_dptr =
        static_cast<FloatingPointType*>(label_blob->mut_dptr());
    FloatingPointType* feature_dptr =
        static_cast<FloatingPointType*>(feature_blob->mut_dptr());
    std::string line;
    bool is_new_line;
    for (size_t i = 0; i != piece_size; ++i) {
      is_new_line = true;
      reader->ReadLine(&line);
      SplitAndParseAs<FloatingPointType>(line, ",",
                                         [&](FloatingPointType data) {
                                           if (is_new_line) {
                                             *label_dptr++ = data;
                                             is_new_line = false;
                                             return;
                                           }
                                           *feature_dptr++ = data;
                                         });
    }
  });
}

INSTANTIATE_CPU_KERNEL_CLASS(DataLoaderKernel);
REGISTER_CPU_KERNEL(OperatorConf::kDataLoaderConf, DataLoaderKernel)

}  // namespace oneflow
