#include "oneflow/core/kernel/data_loader_kernel.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

namespace {

template<typename FloatingPointType>
void Split2FloatingPoint(const std::string&, const char,
                         std::vector<FloatingPointType>&);

template<>
void Split2FloatingPoint(const std::string& line, const char comma,
                         std::vector<float>& datas) {
  std::stringstream ss(line);

  while (ss.good()) {
    std::string substr;
    std::getline(ss, substr, comma);
    datas.push_back(std::stof(substr));
  }
}

template<>
void Split2FloatingPoint(const std::string& line, const char comma,
                         std::vector<double>& datas) {
  std::stringstream ss(line);

  while (ss.good()) {
    std::string substr;
    std::getline(ss, substr, comma);
    datas.push_back(std::stod(substr));
  }
}

}  // namespace

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
  FloatingPointType* label_dptr =
      static_cast<FloatingPointType*>(label_blob->mut_dptr());
  FloatingPointType* feature_dptr =
      static_cast<FloatingPointType*>(feature_blob->mut_dptr());

  kernel_ctx.device_ctx->cpu_stream()->SendWork([=]() {
    int64_t feature_idx = 0;
    int64_t label_idx = 0;

    for (size_t i = 0; i != piece_size; ++i) {
      std::string line;
      std::vector<FloatingPointType> datas;
      reader->ReadLine(&line);
      Split2FloatingPoint<FloatingPointType>(line, ',', datas);
      label_dptr[label_idx++] = datas[0];
      for (size_t j = 1; j != datas.size(); ++j) {
        feature_dptr[feature_idx++] = datas[j];
      }
    }
  });
}

INSTANTIATE_CPU_KERNEL_CLASS(DataLoaderKernel);
REGISTER_CPU_KERNEL(OperatorConf::kDataLoaderConf, DataLoaderKernel)

}  // namespace oneflow
