//#include "oneflow/core/kernel/data_loader_kernel.h"
//#include "oneflow/core/common/str_util.h"
//#include "oneflow/core/job/runtime_context.h"
//
// namespace oneflow {
//
// template<typename FloatingPointType>
// class DataLoaderKernel<DeviceType::kCPU, FloatingPointType> final
//    : public Kernel {
// public:
//  OF_DISALLOW_COPY_AND_MOVE(DataLoaderKernel);
//  DataLoaderKernel() = default;
//  ~DataLoaderKernel() = default;
//
//  void Forward(const KernelCtx&,
//               std::function<Blob*(const std::string&)>) const override;
//  void Backward(const KernelCtx&,
//                std::function<Blob*(const std::string&)>) const override {
//    UNEXPECTED_RUN();
//  }
//};
//
// template<typename FloatingPointType>
// void DataLoaderKernel<DeviceType::kCPU, FloatingPointType>::Forward(
//    const KernelCtx& kernel_ctx,
//    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
//  PersistentCircularLineReader* reader =
//      RuntimeCtx::Singleton()->GetDataReader();
//  if (reader == nullptr) {
//    std::string data_dir = op()->GetStringFromSpecialConf("data_dir");
//    int64_t parallel_id = reinterpret_cast<int64_t>(kernel_ctx.other);
//    std::string file_path = data_dir + "part-" + std::to_string(parallel_id);
//    RuntimeCtx::Singleton()->InitDataReader(file_path);
//    reader = RuntimeCtx::Singleton()->GetDataReader();
//  }
//  Blob* label_blob = BnInOp2BlobPtr("label");
//  Blob* feature_blob = BnInOp2BlobPtr("feature");
//
//  kernel_ctx.device_ctx->cpu_stream()->SendWork([=]() {
//    int64_t piece_size = feature_blob->shape().At(0);
//    FloatingPointType* label_dptr = label_blob->mut_dptr<FloatingPointType>();
//    FloatingPointType* feature_dptr =
//        feature_blob->mut_dptr<FloatingPointType>();
//
//    std::string line;
//    std::string token;
//    token.reserve(29);
//    for (int64_t i = 0; i != piece_size; ++i) {
//      reader->ReadLine(&line);
//      const char* line_ptr = line.c_str();
//      line_ptr = StrToToken(line_ptr, ",", &token) + 1;
//      // TODO: set data id
//      line_ptr = StrToToken(line_ptr, ",", &token) + 1;
//      *label_dptr++ = oneflow_cast<FloatingPointType>(token);
//      int debug = 0;
//      for (int64_t j = 0; j < feature_blob->shape().Count(1); ++j) {
//        line_ptr = StrToToken(line_ptr, ",", &token) + 1;
//        *feature_dptr++ = oneflow_cast<FloatingPointType>(token);
//        debug++;
//      }
//      CHECK_EQ(*(line_ptr - 1), '\0');
//    }
//  });
//}
//
// INSTANTIATE_CPU_KERNEL_CLASS(DataLoaderKernel);
// REGISTER_CPU_KERNEL(OperatorConf::kDataLoaderConf, DataLoaderKernel)
//
//}  // namespace oneflow
