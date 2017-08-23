#include "oneflow/core/kernel/data_loader_kernel.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

template<typename T>
void DataLoaderKernel<T>::Forward(
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
  Blob* out_blob = BnInOp2BlobPtr("out");
  CHECK_EQ(GetDataType<T>::val, out_blob->data_type());

  kernel_ctx.device_ctx->cpu_stream()->SendWork([=]() {
    int64_t piece_size = out_blob->shape().At(0);
    T* out_dptr = out_blob->mut_dptr<T>();
    std::string line;
    std::string token;
    for (int64_t i = 0; i != piece_size; ++i) {
      reader->ReadLine(&line);
      const char* line_ptr = line.c_str();
      line_ptr = StrToToken(line_ptr, ",", &token) + 1;
      if (out_blob->has_data_id()) {
        memset(out_blob->mut_data_id(), '\0',
               JobDesc::Singleton()->SizeOfOneDataId());
        memcpy(out_blob->mut_data_id(i), token.c_str(), token.size());
      }
      for (int64_t j = 0; j < out_blob->shape().Count(1); ++j) {
        line_ptr = StrToToken(line_ptr, ",", &token) + 1;
        *out_dptr++ = oneflow_cast<T>(token);
      }
      CHECK_EQ(*(line_ptr - 1), '\0');
    }
  });
}

namespace {

Kernel* CreateDataLoaderKernel(const OperatorConf& op_conf) {
  static const HashMap<int, std::function<Kernel*()>> data_type2creator = {
#define MACRO_PAIR(type_cpp, type_proto) \
  {type_proto, []() { return new DataLoaderKernel<type_cpp>; }},
      ARITHMETIC_DATA_TYPE_PAIR()
#undef MACRO_PAIR
  };
  return data_type2creator.at(op_conf.data_loader_conf().out().data_type())();
}

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kDataLoaderConf, DeviceType::kCPU,
                         CreateDataLoaderKernel));

}  // namespace oneflow
