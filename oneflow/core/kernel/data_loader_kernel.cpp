#include "oneflow/core/kernel/data_loader_kernel.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/persistence/cyclic_ubf_in_stream.h"
#include "oneflow/core/persistence/normal_ubf_in_stream.h"

namespace oneflow {

template<typename T>
void DataLoaderKernel<T>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitUbfInStream(kernel_ctx);
  Blob* out_blob = BnInOp2Blob("out");
  CHECK_EQ(GetDataType<T>::val, out_blob->data_type());
  kernel_ctx.device_ctx->cpu_stream()->SendWork([out_blob, this]() {
    int64_t piece_size = out_blob->shape().At(0);
    T* out_dptr = out_blob->mut_dptr<T>();
    auto ubf_item = UbfItem::NewEmpty();
    for (int64_t i = 0; i != piece_size; ++i) {
      int32_t read_status = ubf_in_stream_->ReadOneItem(&ubf_item);
      if (read_status == 0) {
        if (out_blob->has_data_id()) {
          std::string token = ubf_item->GetDataId();
          CHECK_LE(token.size(), JobDesc::Singleton()->SizeOfOneDataId());
          memcpy(out_blob->mut_data_id(i), token.c_str(), token.size());
          if (token.size() != JobDesc::Singleton()->SizeOfOneDataId()) {
            *(out_blob->mut_data_id(i) + token.size()) = '\0';
          }
        }
        ubf_item->Decode(out_blob->shape(), out_dptr);
      } else {
        CHECK_EQ(read_status, -1);
        CHECK(out_blob->has_data_id());
        memset(out_blob->mut_data_id(i), '\0',
               JobDesc::Singleton()->SizeOfOneDataId());
        for (int64_t j = 0; j < out_blob->shape().Count(1); ++j) {
          *out_dptr++ = static_cast<T>(0);
        }
      }
    }
  });
}

template<typename T>
void DataLoaderKernel<T>::InitUbfInStream(const KernelCtx& kernel_ctx) const {
  if (ubf_in_stream_) { return; }
  std::string data_dir = op()->GetStringFromSpecialConf("data_dir");
  int64_t parallel_id = reinterpret_cast<int64_t>(kernel_ctx.other);
  std::string file_path =
      JoinPath(data_dir, "part-" + std::to_string(parallel_id));
  if (JobDesc::Singleton()->is_train()) {
    ubf_in_stream_.reset(new CyclicUbfInStream(GlobalFS(), file_path));
  } else {
    ubf_in_stream_.reset(new NormalUbfInStream(GlobalFS(), file_path));
  }
}

namespace {

Kernel* CreateDataLoaderKernel(const OperatorConf& op_conf) {
  static const HashMap<int, std::function<Kernel*()>> creators = {
#define DATA_LOADER_KERNEL_ENTRY(type_cpp, type_proto) \
  {type_proto, []() { return new DataLoaderKernel<type_cpp>; }},
      OF_PP_FOR_EACH_TUPLE(DATA_LOADER_KERNEL_ENTRY, ARITHMETIC_DATA_TYPE_SEQ)};
  return creators.at(op_conf.data_loader_conf().data_type())();
}

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kDataLoaderConf,
                         CreateDataLoaderKernel));

}  // namespace oneflow
