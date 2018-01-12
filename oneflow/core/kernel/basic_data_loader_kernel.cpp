#include "oneflow/core/kernel/basic_data_loader_kernel.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/persistence/cyclic_persistent_in_stream.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

namespace oneflow {

template<typename T>
void BasicDataLoaderKernel<T>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  CHECK_EQ(GetDataType<T>::val, out_blob->data_type());
  int64_t piece_size = out_blob->shape().At(0);
  T* out_dptr = out_blob->mut_dptr<T>();
  std::string line;
  std::string token;
  for (int64_t i = 0; i != piece_size; ++i) {
    int32_t read_status = in_stream_->ReadLine(&line);
    if (read_status == 0) {
      const char* line_ptr = line.c_str();
      line_ptr = StrToToken(line_ptr, ",", &token) + 1;
      if (out_blob->has_data_id_field()) {
        CHECK_LE(token.size(), JobDesc::Singleton()->SizeOfOneDataId());
        memcpy(out_blob->mut_data_id(i), token.c_str(), token.size());
        if (token.size() != JobDesc::Singleton()->SizeOfOneDataId()) {
          *(out_blob->mut_data_id(i) + token.size()) = '\0';
        }
      }
      for (int64_t j = 0; j < out_blob->shape().Count(1); ++j) {
        line_ptr = StrToToken(line_ptr, ",", &token) + 1;
        *out_dptr++ = oneflow_cast<T>(token);
      }
      CHECK_EQ(*(line_ptr - 1), '\0');
    } else {
      CHECK(kernel_ctx.other);
      *(static_cast<bool*>(kernel_ctx.other)) = true;
      CHECK_EQ(read_status, -1);
      CHECK(out_blob->has_data_id_field());
      memset(out_blob->mut_data_id(i), '\0',
             JobDesc::Singleton()->SizeOfOneDataId());
      for (int64_t j = 0; j < out_blob->shape().Count(1); ++j) {
        *out_dptr++ = static_cast<T>(0);
      }
    }
  }
}

template<typename T>
void BasicDataLoaderKernel<T>::VirtualKernelInit(
    const ParallelContext* parallel_ctx) {
  const std::string& data_dir = op_conf().basic_data_loader_conf().data_dir();
  std::string parallel_id = std::to_string(parallel_ctx->parallel_id());
  std::string file_path = JoinPath(data_dir, "part-" + parallel_id);
  if (JobDesc::Singleton()->IsTrain()) {
    in_stream_.reset(new CyclicPersistentInStream(GlobalFS(), file_path));
  } else {
    in_stream_.reset(new NormalPersistentInStream(GlobalFS(), file_path));
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBasicDataLoaderConf,
                               BasicDataLoaderKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
