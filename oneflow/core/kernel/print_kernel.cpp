#include "oneflow/core/kernel/print_kernel.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/record/ofrecord_encoder.h"

namespace oneflow {

void PrintKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  const auto& conf = op_conf().print_conf();
  const std::string& root_path = conf.print_dir();
  OF_CALL_ONCE(root_path, GlobalFS()->RecursivelyCreateDir(root_path));
  int32_t part_name_suffix_length = conf.part_name_suffix_length();
  std::string num = std::to_string(parallel_ctx->parallel_id());
  int32_t zero_count =
      std::max(part_name_suffix_length - static_cast<int32_t>(num.length()), 0);
  std::string file_path = JoinPath(
      root_path, conf.part_name_prefix() + std::string(zero_count, '0') + num);
  out_stream_.reset(new PersistentOutStream(GlobalFS(), file_path));
}

void PrintKernel::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto GetBlob = [&](int64_t blob_id) -> Blob* {
    return BnInOp2Blob(this->kernel_conf().input_bns(blob_id));
  };
  const auto& conf = op_conf().print_conf();
  int32_t total_blob_num = kernel_conf().input_bns().size();
  const Blob* first_blob = GetBlob(0);
  int64_t max_record_num = first_blob->shape().At(0);
  bool has_data_id_field = first_blob->has_data_id_field();
  bool has_col_num_field = first_blob->has_col_num_field();
  if (has_col_num_field) { TODO(); }
  FOR_RANGE(int32_t, blob_id, 1, total_blob_num) {
    const Blob* cur_blob = GetBlob(blob_id);
    CHECK_EQ(cur_blob->shape().At(0), max_record_num);
    CHECK_EQ(cur_blob->has_data_id_field(), has_data_id_field);
    CHECK_EQ(cur_blob->has_col_num_field(), has_col_num_field);
  }
  OFRecord record;
  FOR_RANGE(int64_t, record_id, 0, max_record_num) {
    record.clear_feature();
    if (has_data_id_field) {
      const char* data_id_str = first_blob->data_id(record_id);
      FOR_RANGE(int32_t, blob_id, 1, kernel_conf().input_bns().size()) {
        CHECK_STREQ(data_id_str, GetBlob(blob_id)->data_id(record_id));
      }
      if (*data_id_str == '\0') { break; }
      OFRecordEncoderIf::EncodeOneDataId(ctx.device_ctx, data_id_str, record);
    }
    FOR_RANGE(int32_t, blob_id, 0, total_blob_num) {
      const Blob* cur_blob = GetBlob(blob_id);
      const PrintRecordConf& cur_print_conf = conf.in(blob_id);
      std::string field_name = cur_print_conf.lbn();
      if (cur_print_conf.has_name()) { field_name = cur_print_conf.name(); }
      CHECK(record.feature().find(field_name) == record.feature().end())
          << "Field " << field_name << " found repeatedly in OfRecord";
      int64_t one_col_elem_num = cur_blob->shape().Count(1);
      Feature& feature = (*(record.mutable_feature()))[field_name];
      GetOFRecordEncoder(cur_print_conf.encode_case().encode_case(),
                         cur_blob->data_type())
          ->EncodeOneCol(ctx.device_ctx, cur_blob, record_id * one_col_elem_num,
                         feature, field_name, one_col_elem_num);
    }
    *out_stream_ << record;
  }
  out_stream_->Flush();
}

COMMAND(AddKernelCreator(OperatorConf::kPrintConf,
                         []() { return new PrintKernel; }));

}  // namespace oneflow
