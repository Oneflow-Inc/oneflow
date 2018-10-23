#include "oneflow/core/kernel/blob_dump_kernel.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/record/ofrecord_encoder.h"

namespace oneflow {

void BlobDumpKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  const auto& conf = op_conf().blob_dump_conf();
  const std::string& root_path = conf.dump_dir();
  OfCallOnce(root_path, SnapshotFS(), &fs::FileSystem::RecursivelyCreateDir);
  int32_t part_name_suffix_length = conf.part_name_suffix_length();
  std::string num = std::to_string(parallel_ctx->parallel_id());
  int32_t zero_count = std::max(part_name_suffix_length - static_cast<int32_t>(num.length()), 0);
  std::string file_path =
      JoinPath(root_path, conf.part_name_prefix() + std::string(zero_count, '0') + num);
  out_stream_.reset(new PersistentOutStream(SnapshotFS(), file_path));
}

void BlobDumpKernel::Forward(const KernelCtx& ctx,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (HasEmptyShapeBlob(this->op_attribute().input_bns(), BnInOp2Blob)) { return; }
  auto GetBlob = [&](int64_t blob_id) -> Blob* {
    return BnInOp2Blob("in_" + std::to_string(blob_id));
  };
  const auto& conf = op_conf().blob_dump_conf();
  int32_t total_blob_num = op_attribute().input_bns_size();

  OFRecord record;
  FOR_RANGE(int32_t, blob_id, 0, total_blob_num) {
    const Blob* cur_blob = GetBlob(blob_id);
    const PrintRecordConf& cur_dump_conf = conf.in(blob_id);
    std::string field_name = cur_dump_conf.lbn();
    if (cur_dump_conf.has_name()) { field_name = cur_dump_conf.name(); }
    CHECK(record.feature().find(field_name) == record.feature().end())
        << "Field " << field_name << " found repeatedly in OfRecord";
    int64_t one_col_elem_num = cur_blob->shape().elem_cnt();
    Feature& feature = (*(record.mutable_feature()))[field_name];
    GetOFRecordEncoder(cur_dump_conf.encode_case().encode_case(), cur_blob->data_type())
        ->EncodeOneCol(ctx.device_ctx, cur_blob, 0, feature, field_name, one_col_elem_num);
  }
  *out_stream_ << record;
  out_stream_->Flush();
}

REGISTER_KERNEL(OperatorConf::kBlobDumpConf, BlobDumpKernel);

}  // namespace oneflow
