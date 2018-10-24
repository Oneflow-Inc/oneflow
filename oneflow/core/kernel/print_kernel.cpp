#include "oneflow/core/kernel/print_kernel.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/record/ofrecord_encoder.h"

namespace oneflow {

struct RecordOffsets {
  HashMap<int64_t, std::vector<int64_t>> blob_id2offsets;
};

namespace {

void SplitBns(PbRpf<std::string>* bns_with_record_ids, PbRpf<std::string>* bns_without_record_ids,
              const std::function<Blob*(const std::string&)> BnInOp2Blob,
              const PbRpf<std::string>& bns) {
  for (const auto& bn : bns) {
    if (BnInOp2Blob(bn)->has_record_id_in_device_piece_field()) {
      *bns_with_record_ids->Add() = bn;
    } else {
      *bns_without_record_ids->Add() = bn;
    }
  }
}

void CheckSameDim0Size(const PbRpf<std::string>& bns,
                       const std::function<Blob*(const std::string&)> BnInOp2Blob) {
  FOR_RANGE(int32_t, i, 1, bns.size()) {
    CHECK_EQ(BnInOp2Blob(bns.Get(0))->shape().At(0), BnInOp2Blob(bns.Get(i))->shape().At(0));
  }
}

void CheckRecordIdInDevicePieceIsValid(const Blob* blob, size_t max_size) {
  FOR_RANGE(int64_t, i, 0, blob->shape().At(0)) {
    CHECK_LT(blob->record_id_in_device_piece(i), max_size);
  }
}

void CheckRecordIds(const PbRpf<std::string>& bns_with_record_ids,
                    const PbRpf<std::string>& bns_without_record_ids,
                    const std::function<Blob*(const std::string&)> BnInOp2Blob) {
  CHECK(!(bns_with_record_ids.empty() && bns_without_record_ids.empty()));
  if (bns_without_record_ids.size() > 0 && bns_with_record_ids.size() > 0) {
    size_t max_size = BnInOp2Blob(bns_without_record_ids.Get(0))->shape().At(0);
    CheckRecordIdInDevicePieceIsValid(BnInOp2Blob(bns_with_record_ids.Get(0)), max_size);
  }
}

void InitRecordOffsets(std::map<int64_t, RecordOffsets>* record_id2record_offsets,
                       const std::function<Blob*(const std::string&)> BnInOp2Blob,
                       const PbRpf<std::string>& bns) {
  PbRpf<std::string> bns_with_record_ids;
  PbRpf<std::string> bns_without_record_ids;
  SplitBns(&bns_with_record_ids, &bns_without_record_ids, BnInOp2Blob, bns);
  CheckSameRecordIdInDevicePiece(bns_with_record_ids, BnInOp2Blob);
  CheckSameDim0Size(bns_without_record_ids, BnInOp2Blob);
  CheckRecordIds(bns_with_record_ids, bns_without_record_ids, BnInOp2Blob);
  FOR_RANGE(int64_t, blob_id, 0, bns.size()) {
    const Blob* blob = BnInOp2Blob(bns.Get(blob_id));
    FOR_RANGE(int64_t, i, 0, blob->shape().At(0)) {
      int64_t record_id = blob->record_id_in_device_piece(i);
      std::vector<int64_t>* vec = &(*record_id2record_offsets)[record_id].blob_id2offsets[blob_id];
      vec->push_back(i * blob->shape().Count(1));
    }
  }
}

}  // namespace

void PrintKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  const auto& conf = op_conf().print_conf();
  const std::string& root_path = conf.print_dir();
  OfCallOnce(root_path, SnapshotFS(), &fs::FileSystem::RecursivelyCreateDir);
  int32_t part_name_suffix_length = conf.part_name_suffix_length();
  std::string num = std::to_string(parallel_ctx->parallel_id());
  int32_t zero_count = std::max(part_name_suffix_length - static_cast<int32_t>(num.length()), 0);
  std::string file_path =
      JoinPath(root_path, conf.part_name_prefix() + std::string(zero_count, '0') + num);
  out_stream_.reset(new PersistentOutStream(SnapshotFS(), file_path));
}

void PrintKernel::Forward(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (HasEmptyShapeBlob(this->op_attribute().input_bns(), BnInOp2Blob)) { return; }
  std::map<int64_t, RecordOffsets> record_id2record_offsets;
  InitRecordOffsets(&record_id2record_offsets, BnInOp2Blob, this->op_attribute().input_bns());
  auto GetBlob = [&](int64_t blob_id) -> Blob* {
    return BnInOp2Blob(this->op_attribute().input_bns(blob_id));
  };
  const auto& conf = op_conf().print_conf();
  OFRecord record;
  for (const auto& record_id7record_offsets : record_id2record_offsets) {
    record.clear_feature();
    for (const auto& pair : record_id7record_offsets.second.blob_id2offsets) {
      int64_t blob_id = pair.first;
      const std::vector<int64_t>& offsets = pair.second;
      const Blob* cur_blob = GetBlob(blob_id);
      const PrintRecordConf& cur_print_conf = conf.in(blob_id);
      std::string field_name = cur_print_conf.lbn();
      if (cur_print_conf.has_name()) { field_name = cur_print_conf.name(); }
      CHECK(record.feature().find(field_name) == record.feature().end())
          << "Field " << field_name << " found repeatedly in OfRecord";
      int64_t one_col_elem_num = cur_blob->shape().Count(1);
      Feature& feature = (*(record.mutable_feature()))[field_name];
      if (cur_blob->has_record_id_in_device_piece_field()) {
        GetOFRecordEncoder(cur_print_conf.encode_case().encode_case(), cur_blob->data_type())
            ->EncodeMultiCol(ctx.device_ctx, cur_blob, offsets, feature, field_name,
                             one_col_elem_num);
      } else {
        CHECK_EQ(offsets.size(), 1);
        GetOFRecordEncoder(cur_print_conf.encode_case().encode_case(), cur_blob->data_type())
            ->EncodeOneCol(ctx.device_ctx, cur_blob, offsets.at(0), feature, field_name,
                           one_col_elem_num);
      }
    }
    *out_stream_ << record;
  }
  out_stream_->Flush();
}

REGISTER_KERNEL(OperatorConf::kPrintConf, PrintKernel);

}  // namespace oneflow
