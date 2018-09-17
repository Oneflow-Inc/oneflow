#include "oneflow/core/kernel/record_load_kernel.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

void RecordLoadKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  const RecordLoadOpConf& record_load_conf = op_conf().record_load_conf();

  int32_t data_part_num = Global<JobDesc>::Get()->DataPartNum();
  std::string data_dir = record_load_conf.data_dir();
  std::string part_name_prefix = record_load_conf.part_name_prefix();
  int32_t part_name_suffix_length = record_load_conf.part_name_suffix_length();
  int32_t parallel_num = parallel_ctx->parallel_num();
  CHECK_LE(parallel_num, data_part_num);
  BalancedSplitter bs(data_part_num, parallel_num);
  Range range = bs.At(parallel_ctx->parallel_id());
  std::vector<std::string> data_paths;
  FOR_RANGE(int32_t, part_id, range.begin(), range.end()) {
    std::string num = std::to_string(part_id);
    int32_t zero_count = std::max(part_name_suffix_length - static_cast<int32_t>(num.length()), 0);
    data_paths.push_back(JoinPath(data_dir, part_name_prefix + std::string(zero_count, '0') + num));
  }
  if (Global<JobDesc>::Get()->IsTrain()) {
    in_stream_.reset(new PersistentInStream(
        DataFS(), data_paths, true, Global<JobDesc>::Get()->save_downloaded_file_to_local_fs()));
  } else {
    in_stream_.reset(new PersistentInStream(DataFS(), data_paths, false, false));
  }
  int64_t global_piece_size = Global<JobDesc>::Get()->PieceSize();
  CHECK_EQ(global_piece_size % parallel_ctx->parallel_num(), 0);
  piece_size_in_one_loader_ = global_piece_size / parallel_ctx->parallel_num();
  loaded_cnt_ = 0;
}

void RecordLoadKernel::Forward(const KernelCtx& ctx,
                               std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto status = static_cast<RecordLoadStatus*>(ctx.other);
  Blob* out_blob = BnInOp2Blob("out");
  RecordBlob<OFRecord> record_blob(out_blob);
  if (!Global<JobDesc>::Get()->use_synthetic_data() || loaded_cnt_ < 2) {
    record_blob.ReadFrom(in_stream_.get());
    ++loaded_cnt_;
  }
  status->record_num = record_blob.record_num();
  if (status->record_num < piece_size_in_one_loader_) { status->is_eof = true; }
}

REGISTER_KERNEL(OperatorConf::kRecordLoadConf, RecordLoadKernel);

}  // namespace oneflow
