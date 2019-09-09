#include "oneflow/core/kernel/record_load_kernel.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

void RecordLoadKernel::VirtualKernelInit() {
  const RecordLoadOpConf& record_load_op_conf = op_conf().record_load_conf();
  const RecordLoadKernelConf& record_load_kernel_conf = kernel_conf().record_load_conf();

  int32_t data_part_num = record_load_op_conf.data_part_num();
  std::string data_dir = record_load_op_conf.data_dir();
  std::string part_name_prefix = record_load_op_conf.part_name_prefix();
  int32_t part_name_suffix_length = record_load_op_conf.part_name_suffix_length();
  int32_t parallel_num = record_load_kernel_conf.parallel_num();
  CHECK_LE(parallel_num, data_part_num);
  BalancedSplitter bs(data_part_num, parallel_num);
  Range range = bs.At(record_load_kernel_conf.parallel_id());
  std::vector<std::string> data_paths;
  FOR_RANGE(int32_t, part_id, range.begin(), range.end()) {
    std::string num = std::to_string(part_id);
    int32_t zero_count = std::max(part_name_suffix_length - static_cast<int32_t>(num.length()), 0);
    data_paths.push_back(JoinPath(data_dir, part_name_prefix + std::string(zero_count, '0') + num));
  }
  piece_size_in_one_loader_ = record_load_kernel_conf.device_piece_size();
  if (this->job_desc().IsTrain()) {
    const size_t num_max_read = GetMaxVal<int64_t>();
    bool save_to_local = Global<const IOConf>::Get()->save_downloaded_file_to_local_fs();
    in_stream_.reset(new PersistentInStream(DataFS(), data_paths, true, save_to_local));
    if (record_load_op_conf.has_random_shuffle_conf()) {
      const int32_t shuffle_buffer_size = record_load_op_conf.random_shuffle_conf().buffer_size();
      CHECK_GT(shuffle_buffer_size, 0);
      record_reader_.reset(new RandomShuffleOFRecordReader(
          in_stream_.get(), static_cast<size_t>(shuffle_buffer_size), num_max_read));
    } else {
      record_reader_.reset(new NaiveOFRecordReader(in_stream_.get(), num_max_read));
    }
  } else {
    in_stream_.reset(new PersistentInStream(DataFS(), data_paths, false, false));
    if (record_load_op_conf.has_random_shuffle_conf()) {
      const int32_t shuffle_buffer_size = record_load_op_conf.random_shuffle_conf().buffer_size();
      CHECK_GT(shuffle_buffer_size, 0);
      record_reader_.reset(new RandomShuffleOFRecordReader(
          in_stream_.get(), static_cast<size_t>(shuffle_buffer_size)));
    } else {
      record_reader_.reset(new NaiveOFRecordReader(in_stream_.get()));
    }
  }
}

void RecordLoadKernel::Forward(const KernelCtx& ctx,
                               std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto status = static_cast<RecordLoadStatus*>(ctx.other);
  status->record_num = record_reader_->Read(static_cast<size_t>(piece_size_in_one_loader_),
                                            BnInOp2Blob("out")->mut_dptr<OFRecord>());
  BnInOp2Blob("out")->set_record_num(status->record_num);
  if (status->record_num < piece_size_in_one_loader_) { status->is_eof = true; }
}

REGISTER_KERNEL(OperatorConf::kRecordLoadConf, RecordLoadKernel);

}  // namespace oneflow
