#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/hadoop/hadoop_file_system.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

int64_t JobDesc::MachineID4MachineName(const std::string& machine_name) const {
  auto it = machine_name2machine_id_.find(machine_name);
  CHECK(it != machine_name2machine_id_.end()) << "Undefined machine name: " << machine_name;
  return it->second;
}
const std::string& JobDesc::MachineName4MachineId(int64_t machine_id) const {
  return machine_id2machine_name_.at(machine_id);
}

int64_t JobDesc::piece_num_of_experiment_phase() const {
  return job_conf_.other().piece_num_of_experiment_phase();
}

size_t JobDesc::persistence_buf_byte() const {
  return job_conf_.other().persistence_buf_mbyte() * 1024 * 1024;
}

size_t JobDesc::reserved_host_mem_byte() const {
  return job_conf_.other().reserved_host_mem_mbyte() * 1024 * 1024;
}

size_t JobDesc::reserved_device_mem_byte() const {
  return job_conf_.other().reserved_device_mem_mbyte() * 1024 * 1024;
}

bool JobDesc::save_downloaded_file_to_local_fs() const {
  return job_conf_.other().save_downloaded_file_to_local_fs();
}

size_t JobDesc::rdma_mem_block_byte() const {
  return job_conf_.other().rdma_mem_block_mbyte() * 1024 * 1024;
}

size_t JobDesc::rdma_recv_msg_buf_byte() const {
  return job_conf_.other().rdma_recv_msg_buf_mbyte() * 1024 * 1024;
}

const std::string& JobDesc::MdSaveSnapshotsPath() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().model_save_snapshots_path();
}
int32_t JobDesc::NumOfBatchesInSnapshot() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().num_of_batches_in_snapshot();
}
int64_t JobDesc::TotalBatchNum() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().total_batch_num();
}
const InitializerConf* JobDesc::DefaultInitializerConf() const {
  CHECK(IsTrain());
  return GetMsgPtrFromPbMessage<InitializerConf>(job_conf_.other().train_conf(),
                                                 "default_initializer_conf");
}
int32_t JobDesc::PieceNumOfPrintLoss() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().piece_num_of_print_loss();
}
int32_t JobDesc::PieceNumOfPrintAccuracy() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().piece_num_of_print_accuracy();
}
int64_t JobDesc::BatchSize() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().batch_size();
}
int64_t JobDesc::NumOfPiecesInBatch() const {
  CHECK_EQ(BatchSize() % PieceSize(), 0);
  return BatchSize() / PieceSize();
}
float JobDesc::L1() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().l1();
}

float JobDesc::L2() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().l2();
}

int32_t JobDesc::DataPartNum() const { return job_conf_.other().data_part_num(); }

JobDesc::JobDesc(const std::string& job_conf_filepath) {
  if (TryParseProtoFromTextFile(job_conf_filepath, &job_conf_) == false) {
    JobConf2 job_conf;
    std::string job_conf_path = Dirname(job_conf_filepath);
    ParseProtoFromTextFile(job_conf_filepath, &job_conf);

    ParseProtoFromTextFile(job_conf_path == Dirname(job_conf.net())
                               ? job_conf.net()
                               : JoinPath(job_conf_path, job_conf.net()),
                           job_conf_.mutable_net());
    ParseProtoFromTextFile(job_conf_path == Dirname(job_conf.net())
                               ? job_conf.net()
                               : JoinPath(job_conf_path, job_conf.resource()),
                           job_conf_.mutable_resource());
    ParseProtoFromTextFile(job_conf_path == Dirname(job_conf.net())
                               ? job_conf.net()
                               : JoinPath(job_conf_path, job_conf.placement()),
                           job_conf_.mutable_placement());
    ParseProtoFromTextFile(job_conf_path == Dirname(job_conf.net())
                               ? job_conf.net()
                               : JoinPath(job_conf_path, job_conf.other()),
                           job_conf_.mutable_other());
  }

  SplitDecodeOps();
  AddRecordLoadOps();
#ifndef WITH_RDMA
  CHECK_EQ(job_conf_.other().use_rdma(), false) << "Please compile ONEFLOW with RDMA";
#endif
  int64_t piece_exp = job_conf_.other().piece_num_of_experiment_phase();
  if (job_conf_.other().has_train_conf()) {
    TrainConf* train_conf = job_conf_.mutable_other()->mutable_train_conf();
    if (train_conf->piece_num_of_print_loss() == -1) {
      train_conf->set_piece_num_of_print_loss(NumOfPiecesInBatch());
    }
    if (train_conf->piece_num_of_print_accuracy() == -1) {
      train_conf->set_piece_num_of_print_accuracy(NumOfPiecesInBatch());
    }
    if (piece_exp == -1) { piece_exp = 19 * NumOfPiecesInBatch(); }
    piece_exp = std::max(piece_exp, NumOfPiecesInBatch());
    piece_exp = std::max(piece_exp, train_conf->piece_num_of_print_loss());
    piece_exp = std::min(piece_exp, train_conf->total_batch_num() * NumOfPiecesInBatch());
  } else {
    if (piece_exp == -1) { piece_exp = 19; }
  }
  LOG(INFO) << "Set piece_num_of_experiment_phase " << piece_exp;
  job_conf_.mutable_other()->set_piece_num_of_experiment_phase(piece_exp);
#ifndef WITH_CUDA
  CHECK_EQ(job_conf_.resource().gpu_device_num(), 0);
#endif
  int64_t machine_num = job_conf_.resource().machine_size();
  for (int64_t i = 0; i < machine_num; ++i) {
    const std::string& machine_name = job_conf_.resource().machine(i).name();
    CHECK(machine_name2machine_id_.emplace(machine_name, i).second);
    CHECK(machine_id2machine_name_.emplace(i, machine_name).second);
  }
}

void JobDesc::SplitDecodeOps() {
  std::vector<OperatorConf> gen_op_confs;
  for (OperatorConf& op_conf : *(job_conf_.mutable_net()->mutable_op())) {
    if (op_conf.has_decode_ofrecord_conf() == false) { continue; }
    if (op_conf.decode_ofrecord_conf().blob_size() == 1) { continue; }
    const DecodeOFRecordOpConf& decode_conf = op_conf.decode_ofrecord_conf();
    PbRpf<BlobConf>* blobs = op_conf.mutable_decode_ofrecord_conf()->mutable_blob();
    Erase<PbRpf<BlobConf>>(
        *blobs,
        [&](const BlobConf& blob_conf) -> bool { return blob_conf.max_sequence_size() > 1; },
        [&](const BlobConf& blob_conf) {
          gen_op_confs.emplace_back(op_conf);
          DecodeOFRecordOpConf* gen_decode_conf =
              gen_op_confs.back().mutable_decode_ofrecord_conf();
          *gen_decode_conf = decode_conf;
          gen_decode_conf->clear_blob();
          *gen_decode_conf->add_blob() = blob_conf;
        });
  }
  // TODO: erase decode op which has no blob any more
  for (OperatorConf& gen_op_conf : gen_op_confs) {
    *(job_conf_.mutable_net()->add_op()) = gen_op_conf;
  }
}

void JobDesc::AddRecordLoadOps() {
  HashMap<std::pair<std::string, std::string>, std::vector<OperatorConf*>> data_info2decode_ops;
  HashMap<std::pair<std::string, std::string>, int32_t> data_info2suffix_length;
  size_t op_num = job_conf_.net().op_size();
  FOR_RANGE(size_t, idx, 0, op_num) {
    OperatorConf* op_conf = job_conf_.mutable_net()->mutable_op()->Mutable(idx);
    if (op_conf->has_decode_ofrecord_conf() == false) { continue; }
    const DecodeOFRecordOpConf& decode_conf = op_conf->decode_ofrecord_conf();
    if (decode_conf.blob_size() == 0) { continue; }
    std::pair<std::string, std::string> data_info = {decode_conf.data_dir(),
                                                     decode_conf.part_name_prefix()};
    data_info2decode_ops[data_info].emplace_back(op_conf);
    int32_t part_name_suffix_length = decode_conf.part_name_suffix_length();
    if (data_info2suffix_length.find(data_info) != data_info2suffix_length.end()) {
      CHECK_EQ(data_info2suffix_length[data_info], part_name_suffix_length);
    } else {
      data_info2suffix_length[data_info] = part_name_suffix_length;
    }
  }

  HashMap<std::string, const ParallelConf*> name2parallel_conf;
  for (const PlacementGroup& p_group : job_conf_.placement().placement_group()) {
    for (const std::string& op_name : p_group.op_set().op_name()) {
      CHECK(name2parallel_conf.emplace(op_name, &p_group.parallel_conf()).second);
    }
  }

  for (const auto& pair : data_info2decode_ops) {
    std::vector<const ParallelConf*> parallel_confs;
    for (const OperatorConf* op_conf : pair.second) {
      auto op_parallel_conf_it = name2parallel_conf.find(op_conf->name());
      CHECK(op_parallel_conf_it != name2parallel_conf.end());
      auto iter = std::find_if(
          parallel_confs.begin(), parallel_confs.end(), [&](const ParallelConf* parallel_conf) {
            PbMd message_diff;
            return message_diff.Equivalent(*parallel_conf, *(op_parallel_conf_it->second));
          });
      if (iter == parallel_confs.end()) {
        parallel_confs.emplace_back(op_parallel_conf_it->second);
      }
    }
    LOG_IF(WARNING, parallel_confs.size() > 1)
        << "Operators sharing the same data information belong to different placement groups";
    for (const ParallelConf* parallel_conf : parallel_confs) {
      std::string record_load_op_name = "loader" + NewUniqueId();
      std::string record_load_out_name = "out";
      std::string record_load_lbi_name = record_load_op_name + "/" + record_load_out_name;
      OperatorConf* op = job_conf_.mutable_net()->add_op();
      RecordLoadOpConf* record_load_op = op->mutable_record_load_conf();
      op->set_name(record_load_op_name);
      record_load_op->set_out(record_load_out_name);
      record_load_op->set_data_dir(pair.first.first);
      record_load_op->set_part_name_prefix(pair.first.second);
      record_load_op->set_part_name_suffix_length(data_info2suffix_length.at(pair.first));
      PlacementGroup* p_group = job_conf_.mutable_placement()->add_placement_group();
      *(p_group->mutable_op_set()->add_op_name()) = record_load_op_name;
      *(p_group->mutable_parallel_conf()) = *parallel_conf;
      for (OperatorConf* op : pair.second) {
        std::string op_name = op->name();
        auto op_parallel_conf_it = name2parallel_conf.find(op_name);
        CHECK(op_parallel_conf_it != name2parallel_conf.end());
        PbMd message_diff;
        if (!message_diff.Equivalent(*parallel_conf, *(op_parallel_conf_it->second))) { continue; }
        op->mutable_decode_ofrecord_conf()->set_in(record_load_lbi_name);
      }
    }
  }
}

}  // namespace oneflow
