#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/hadoop/hadoop_file_system.h"
#include "oneflow/core/graph/graph.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_builder.h"

namespace oneflow {

namespace {

const TrainConf& GetTrainConf(const JobConf& job_conf) {
  if (job_conf.other().has_predict_conf()
      && job_conf.other().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    return job_conf.other().predict_conf().tmp_split_fw_bw_train_conf();
  }
  CHECK(job_conf.other().has_train_conf());
  return job_conf.other().train_conf();
}

}  // namespace

std::function<const ParallelConf*(const std::string&)> MakeGetterParallelConf4OpName(
    const Placement& placement) {
  auto op_name2parallel_conf = std::make_shared<HashMap<std::string, const ParallelConf*>>();
  for (const auto& placement_group : placement.placement_group()) {
    for (const std::string& op_name : placement_group.op_set().op_name()) {
      const ParallelConf* parallel_conf = &placement_group.parallel_conf();
      CHECK(op_name2parallel_conf->emplace(op_name, parallel_conf).second);
    }
  }
  return [op_name2parallel_conf](const std::string& op_name) {
    return op_name2parallel_conf->at(op_name);
  };
}

int64_t JobDesc::all_reduce_group_min_byte() const {
  int64_t ret = job_conf_.other().all_reduce_group_min_mbyte() * 1024 * 1024;
  CHECK_GT(ret, 0);
  return ret;
}

float JobDesc::all_reduce_group_size_warmup() const {
  float ret = job_conf_.other().all_reduce_group_size_warmup();
  CHECK_GT(ret, 1);
  return ret;
}

int64_t JobDesc::all_reduce_group_num() const {
  int64_t ret = job_conf_.other().all_reduce_group_num();
  CHECK_GT(ret, 0);
  return ret;
}

float JobDesc::all_reduce_lazy_ratio() const {
  float ratio = job_conf_.other().all_reduce_lazy_ratio();
  CHECK_GE(ratio, 0.0);
  CHECK_LE(ratio, 1.0);
  return ratio;
}

bool JobDesc::all_reduce_fp16() const { return job_conf_.other().all_reduce_fp16(); }

bool JobDesc::use_boxing_v2() const { return job_conf_.other().use_boxing_v2(); }

int64_t JobDesc::piece_num_of_experiment_phase() const {
  return job_conf_.other().exp_run_conf().piece_num_of_experiment_phase();
}

bool JobDesc::enable_experiment_run() const {
  return job_conf_.other().exp_run_conf().enable_experiment_run();
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
  if (IsPredict()
      && Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    return job_conf_.other()
        .predict_conf()
        .tmp_split_fw_bw_train_conf()
        .model_save_snapshots_path();
  } else {
    return job_conf_.other().train_conf().model_save_snapshots_path();
  }
}

int32_t JobDesc::NumOfBatchesInSnapshot() const {
  return GetTrainConf(job_conf_).num_of_batches_in_snapshot();
}
int64_t JobDesc::TotalBatchNum() const { return job_conf_.other().total_batch_num(); }
const InitializerConf* JobDesc::DefaultInitializerConf() const {
  return GetMsgPtrFromPbMessage<InitializerConf>(GetTrainConf(job_conf_),
                                                 "default_initializer_conf");
}
int32_t JobDesc::PieceNumOfPrintLoss() const {
  return job_conf_.other().train_conf().piece_num_of_print_loss();
}
int32_t JobDesc::PieceNumOfPrintAccuracy() const {
  return job_conf_.other().train_conf().piece_num_of_print_accuracy();
}
int64_t JobDesc::BatchSize() const { return GetTrainConf(job_conf_).batch_size(); }
int64_t JobDesc::NumOfPiecesInBatch() const {
  if (IsPredict()
      && !Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    return 1;
  }
  CHECK_EQ(BatchSize() % RecordPieceSize(), 0);
  return BatchSize() / RecordPieceSize();
}
float JobDesc::primary_lr() const { return GetTrainConf(job_conf_).primary_lr(); }
float JobDesc::secondary_lr() const { return GetTrainConf(job_conf_).secondary_lr(); }
float JobDesc::weight_l1() const { return GetTrainConf(job_conf_).weight_l1(); }
float JobDesc::bias_l1() const { return GetTrainConf(job_conf_).bias_l1(); }
float JobDesc::weight_l2() const { return GetTrainConf(job_conf_).weight_l2(); }
float JobDesc::bias_l2() const { return GetTrainConf(job_conf_).bias_l2(); }

int32_t JobDesc::DataPartNum() const { return job_conf_.other().data_part_num(); }

const FileSystemConf& JobDesc::data_fs_conf() const { return job_conf_.other().data_fs_conf(); }
const FileSystemConf& JobDesc::snapshot_fs_conf() const {
  return job_conf_.other().snapshot_fs_conf();
}

bool JobDesc::enable_write_snapshot() const {
  if (IsTrain()
      || (IsPredict() && job_conf_.other().predict_conf().has_tmp_split_fw_bw_train_conf())) {
    return job_conf_.other().enable_write_snapshot();
  } else {
    return false;
  }
}

JobDesc::JobDesc(const std::string& job_conf_filepath) {
  ParseProtoFromTextFile(job_conf_filepath, &job_conf_);
  Init();
}

void JobDesc::Init() {
  SanityCheck();
  SplitDecodeOps();
  AddRecordLoadOps();
#ifndef WITH_RDMA
  CHECK_EQ(job_conf_.other().use_rdma(), false) << "Please compile ONEFLOW with RDMA";
#endif
#ifndef WITH_CUDA
  CHECK_EQ(job_conf_.other().enable_nccl(), false) << "Please compile ONEFLOW with NCCL";
#endif  // WITH_CUDA
  int64_t piece_exp = job_conf_.other().exp_run_conf().piece_num_of_experiment_phase();
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
    piece_exp = std::min(piece_exp, job_conf_.other().total_batch_num() * NumOfPiecesInBatch());
  } else {
    if (piece_exp == -1) { piece_exp = 19; }
  }
  LOG(INFO) << "Set piece_num_of_experiment_phase " << piece_exp;
  job_conf_.mutable_other()->mutable_exp_run_conf()->set_piece_num_of_experiment_phase(piece_exp);
#ifndef WITH_CUDA
  CHECK_EQ(job_conf_.resource().gpu_device_num(), 0);
#endif
}

void JobDesc::SanityCheck() {
  int64_t machine_num = job_conf_.resource().machine_size();
  FOR_RANGE(int64_t, i, 0, machine_num) { CHECK_EQ(job_conf_.resource().machine(i).id(), i); }
}

int64_t JobDesc::GetMachineId(const std::string& addr) const {
  int64_t machine_id = -1;
  auto resource_conf = job_conf_.resource();
  int64_t machine_num = resource_conf.machine_size();
  FOR_RANGE(int64_t, i, 0, machine_num) {
    if (addr == resource_conf.machine(i).addr()) {
      machine_id = i;
      break;
    }
  }
  CHECK_GE(machine_id, 0);
  CHECK_LT(machine_id, machine_num);
  return machine_id;
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
  HashMap<std::pair<std::string, std::string>, const RandomShuffleConf*> data_info2shuffle_conf;
  size_t op_num = job_conf_.net().op_size();
  FOR_RANGE(size_t, idx, 0, op_num) {
    OperatorConf* op_conf = job_conf_.mutable_net()->mutable_op()->Mutable(idx);
    if (op_conf->has_decode_ofrecord_conf() == false) { continue; }
    const DecodeOFRecordOpConf& decode_conf = op_conf->decode_ofrecord_conf();
    if (decode_conf.has_in()) { continue; }
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
    const RandomShuffleConf* shuffle_conf =
        decode_conf.has_random_shuffle_conf() ? &decode_conf.random_shuffle_conf() : nullptr;
    if (data_info2shuffle_conf.find(data_info) != data_info2shuffle_conf.end()) {
      if (shuffle_conf == nullptr) {
        CHECK(data_info2shuffle_conf.at(data_info) == nullptr);
      } else {
        CHECK(data_info2shuffle_conf.at(data_info) != nullptr);
        CHECK_EQ(data_info2shuffle_conf.at(data_info)->buffer_size(), shuffle_conf->buffer_size());
      }
    } else {
      CHECK(data_info2shuffle_conf.emplace(data_info, shuffle_conf).second);
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
      if (data_info2shuffle_conf.at(pair.first) != nullptr) {
        *record_load_op->mutable_random_shuffle_conf() = *data_info2shuffle_conf.at(pair.first);
      }
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
