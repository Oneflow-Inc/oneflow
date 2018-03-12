#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/hadoop/hadoop_file_system.h"

namespace oneflow {

const std::string& JobDesc::MdLoadSnapshotPath() {
  return job_conf_.model_load_snapshot_path();
}
size_t JobDesc::SizeOfOneDataId() const {
  return job_conf_.max_data_id_length() * sizeof(char);
}
int32_t JobDesc::DecodeWorkerNum() const {
  CHECK(resource_.has_decode_worker_num());
  return resource_.decode_worker_num();
}
void JobDesc::set_decode_worker_num(int32_t decode_worker_num) {
  resource_.set_decode_worker_num(decode_worker_num);
}
int32_t JobDesc::BoxingWorkerNum() const {
  CHECK(resource_.has_boxing_worker_num());
  return resource_.boxing_worker_num();
}
void JobDesc::set_boxing_worker_num(int32_t boxing_worker_num) {
  resource_.set_boxing_worker_num(boxing_worker_num);
}
int32_t JobDesc::CommNetWorkerNum() const {
  CHECK(resource_.has_comm_net_worker_num());
  return resource_.comm_net_worker_num();
}
void JobDesc::set_comm_net_worker_num(int32_t comm_net_worker_num) {
  resource_.set_comm_net_worker_num(comm_net_worker_num);
}
int32_t JobDesc::PersistenceWorkerNum() const {
  CHECK(resource_.has_persistence_worker_num());
  return resource_.persistence_worker_num();
}
void JobDesc::set_persistence_worker_num(int32_t persistence_worker_num) {
  resource_.set_persistence_worker_num(persistence_worker_num);
}
int32_t JobDesc::ParallelPieceSize() const {
  return job_conf_.data_part_num() * SinglePieceSize();
}
int64_t JobDesc::piece_num_of_experiment_phase() const {
  return job_conf_.piece_num_of_experiment_phase();
}

const std::string& JobDesc::MdSaveSnapshotsPath() const {
  CHECK(IsTrain());
  return job_conf_.train_conf().model_save_snapshots_path();
}
int32_t JobDesc::NumOfBatchesInSnapshot() const {
  CHECK(IsTrain());
  return job_conf_.train_conf().num_of_batches_in_snapshot();
}
int32_t JobDesc::NumOfPiecesInBatch() const {
  CHECK(IsTrain());
  return job_conf_.train_conf().num_of_pieces_in_batch();
}
int32_t JobDesc::Staleness() const {
  CHECK(IsTrain());
  return job_conf_.train_conf().staleness();
}
int64_t JobDesc::TotalBatchNum() const {
  CHECK(IsTrain());
  return job_conf_.train_conf().total_batch_num();
}
const InitializerConf* JobDesc::DefaultInitializerConf() const {
  CHECK(IsTrain());
  return OF_PB_POINTER_GET(job_conf_.train_conf(), default_initializer_conf);
}
int32_t JobDesc::PieceNumOfPrintLoss() const {
  CHECK(IsTrain());
  return job_conf_.train_conf().piece_num_of_print_loss();
}
int32_t JobDesc::BatchSize() const {
  return NumOfPiecesInBatch() * ParallelPieceSize();
}
float JobDesc::L1() const {
  CHECK(IsTrain());
  return job_conf_.train_conf().l1();
}

float JobDesc::L2() const {
  CHECK(IsTrain());
  return job_conf_.train_conf().l2();
}

JobDesc::JobDesc(const JobDescProto& job_desc) {
  job_conf_ = job_desc.job_conf();
  dlnet_conf_ = job_desc.dlnet_conf();
  resource_ = job_desc.resource();
  placement_ = job_desc.placement();
  SplitDecodeOps();
#ifndef WITH_RDMA
  CHECK_EQ(job_conf_.use_rdma(), false) << "Please compile ONEFLOW with RDMA";
#endif
  int64_t piece_experiment = job_conf_.piece_num_of_experiment_phase();
  if (job_conf_.has_train_conf()) {
    TrainConf* train_conf = job_conf_.mutable_train_conf();
    piece_experiment = std::max<int64_t>(
        piece_experiment, train_conf->num_of_batches_in_snapshot()
                              * train_conf->num_of_pieces_in_batch());
    piece_experiment = std::max<int64_t>(piece_experiment,
                                         train_conf->piece_num_of_print_loss());
    if (piece_experiment != job_conf_.piece_num_of_experiment_phase()) {
      LOG(WARNING) << "Set piece_num_of_experiment_phase " << piece_experiment;
      job_conf_.set_piece_num_of_experiment_phase(piece_experiment);
    }
    if (train_conf->has_piece_num_of_print_loss() == false) {
      train_conf->set_piece_num_of_print_loss(
          train_conf->num_of_pieces_in_batch());
    }
  }
#ifndef WITH_CUDA
  CHECK_EQ(resource_.gpu_device_num(), 0);
#endif
}

void JobDesc::SplitDecodeOps() {
  FOR_RANGE(int32_t, i, 0, dlnet_conf_.op_size()) {
    const OperatorConf& op_conf = dlnet_conf_.op(i);
    if (op_conf.has_decode_ofrecord_conf() == false) { continue; }
    if (op_conf.decode_ofrecord_conf().blob_size() == 1) { continue; }
    const DecodeOFRecordOpConf& decode_conf = op_conf.decode_ofrecord_conf();
    HashMap<int32_t, std::vector<int32_t>> max_seq_size2blob_idx;
    FOR_RANGE(int32_t, j, 0, decode_conf.blob_size()) {
      int32_t max_seq_size = decode_conf.blob(j).max_sequence_size();
      if (max_seq_size2blob_idx.find(max_seq_size)
          == max_seq_size2blob_idx.end()) {
        CHECK(
            max_seq_size2blob_idx.emplace(max_seq_size, std::vector<int32_t>{j})
                .second);
      } else {
        max_seq_size2blob_idx.at(max_seq_size).push_back(j);
      }
    }
    if (max_seq_size2blob_idx.size() == 1) { continue; }
    int32_t split_cnt = 0;
    for (auto& pair : max_seq_size2blob_idx) {
      DecodeOFRecordOpConf gen_decode_conf(decode_conf);
      gen_decode_conf.clear_blob();
      FOR_RANGE(size_t, j, 0, pair.second.size()) {
        *gen_decode_conf.add_blob() = decode_conf.blob(pair.second.at(j));
      }
      OperatorConf gen_op_conf(op_conf);
      *gen_op_conf.mutable_decode_ofrecord_conf() = gen_decode_conf;
      split_cnt++;
      if (split_cnt == max_seq_size2blob_idx.size()) {
        *dlnet_conf_.mutable_op(i) = gen_op_conf;
      } else {
        *dlnet_conf_.add_op() = gen_op_conf;
      }
    }
  }
}

}  // namespace oneflow
