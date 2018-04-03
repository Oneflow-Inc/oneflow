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
int32_t JobDesc::CommNetWorkerNum() const {
  return resource_.comm_net_worker_num();
}
int32_t JobDesc::PersistenceWorkerNum() const {
  return resource_.persistence_worker_num();
}
int32_t JobDesc::ParallelPieceSize() const {
  return job_conf_.data_part_num() * SinglePieceSize();
}
int64_t JobDesc::piece_num_of_experiment_phase() const {
  return job_conf_.piece_num_of_experiment_phase();
}
float JobDesc::available_zone_mem_ratio() const {
  float ratio = job_conf_.available_zone_mem_ratio();
  CHECK_GT(ratio, 0.f);
  CHECK_LE(ratio, 1.f);
  return ratio;
}
uint64_t JobDesc::one_data_part_buffer_byte_size() const {
  return job_conf_.one_data_part_buffer_mbyte_size() * 1024 * 1024;
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
  return GetMsgPtrFromPbMessage<InitializerConf>(job_conf_.train_conf(),
                                                 "default_initializer_conf");
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
    piece_experiment = std::min<int64_t>(
        piece_experiment,
        train_conf->total_batch_num() * train_conf->num_of_pieces_in_batch());
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
  std::vector<OperatorConf> gen_op_confs;
  for (OperatorConf& op_conf : *(dlnet_conf_.mutable_op())) {
    if (op_conf.has_decode_ofrecord_conf() == false) { continue; }
    if (op_conf.decode_ofrecord_conf().blob_size() == 1) { continue; }
    const DecodeOFRecordOpConf& decode_conf = op_conf.decode_ofrecord_conf();
    PbRpf<BlobConf>* blobs =
        op_conf.mutable_decode_ofrecord_conf()->mutable_blob();
    Erase<PbRpf<BlobConf>>(
        *blobs,
        [&](const BlobConf& blob_conf) -> bool {
          return blob_conf.max_sequence_size() > 1;
        },
        [&](const BlobConf& blob_conf) {
          gen_op_confs.emplace_back(op_conf);
          DecodeOFRecordOpConf* gen_decode_conf =
              gen_op_confs.back().mutable_decode_ofrecord_conf();
          *gen_decode_conf = decode_conf;
          gen_decode_conf->clear_blob();
          *gen_decode_conf->add_blob() = blob_conf;
        });
  }
  for (OperatorConf& gen_op_conf : gen_op_confs) {
    *dlnet_conf_.add_op() = gen_op_conf;
  }
}

}  // namespace oneflow
