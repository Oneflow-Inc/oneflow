#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/hadoop/hadoop_file_system.h"
#include "oneflow/core/graph/graph.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job/job_desc.h"

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

int64_t JobDesc::piece_num_of_experiment_phase() const {
  return job_conf_.other().exp_run_conf().piece_num_of_experiment_phase();
}

bool JobDesc::enable_experiment_run() const {
  return job_conf_.other().exp_run_conf().enable_experiment_run();
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
  if (IsPredict() && !other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) { return 1; }
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

JobDesc::JobDesc(const JobConf& job_conf, int32_t job_id) : job_conf_(job_conf), job_id_(job_id) {
  Init();
}

void JobDesc::Init() {
#ifndef WITH_RDMA
  CHECK_EQ(Global<ResourceDesc>::Get()->use_rdma(), false) << "Please compile ONEFLOW with RDMA";
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
  CHECK_EQ(Global<ResourceDesc>::Get()->GpuDeviceNum(), 0);
#endif
}

bool IsInterfaceOpConf(const OperatorConf& op_conf) {
  return op_conf.has_variable_conf() || op_conf.has_input_conf() || op_conf.has_output_conf();
}

const JobDesc& GlobalJobDesc() {
  return *Global<std::vector<std::unique_ptr<JobDesc>>>::Get()->at(Global<JobId>::Get()->value());
}

}  // namespace oneflow
