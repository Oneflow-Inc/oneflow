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

const TrainConf& GetTrainConf(const Job& job) {
  CHECK(job.job_conf().has_train_conf());
  return job.job_conf().train_conf();
}

class JobId final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobId);
  ~JobId() = default;

  operator int64_t() const { return value_; }

 private:
  friend class Global<JobId>;
  JobId(int64_t value) : value_(value) {}

  int64_t value_;
};

bool FindForeignInputOp(const Job& job) {
  bool foreigin_input_op_found = false;
  for (const auto& op_conf : job.net().op()) {
    if (op_conf.has_foreign_input_conf()) {
      CHECK_EQ(foreigin_input_op_found, false);
      foreigin_input_op_found = true;
    }
  }
  return foreigin_input_op_found;
}

bool FindForeignOutputOp(const Job& job) {
  bool foreigin_output_op_found = false;
  for (const auto& op_conf : job.net().op()) {
    if (op_conf.has_foreign_output_conf()) {
      CHECK_EQ(foreigin_output_op_found, false);
      foreigin_output_op_found = true;
    }
  }
  return foreigin_output_op_found;
}

}  // namespace

int64_t JobDesc::all_reduce_group_min_byte() const {
  int64_t ret = job_.job_conf().all_reduce_group_min_mbyte() * 1024 * 1024;
  CHECK_GT(ret, 0);
  return ret;
}

float JobDesc::all_reduce_group_size_warmup() const {
  float ret = job_.job_conf().all_reduce_group_size_warmup();
  CHECK_GT(ret, 1);
  return ret;
}

int64_t JobDesc::all_reduce_group_num() const {
  int64_t ret = job_.job_conf().all_reduce_group_num();
  CHECK_GT(ret, 0);
  return ret;
}

float JobDesc::all_reduce_lazy_ratio() const {
  float ratio = job_.job_conf().all_reduce_lazy_ratio();
  CHECK_GE(ratio, 0.0);
  CHECK_LE(ratio, 1.0);
  return ratio;
}

bool JobDesc::all_reduce_fp16() const { return job_.job_conf().all_reduce_fp16(); }

int64_t JobDesc::piece_num_of_experiment_phase() const {
  return job_.job_conf().exp_run_conf().piece_num_of_experiment_phase();
}

bool JobDesc::enable_experiment_run() const {
  return job_.job_conf().exp_run_conf().enable_experiment_run();
}

int32_t JobDesc::NumOfBatchesInSnapshot() const {
  return GetTrainConf(job_).num_of_batches_in_snapshot();
}
int64_t JobDesc::TotalBatchNum() const { return job_.job_conf().total_batch_num(); }
int32_t JobDesc::PieceNumOfPrintLoss() const {
  return job_.job_conf().train_conf().piece_num_of_print_loss();
}
int32_t JobDesc::PieceNumOfPrintAccuracy() const {
  return job_.job_conf().train_conf().piece_num_of_print_accuracy();
}
int64_t JobDesc::BatchSize() const { return GetTrainConf(job_).batch_size(); }
int64_t JobDesc::NumOfPiecesInBatch() const {
  if (IsPredict()) { return 1; }
  CHECK_EQ(BatchSize() % RecordPieceSize(), 0);
  return BatchSize() / RecordPieceSize();
}
float JobDesc::primary_lr() const { return GetTrainConf(job_).primary_lr(); }
float JobDesc::secondary_lr() const { return GetTrainConf(job_).secondary_lr(); }
float JobDesc::weight_l1() const { return GetTrainConf(job_).weight_l1(); }
float JobDesc::bias_l1() const { return GetTrainConf(job_).bias_l1(); }
float JobDesc::weight_l2() const { return GetTrainConf(job_).weight_l2(); }
float JobDesc::bias_l2() const { return GetTrainConf(job_).bias_l2(); }
int32_t JobDesc::loss_scale_factor() const {
  int32_t loss_scale_factor = GetTrainConf(job_).loss_scale_factor();
  CHECK_GE(loss_scale_factor, 1);
  return loss_scale_factor;
}

int32_t JobDesc::DataPartNum() const { return job_.job_conf().data_part_num(); }

JobDesc::JobDesc(const Job& job, int32_t job_id) : job_(job), job_id_(job_id) { Init(); }

void JobDesc::Init() {
#ifndef WITH_RDMA
  CHECK_EQ(Global<ResourceDesc>::Get()->use_rdma(), false) << "Please compile ONEFLOW with RDMA";
#endif
#ifndef WITH_CUDA
  CHECK_EQ(job_.job_conf().enable_nccl(), false) << "Please compile ONEFLOW with NCCL";
  CHECK_EQ(job_.job_conf().enable_cuda_ring_all_reduce(), false)
      << "Please compile ONEFLOW with CUDA";
#endif  // WITH_CUDA
  int64_t piece_exp = job_.job_conf().exp_run_conf().piece_num_of_experiment_phase();
  if (job_.job_conf().has_train_conf()) {
    TrainConf* train_conf = job_.mutable_job_conf()->mutable_train_conf();
    if (train_conf->piece_num_of_print_loss() == -1) {
      train_conf->set_piece_num_of_print_loss(NumOfPiecesInBatch());
    }
    if (train_conf->piece_num_of_print_accuracy() == -1) {
      train_conf->set_piece_num_of_print_accuracy(NumOfPiecesInBatch());
    }
    if (piece_exp == -1) { piece_exp = 19 * NumOfPiecesInBatch(); }
    piece_exp = std::max(piece_exp, NumOfPiecesInBatch());
    piece_exp = std::max(piece_exp, train_conf->piece_num_of_print_loss());
    piece_exp = std::min(piece_exp, job_.job_conf().total_batch_num() * NumOfPiecesInBatch());
  } else {
    if (piece_exp == -1) { piece_exp = 19; }
  }
  LOG(INFO) << "Set piece_num_of_experiment_phase " << piece_exp;
  job_.mutable_job_conf()->mutable_exp_run_conf()->set_piece_num_of_experiment_phase(piece_exp);
#ifndef WITH_CUDA
  CHECK_EQ(Global<ResourceDesc>::Get()->GpuDeviceNum(), 0);
#endif
  is_push_job_ = FindForeignInputOp(job_);
  is_pull_job_ = FindForeignOutputOp(job_);
}

bool IsInterfaceOpConf(const OperatorConf& op_conf) {
  return IsClassRegistered<IsInterfaceOpConf4OpTypeCase>(op_conf.op_type_case());
}

void WithJobIdGlobal(int64_t job_id, const std::function<void()>& Handler) {
  Global<JobId>::New(job_id);
  Handler();
  Global<JobId>::Delete();
}

const JobDesc& GlobalJobDesc(int64_t job_id) {
  return *Global<std::vector<std::unique_ptr<JobDesc>>>::Get()->at(job_id);
}

const JobDesc& GlobalJobDesc() { return GlobalJobDesc(*Global<JobId>::Get()); }

}  // namespace oneflow
