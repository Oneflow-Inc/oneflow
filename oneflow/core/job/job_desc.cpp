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

int64_t JobDesc::all_reduce_group_min_byte() const {
  int64_t ret = job_conf_.all_reduce_group_min_mbyte() * 1024 * 1024;
  CHECK_GT(ret, 0);
  return ret;
}

float JobDesc::all_reduce_group_size_warmup() const {
  float ret = job_conf_.all_reduce_group_size_warmup();
  CHECK_GT(ret, 1);
  return ret;
}

int64_t JobDesc::all_reduce_group_num() const {
  int64_t ret = job_conf_.all_reduce_group_num();
  CHECK_GT(ret, 0);
  return ret;
}

float JobDesc::all_reduce_lazy_ratio() const {
  float ratio = job_conf_.all_reduce_lazy_ratio();
  CHECK_GE(ratio, 0.0);
  CHECK_LE(ratio, 1.0);
  return ratio;
}

bool JobDesc::all_reduce_fp16() const { return job_conf_.all_reduce_fp16(); }

int64_t JobDesc::piece_num_of_experiment_phase() const {
  return job_conf_.exp_run_conf().piece_num_of_experiment_phase();
}

bool JobDesc::enable_experiment_run() const {
  return job_conf_.exp_run_conf().enable_experiment_run();
}

int32_t JobDesc::NumOfBatchesInSnapshot() const {
  return job_conf_.train_conf().num_of_batches_in_snapshot();
}
int64_t JobDesc::TotalBatchNum() const { return job_conf_.total_batch_num(); }
int64_t JobDesc::BatchSize() const { return job_conf_.train_conf().batch_size(); }
int64_t JobDesc::NumOfPiecesInBatch() const {
  if (IsPredict()) { return 1; }
  CHECK_EQ(BatchSize() % RecordPieceSize(), 0);
  return BatchSize() / RecordPieceSize();
}
float JobDesc::weight_l1() const { return job_conf_.train_conf().weight_l1(); }
float JobDesc::bias_l1() const { return job_conf_.train_conf().bias_l1(); }
float JobDesc::weight_l2() const { return job_conf_.train_conf().weight_l2(); }
float JobDesc::bias_l2() const { return job_conf_.train_conf().bias_l2(); }
int32_t JobDesc::loss_scale_factor() const {
  int32_t loss_scale_factor = job_conf_.train_conf().loss_scale_factor();
  CHECK_GE(loss_scale_factor, 1);
  return loss_scale_factor;
}

int32_t JobDesc::DataPartNum() const { return job_conf_.data_part_num(); }

JobDesc::JobDesc(const JobConfigProto& job_conf, int64_t job_id)
    : job_conf_(job_conf), job_id_(job_id) {
  Init();
}

void JobDesc::Init() {
#ifndef WITH_RDMA
  CHECK_EQ(Global<ResourceDesc>::Get()->use_rdma(), false) << "Please compile ONEFLOW with RDMA";
#endif
#ifndef WITH_CUDA
  CHECK_EQ(job_conf_.enable_nccl(), false) << "Please compile ONEFLOW with NCCL";
#endif  // WITH_CUDA
  int64_t piece_exp = job_conf_.exp_run_conf().piece_num_of_experiment_phase();
  if (job_conf_.has_train_conf()) {
    if (piece_exp == -1) { piece_exp = 19 * NumOfPiecesInBatch(); }
    piece_exp = std::max(piece_exp, NumOfPiecesInBatch());
    piece_exp = std::min(piece_exp, job_conf_.total_batch_num() * NumOfPiecesInBatch());
  } else {
    if (piece_exp == -1) { piece_exp = 19; }
  }
  LOG(INFO) << "Set piece_num_of_experiment_phase " << piece_exp;
  job_conf_.mutable_exp_run_conf()->set_piece_num_of_experiment_phase(piece_exp);
#ifndef WITH_CUDA
  CHECK_EQ(Global<ResourceDesc>::Get()->GpuDeviceNum(), 0);
#endif
}

bool IsInterfaceOpConf(const OperatorConf& op_conf) {
  return IsClassRegistered<IsInterfaceOpConf4OpTypeCase>(op_conf.op_type_case());
}

GlobalJobDescScope::GlobalJobDescScope(const JobConfigProto& job_conf, int64_t job_id) {
  Global<JobDesc>::New(job_conf, job_id);
}

GlobalJobDescScope::~GlobalJobDescScope() { Global<JobDesc>::Delete(); }

const JobDesc& GlobalJobDesc() { return *Global<JobDesc>::Get(); }

bool IsPullJob(const std::string& job_name, const InterUserJobInfo& inter_user_job_info) {
  for (const auto& pair : inter_user_job_info.output_or_var_op_name2pull_job_name()) {
    if (pair.second == job_name) { return true; }
  }
  if (job_name == inter_user_job_info.global_model_save_job_name()) { return true; }
  return false;
}

bool IsPushJob(const std::string& job_name, const InterUserJobInfo& inter_user_job_info) {
  for (const auto& pair : inter_user_job_info.input_or_var_op_name2push_job_name()) {
    if (pair.second == job_name) { return true; }
  }
  if (job_name == inter_user_job_info.global_model_init_job_name()) { return true; }
  return false;
}

}  // namespace oneflow
