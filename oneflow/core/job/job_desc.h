#ifndef ONEFLOW_CORE_JOB_JOB_DESC_H_
#define ONEFLOW_CORE_JOB_JOB_DESC_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/dlnet_conf.pb.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/register/logical_blob_id.pb.h"

namespace oneflow {

bool IsInterfaceOpConf(const OperatorConf& op_conf);

class JobDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobDesc);
  JobDesc() = default;
  JobDesc(const JobConf& job_conf, int32_t job_id);
  ~JobDesc() = default;

  // Common
  int32_t job_id() const { return job_id_; }
  const std::string& job_name() const { return job_conf_.name(); }
  const PbRpf<std::string>& arg_op_name() const { return job_conf_.arg_op_name(); }
  int64_t concurrency_width() const { return job_conf_.other().concurrency_width(); }
  const Config& other_conf() const { return job_conf_.other(); }
  DataType DefaultDataType() const { return job_conf_.other().default_data_type(); }
  size_t SizeOfOneDataId() const { return job_conf_.other().max_data_id_length() * sizeof(char); }
  bool EnableCudnn() const { return job_conf_.other().enable_cudnn(); }
  bool IsTrain() const { return job_conf_.other().has_train_conf(); }
  bool IsPredict() const { return job_conf_.other().has_predict_conf(); }
  int64_t RecordPieceSize() const { return job_conf_.other().piece_size(); }
  int64_t piece_num_of_experiment_phase() const;
  bool enable_experiment_run() const;
  bool enable_mem_sharing() const { return job_conf_.other().enable_mem_sharing(); }
  bool enable_inplace() const { return job_conf_.other().enable_inplace(); }
  bool enable_blob_mem_sharing() const { return job_conf_.other().enable_blob_mem_sharing(); }
  bool enable_nccl() const { return job_conf_.other().enable_nccl(); }
  bool use_nccl_inter_node_communication() const {
    return job_conf_.other().use_nccl_inter_node_communication();
  }
  int64_t all_reduce_group_num() const;
  int64_t all_reduce_group_min_byte() const;
  float all_reduce_group_size_warmup() const;
  float all_reduce_lazy_ratio() const;
  bool all_reduce_fp16() const;
  int64_t cudnn_buf_limit_mbyte() const { return job_conf_.other().cudnn_buf_limit_mbyte(); }

  // Train conf
  int32_t NumOfBatchesInSnapshot() const;
  int64_t TotalBatchNum() const;
  const InitializerConf* DefaultInitializerConf() const;
  int32_t PieceNumOfPrintLoss() const;
  int32_t PieceNumOfPrintAccuracy() const;
  int64_t BatchSize() const;
  int64_t NumOfPiecesInBatch() const;
  float primary_lr() const;
  float secondary_lr() const;
  float weight_l1() const;
  float bias_l1() const;
  float weight_l2() const;
  float bias_l2() const;
  int32_t DataPartNum() const;

 private:
  void Init();

  JobConf job_conf_;
  int32_t job_id_;
};

typedef HashMap<std::string, int64_t> JobName2JobId;

void WithGlobalJobId(int64_t job_id, const std::function<void()>& Handler);
const JobDesc& GlobalJobDesc();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_DESC_H_
