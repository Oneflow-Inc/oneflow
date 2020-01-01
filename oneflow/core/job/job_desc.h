#ifndef ONEFLOW_CORE_JOB_JOB_DESC_H_
#define ONEFLOW_CORE_JOB_JOB_DESC_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/dlnet_conf.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/inter_user_job_info.pb.h"
#include "oneflow/core/register/logical_blob_id.pb.h"
#include "oneflow/core/framework/config_def.h"

namespace oneflow {

bool IsInterfaceOpConf(const OperatorConf& op_conf);

class JobDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobDesc);
  JobDesc(const JobConfigProto& job_conf, int64_t job_id);
  ~JobDesc() = default;
  // Common
  int64_t job_id() const { return job_id_; }
  const std::string& job_name() const { return job_conf_.job_name(); }
  int64_t concurrency_width() const { return job_conf_.concurrency_width(); }
  const JobConfigProto& job_conf() const { return job_conf_; }
  DataType DefaultDataType() const { return job_conf_.default_data_type(); }
  size_t SizeOfOneDataId() const { return job_conf_.max_data_id_length() * sizeof(char); }
  bool EnableCudnn() const { return job_conf_.enable_cudnn(); }
  bool IsTrain() const { return job_conf_.has_train_conf(); }
  bool IsPredict() const { return job_conf_.has_predict_conf(); }
  int64_t piece_num_of_experiment_phase() const;
  bool use_memory_allocation_algorithm_v2() const {
    return job_conf_.use_memory_allocation_algorithm_v2();
  }
  bool enable_experiment_run() const;
  bool enable_reuse_mem() const { return job_conf_.enable_reuse_mem(); }
  bool enable_inplace() const { return job_conf_.enable_inplace(); }
  bool enable_inplace_in_reduce_struct() const {
    return job_conf_.enable_inplace_in_reduce_struct();
  }
  bool cudnn_conv_enable_true_half() const { return job_conf_.cudnn_conv_enable_true_half(); }
  bool enable_float_compute_for_half_gemm() const {
    return job_conf_.enable_float_compute_for_half_gemm();
  }
  bool enable_auto_mixed_precision() const { return job_conf_.enable_auto_mixed_precision(); }
  bool enable_nccl() const { return job_conf_.enable_nccl(); }
  bool use_nccl_inter_node_communication() const {
    return job_conf_.use_nccl_inter_node_communication();
  }
  bool use_boxing_v2() const { return job_conf_.use_boxing_v2(); }
  bool enable_all_reduce_group() const { return job_conf_.enable_all_reduce_group(); }
  bool enable_non_distributed_optimizer() const {
    return job_conf_.enable_non_distributed_optimizer();
  }
  int64_t non_distributed_optimizer_group_size_mbyte() const {
    return job_conf_.non_distributed_optimizer_group_size_mbyte();
  }
  bool disable_all_reduce_sequence() const { return job_conf_.disable_all_reduce_sequence(); }
  int64_t all_reduce_group_num() const;
  int64_t all_reduce_group_min_byte() const;
  float all_reduce_group_size_warmup() const;
  float all_reduce_lazy_ratio() const;
  bool all_reduce_fp16() const;
  int64_t cudnn_buf_limit_mbyte() const { return job_conf_.cudnn_buf_limit_mbyte(); }

  bool has_xrt_config() const { return job_conf_.has_xrt_config(); }
  const XrtConfig& xrt_config() const { return job_conf_.xrt_config(); }

#define DEFINE_FUNCTION_CONFIG_GETTER(T, func_name, field_name)     \
  T func_name(const std::string& field_name) const {                \
    const UserOpAttrVal& attr_val = GetFunctionFlagVal(field_name); \
    CHECK(attr_val.has_##field_name());                             \
    return attr_val.field_name();                                   \
  }
  DEFINE_FUNCTION_CONFIG_GETTER(bool, Bool, at_bool);
  DEFINE_FUNCTION_CONFIG_GETTER(int64_t, Int64, at_int64);
  DEFINE_FUNCTION_CONFIG_GETTER(double, Double, at_double);
  DEFINE_FUNCTION_CONFIG_GETTER(const std::string&, String, at_string);

  // Train conf
  int64_t TotalBatchNum() const;
  int64_t NumOfPiecesInBatch() const;
  float weight_l1() const;
  float bias_l1() const;
  float weight_l2() const;
  float bias_l2() const;
  int32_t loss_scale_factor() const;

 private:
  void Init();
  const UserOpAttrVal& GetFunctionFlagVal(const std::string& field_name) const;

  JobConfigProto job_conf_;
  int64_t job_id_;
};

typedef HashMap<std::string, int64_t> JobName2JobId;

class GlobalJobDescScope final {
 public:
  GlobalJobDescScope(const JobConfigProto& job_conf, int64_t job_id);
  ~GlobalJobDescScope();
};
const JobDesc& GlobalJobDesc();

bool IsPullJob(const std::string& job_name, const InterUserJobInfo& inter_user_job_info);
bool IsPushJob(const std::string& job_name, const InterUserJobInfo& inter_user_job_info);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_DESC_H_
