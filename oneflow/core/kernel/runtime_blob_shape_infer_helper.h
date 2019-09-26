#ifndef ONEFLOW_CORE_KERNEL_RUNTIME_BLOB_SHAPE_INFER_HELPER_H_
#define ONEFLOW_CORE_KERNEL_RUNTIME_BLOB_SHAPE_INFER_HELPER_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class Blob;

class RuntimeBlobShapeInferHelper final {
 public:
  RuntimeBlobShapeInferHelper(const OperatorConf& op_conf, const JobDesc* job_desc);
  ~RuntimeBlobShapeInferHelper() = default;

  void InferDenseShape(std::function<Blob*(const std::string&)> BnInOp2Blob);

 private:
  std::shared_ptr<Operator> op_;
  HashSet<std::string> ibns_;
  HashMap<std::string, std::unique_ptr<BlobDesc>> bn_in_op2blob_desc_;
  ParallelContext parallel_ctx_;
  SbpSignature sbp_signature_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RUNTIME_BLOB_SHAPE_INFER_HELPER_H_
