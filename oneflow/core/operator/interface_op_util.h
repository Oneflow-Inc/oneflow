#ifndef ONEFLOW_CORE_OPERATOR_INTERFACE_OP_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_INTERFACE_OP_UTIL_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/job/job.pb.h"

namespace oneflow {

struct InterfaceOpUtil final {
  static void InferOutBlobDesc(const InterfaceBlobConf& blob_conf, BlobDesc* out_blob_desc,
                               const ParallelContext* parallel_ctx);
  static void InferOutBlobDesc(const InterfaceBlobConf& blob_conf, BlobDesc* out_blob_desc,
                               const ParallelContext* parallel_ctx, int64_t record_piece_size);
  static void InferHasBatchDim(const InterfaceBlobConf& blob_conf, bool* has_batch_dim);
  static void GetInputLikeOpSbpSignature(const InterfaceBlobConf& blob_conf,
                                         const PbRpf<std::string>& input_bns,
                                         const PbRpf<std::string>& output_bns,
                                         SbpSignature* sbp_signature);
  static void GetOutputLikeOpSbpSignature(const InterfaceBlobConf& blob_conf,
                                          const PbRpf<std::string>& input_bns,
                                          const PbRpf<std::string>& output_bns,
                                          SbpSignature* sbp_signature);
  static void InitBlobConf(InterfaceBlobConf* blob_conf,
                           const ParallelBlobConf& parallel_blob_conf);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_INTERFACE_OP_UTIL_H_
