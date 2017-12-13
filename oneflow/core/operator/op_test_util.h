#ifndef ONEFLOW_CORE_OPERATOR_OP_TEST_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_OP_TEST_UTIL_H_

#include "oneflow/core/operator/clone_op.h"

namespace oneflow {

std::function<BlobDesc*(const std::string)> ConstructBn2BlobDescFunc(
    std::shared_ptr<Operator> op) {
  auto InsertBnsWithEmptyBlobDesc2Map =
      [](const std::vector<std::string>& bns,
         HashMap<std::string, BlobDesc*>* bn2blobdesc_map) {
        for (const std::string& bn : bns) {
          CHECK(bn2blobdesc_map->insert({bn, new BlobDesc}).second);
        }
      };
  HashMap<std::string, BlobDesc*>* bn2blobdesc_map =
      new HashMap<std::string, BlobDesc*>();
  InsertBnsWithEmptyBlobDesc2Map(op->data_tmp_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->input_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->input_diff_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->output_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->output_diff_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->model_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->model_diff_bns(), bn2blobdesc_map);
  InsertBnsWithEmptyBlobDesc2Map(op->model_tmp_bns(), bn2blobdesc_map);
  return [bn2blobdesc_map](const std::string& bn) {
    return bn2blobdesc_map->at(bn);
  };
}

}  // namespace oneflow

#endif  //  ONEFLOW_CORE_OPERATOR_OP_TEST_UTIL_H_
