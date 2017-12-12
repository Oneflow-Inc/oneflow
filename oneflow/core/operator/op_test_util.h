#ifndef ONEFLOW_CORE_OPERATOR_OP_TEST_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_OP_TEST_UTIL_H_

#include "oneflow/core/operator/clone_op.h"

namespace oneflow {

std::function<BlobDesc*(const std::string)> ConstructBn2BlobDescFunc(
    Operator* op) {
  HashMap<std::string, BlobDesc*> bn2blobdesc_map =
      new HashMap<std::string, BlobDesc*>();
  for (const std::string& bn : op->data_tmp_bns()) {
    bn2blobdesc_map[bn] = new BlobDesc;
  }
  for (const std::string& bn : op->input_bns()) {
    bn2blobdesc_map[bn] = new BlobDesc;
  }
  for (const std::string& bn : op->input_diff_bns()) {
    bn2blobdesc_map[bn] = new BlobDesc;
  }
  for (const std::string& bn : op->output_bns()) {
    bn2blobdesc_map[bn] = new BlobDesc;
  }
  for (const std::string& bn : op->output_diff_bns()) {
    bn2blobdesc_map[bn] = new BlobDesc;
  }
  for (const std::string& bn : op->model_bns()) {
    bn2blobdesc_map[bn] = new BlobDesc;
  }
  for (const std::string& bn : op->model_diff_bns()) {
    bn2blobdesc_map[bn] = new BlobDesc;
  }
  for (const std::string& bn : op->model_tmp_bns()) {
    bn2blobdesc_map[bn] = new BlobDesc;
  }
  return [bn2blobdesc_map](const std::string& bn) {
    return bn2blobdesc_map->at(bn);
  };
}

}  // namespace oneflow

#endif  //  ONEFLOW_CORE_OPERATOR_OP_TEST_UTIL_H_
