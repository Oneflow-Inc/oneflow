#ifndef ONEFLOW_CORE_OPERATOR_OP_TEST_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_OP_TEST_UTIL_H_

#include "oneflow/core/operator/clone_op.h"

namespace oneflow {

std::function<BlobDesc*(const std::string)> ConstructBn2BlobDescFunc(
    HashMap<std::string, BlobDesc*>& bn2blobdesc_map,
    const std::vector<std::string>& ibns, const std::vector<std::string>& obns,
    const std::vector<std::string>& other_bns,
    const std::vector<std::vector<int64_t>>& in_shapes,
    const DataType& data_type, bool has_data_id) {
  CHECK_EQ(ibns.size(), in_shapes.size());
  FOR_RANGE(size_t, i, 0, ibns.size()) {
    bn2blobdesc_map[ibns.at(i)] =
        new BlobDesc(Shape(in_shapes.at(i)), data_type, has_data_id);
  }
  for (const std::string& obn : obns) { bn2blobdesc_map[obn] = new BlobDesc; }
  for (const std::string& bn : other_bns) {
    bn2blobdesc_map[bn] = new BlobDesc;
  }
  return [&bn2blobdesc_map](const std::string bn) {
    return bn2blobdesc_map.at(bn);
  };
}

}  // namespace oneflow

#endif  //  ONEFLOW_CORE_OPERATOR_OP_TEST_UTIL_H_
