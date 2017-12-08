#include "oneflow/core/common/test_util.h"
#include "oneflow/core/operator/clone_op.h"

namespace oneflow {

std::shared_ptr<Operator> CreateCloneOp(int out_num) {
  OperatorConf op_conf;
  op_conf.set_name("clone_test");
  op_conf.mutable_clone_conf()->set_out_num(out_num);
  op_conf.mutable_clone_conf()->set_lbn("clone_lbn");
  return ConstructOp(op_conf);
}

void GenBn2BlobDescMap(HashMap<std::string, BlobDesc*>& bn2blobdesc_map,
                       const std::vector<std::string>& ibns,
                       const std::vector<std::string>& obns,
                       const std::vector<std::vector<int64_t>>& in_shapes,
                       const DataType& data_type, bool has_data_id) {
  CHECK_EQ(ibns.size(), in_shapes.size());
  FOR_RANGE(size_t, i, 0, ibns.size()) {
    bn2blobdesc_map[ibns.at(i)] =
        new BlobDesc(Shape(in_shapes.at(i)), data_type, has_data_id);
  }
  for (const std::string& obn : obns) { bn2blobdesc_map[obn] = new BlobDesc; }
}

}  // namespace oneflow
