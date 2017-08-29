#include "oneflow/core/operator/clone_op.h"

namespace oneflow {

template<typename T, bool has_data_id>
void TestCloneOp() {
  OperatorConf op_conf;
  op_conf.set_name("clone_test");
  op_conf.mutable_clone_conf()->set_out_num(3);
  op_conf.mutable_clone_conf()->set_lbn("clone_lbn");
  op_conf.mutable_clone_conf()->set_data_type(GetDataType<T>::val);
  auto clone_op = ConstructOp(op_conf);
  HashMap<std::string, BlobDesc*> bn2blobdesc_map;
  bn2blobdesc_map[clone_op->SoleIbn()] =
      new BlobDesc{{4, 3}, GetDataType<T>::val, has_data_id};
  for (const std::string& obn : clone_op->output_bns()) {
    bn2blobdesc_map[obn] = new BlobDesc;
  }
  auto bn2blobdesc_func = [&](const std::string& bn) {
    return bn2blobdesc_map.at(bn);
  };
  clone_op->InferBlobDesc4FwBlobs(bn2blobdesc_func, kDataParallel, 3, 10);
  const BlobDesc* in_blob_desc = bn2blobdesc_map.at(clone_op->SoleIbn());
  for (const std::string& obn : clone_op->output_bns()) {
    const BlobDesc* out_blob_desc = bn2blobdesc_map.at(obn);
    ASSERT_TRUE(*in_blob_desc == *out_blob_desc);
  }
}

TEST(CloneOp, infer_blob_desc) {
#define SEQ (true)(false)
#define MAKE_ENTRY(x, y) TestCloneOp<x, y>();
  SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, ALL_DATA_TYPE_PAIR(), SEQ)
}

}  // namespace oneflow
