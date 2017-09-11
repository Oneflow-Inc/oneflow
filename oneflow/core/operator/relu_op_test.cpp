#include "oneflow/core/operator/relu_op.h"

namespace oneflow {

template<typename T, bool has_data_id>
void TestReluOp() {
  // create relu_op with input shape 3x5x4
  OperatorConf op_conf;
  DataType data_type = GetDataType<T>::val;
  op_conf.set_name("relu_test");
  op_conf.mutable_relu_conf()->set_in("relu/in");
  op_conf.mutable_relu_conf()->set_out("relu/out");
  auto relu_op = ConstructOp(op_conf);
  HashMap<std::string, BlobDesc*> bn2blobdesc_map{
      {relu_op->SoleIbn(),
       new BlobDesc(Shape({3, 5, 4}), data_type, has_data_id)},
      {relu_op->SoleObn(), new BlobDesc}};
  auto fp = [&bn2blobdesc_map](const std::string& bn) {
    return bn2blobdesc_map.at(bn);
  };
  // do infer shape
  relu_op->InferBlobDesc4FwBlobs(fp, kDataParallel, 0, 1);
  // test
  ASSERT_TRUE(*(bn2blobdesc_map.at(relu_op->SoleIbn()))
              == *(bn2blobdesc_map.at(relu_op->SoleObn())));
}

TEST(ReluOp, infer_blob_desc) {
#define MAKE_ENTRY(x, y) TestReluOp<OF_PP_PAIR_FIRST(x), y>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, ARITHMETIC_DATA_TYPE_SEQ,
                                   BOOL_SEQ)
}

}  // namespace oneflow
