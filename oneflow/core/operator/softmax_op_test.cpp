#include "oneflow/core/operator/softmax_op.h"

namespace oneflow {

template<typename T, bool has_data_id>
void TestSoftmaxOp() {
  // create softmax_op
  OperatorConf op_conf;
  DataType data_type = GetDataType<T>::val;
  op_conf.set_name("softmax_test");
  op_conf.mutable_softmax_conf()->set_in("softmax/in");
  op_conf.mutable_softmax_conf()->set_out("softmax/out");
  auto softmax_op = ConstructOp(op_conf);
  HashMap<std::string, BlobDesc*> bn2blobdesc_map{
      {softmax_op->SoleIbn(),
       new BlobDesc(Shape({3, 5}), data_type, has_data_id)},
      {softmax_op->SoleObn(), new BlobDesc},
      {softmax_op->SoleDtbn(), new BlobDesc}};
  auto fp = [&bn2blobdesc_map](const std::string& bn) {
    return bn2blobdesc_map.at(bn);
  };
  // infershape
  softmax_op->InferBlobDesc4FwBlobs(fp, kDataParallel, 0, 1);
  // test
  BlobDesc* in_blobdesc = fp(softmax_op->SoleIbn());
  BlobDesc* out_blobdesc = fp(softmax_op->SoleObn());
  BlobDesc* tmp_blobdesc = fp(softmax_op->SoleDtbn());

  ASSERT_TRUE(in_blobdesc->shape() == in_blobdesc->shape());
  ASSERT_TRUE(tmp_blobdesc->shape() == Shape({3}));
  ASSERT_TRUE(in_blobdesc->data_type() == out_blobdesc->data_type());
  ASSERT_TRUE(in_blobdesc->data_type() == tmp_blobdesc->data_type());
}

TEST(SoftmaxOp, infer_blob_desc) {
#define MAKE_ENTRY(x, y) TestSoftmaxOp<OF_PP_PAIR_FIRST(x), y>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, BOOL_SEQ)
}

}  // namespace oneflow
