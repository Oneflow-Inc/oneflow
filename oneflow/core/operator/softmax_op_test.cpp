#include "oneflow/core/job/mock_job_desc.h"
#include "oneflow/core/operator/softmax_op.h"
#include "oneflow/core/operator/op_test_util.h"

namespace oneflow {

template<typename T, bool has_data_id>
void TestSoftmaxOp() {
  // mock JobDesc
  test::MockJobDesc mock_job_desc;
  test::InitJobDescSingleton(&mock_job_desc);
  EXPECT_CALL(mock_job_desc, DefaultDataType())
      .WillRepeatedly(testing::Return(GetDataType<T>::val));

  std::vector<std::vector<int64_t>> in_shapes = {{3, 5}};
  std::vector<std::string> ibns = {"in"};
  std::vector<std::string> obns = {"out"};
  std::vector<std::string> other_bns = {"tmp"};

  // Construct bn2BlobDesc function
  HashMap<std::string, BlobDesc*> bn2blobdesc_map;
  auto bn2blobdesc_func =
      ConstructBn2BlobDescFunc(bn2blobdesc_map, ibns, obns, other_bns,
                               in_shapes, GetDataType<T>::val, has_data_id);

  // create softmax_op
  OperatorConf op_conf;
  op_conf.set_name("softmax_test");
  op_conf.mutable_softmax_conf()->set_in("softmax/in");
  op_conf.mutable_softmax_conf()->set_out("softmax/out");
  auto softmax_op = ConstructOp(op_conf);

  // infershape
  softmax_op->InferBlobDescs(bn2blobdesc_func, nullptr);

  // test
  BlobDesc* in_blobdesc = bn2blobdesc_func("in");
  BlobDesc* out_blobdesc = bn2blobdesc_func("out");
  BlobDesc* tmp_blobdesc = bn2blobdesc_func("tmp");

  ASSERT_TRUE(in_blobdesc->shape() == out_blobdesc->shape());
  ASSERT_TRUE(tmp_blobdesc->shape() == Shape({3}));
  ASSERT_TRUE(in_blobdesc->data_type() == out_blobdesc->data_type());
  ASSERT_TRUE(in_blobdesc->data_type() == tmp_blobdesc->data_type());
}

TEST(SoftmaxOp, infer_blob_desc) {
#define MAKE_ENTRY(x, y) TestSoftmaxOp<OF_PP_PAIR_FIRST(x), y>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, BOOL_SEQ)
}

}  // namespace oneflow
