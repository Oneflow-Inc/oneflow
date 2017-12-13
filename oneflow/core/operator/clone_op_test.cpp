#include "oneflow/core/job/mock_job_desc.h"
#include "oneflow/core/operator/clone_op.h"
#include "oneflow/core/operator/op_test_util.h"

namespace oneflow {

std::shared_ptr<Operator> CreateCloneOp(int out_num) {
  OperatorConf op_conf;
  op_conf.set_name("clone_test");
  op_conf.mutable_clone_conf()->set_out_num(out_num);
  op_conf.mutable_clone_conf()->set_lbn("clone_lbn");
  return ConstructOp(op_conf);
}

template<typename T, bool has_data_id>
void DoCloneOpTest(int out_num, const std::vector<int64_t>& in_shape) {
  auto clone_op = CreateCloneOp(out_num);

  auto bn2blobdesc_func = ConstructBn2BlobDescFunc(clone_op);
  BlobDesc* in_blob_desc = bn2blobdesc_func("in");
  in_blob_desc->mut_shape().dim_vec_ = in_shape;
  in_blob_desc->set_data_type(GetDataType<T>::val);
  in_blob_desc->set_has_data_id(has_data_id);

  clone_op->InferBlobDescs(bn2blobdesc_func, nullptr);

  for (const std::string& obn : clone_op->output_bns()) {
    const BlobDesc* out_blob_desc = bn2blobdesc_func(obn);
    ASSERT_TRUE(*in_blob_desc == *out_blob_desc);
  }
}

template<typename T, bool has_data_id>
void TestCloneOp() {
  test::MockJobDesc mock_job_desc;
  test::InitJobDescSingleton(&mock_job_desc);
  EXPECT_CALL(mock_job_desc, DefaultDataType())
      .WillRepeatedly(testing::Return(GetDataType<T>::val));

  int out_num = 3;
  std::vector<int64_t> in_shape = {3, 4};
  DoCloneOpTest<T, has_data_id>(out_num, in_shape);

  out_num = 1;
  DoCloneOpTest<T, has_data_id>(out_num, in_shape);
}

TEST(CloneOp, infer_blob_desc) {
#define MAKE_ENTRY(x, y) TestCloneOp<OF_PP_PAIR_FIRST(x), y>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, ALL_DATA_TYPE_SEQ, BOOL_SEQ)
}

}  // namespace oneflow
