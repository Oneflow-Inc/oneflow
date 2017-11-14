#include "oneflow/core/operator/concat_op.h"

namespace oneflow {

TEST(ConcatOp, concat_two_3x3) {
  // create op
  OperatorConf op_conf;
  op_conf.set_name("concat_test");
  op_conf.mutable_concat_conf()->add_in("concat/in0");
  op_conf.mutable_concat_conf()->add_in("concat/in1");
  op_conf.mutable_concat_conf()->add_in("concat/in2");
  op_conf.mutable_concat_conf()->set_out("concat_test_lbn");
  op_conf.mutable_concat_conf()->set_data_type(DataType::kFloat);
  op_conf.mutable_concat_conf()->set_axis(2);
  auto concat_op = ConstructOp(op_conf);

  HashMap<std::string, BlobDesc*> bn2blob_desc_map{
      {concat_op->input_bns().at(0),
       new BlobDesc(Shape({2, 3, 4, 5}), DataType::kFloat, false)},
      {concat_op->input_bns().at(1),
       new BlobDesc(Shape({2, 3, 1, 5}), DataType::kFloat, false)},
      {concat_op->input_bns().at(2),
       new BlobDesc(Shape({2, 3, 9, 5}), DataType::kFloat, false)},
      {concat_op->SoleObn(), new BlobDesc}};
  auto bn2blob_desc_func = [&](const std::string& bn) {
    return bn2blob_desc_map.at(bn);
  };
  concat_op->InferBlobDescs(bn2blob_desc_func, kDataParallel, 0, 1);
  ASSERT_TRUE(*(bn2blob_desc_map.at(concat_op->SoleObn()))
              == BlobDesc(Shape({2, 3, 14, 5}), DataType::kFloat, false));
}

}  // namespace oneflow
