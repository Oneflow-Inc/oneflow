#include "oneflow/core/operator/boxing_op.h"

namespace oneflow {

template<typename T, bool has_data_id_field>
void TestBoxingOp() {
  // input shape is
  // in1 {10, 5, 6, 6}
  // in2 {10, 4, 6, 6}
  // in3 {10, 4, 6, 6}
  // in3 {10, 4, 6, 6}
  OperatorConf op_conf;
  DataType data_type = GetDataType<T>::val;
  op_conf.set_name("boxing_test");
  BoxingOpConf* boxing_conf = op_conf.mutable_boxing_conf();
  boxing_conf->set_data_type(data_type);
  boxing_conf->set_lbn("boxing_blob");
  boxing_conf->set_in_num(4);
  boxing_conf->set_out_num(3);
  std::vector<int64_t> input_shape_vec1 = {10, 5, 6, 6};
  std::vector<int64_t> input_shape_vec2 = {10, 4, 6, 6};

  // test concat_box shape function
  boxing_conf->mutable_concat_box()->set_axis(1);
  boxing_conf->mutable_data_split_box();
  auto boxing_op = ConstructOp(op_conf);
  HashMap<std::string, BlobDesc*> bn2blobdesc_map{
      {boxing_op->input_bns()[0],
       new BlobDesc(Shape(input_shape_vec2), data_type, has_data_id_field)},
      {boxing_op->input_bns()[1],
       new BlobDesc(Shape(input_shape_vec2), data_type, has_data_id_field)},
      {boxing_op->input_bns()[2],
       new BlobDesc(Shape(input_shape_vec2), data_type, has_data_id_field)},
      {boxing_op->input_bns()[3],
       new BlobDesc(Shape(input_shape_vec1), data_type, has_data_id_field)},
      {"middle", new BlobDesc},
      {boxing_op->output_bns()[0], new BlobDesc},
      {boxing_op->output_bns()[1], new BlobDesc},
      {boxing_op->output_bns()[2], new BlobDesc},
  };
  auto fp = [&bn2blobdesc_map](const std::string& bn) {
    return bn2blobdesc_map.at(bn);
  };

  // do infer shape
  boxing_op->InferBlobDescs(fp, kModelParallel, 0, 1);

  // test results
  // output_shape should be:
  // out1 {4, 17, 6, 6}
  // out2 {3, 17, 6, 6}
  // out3 {3, 17, 6, 6}
  for (size_t i = 0; i < boxing_op->output_bns().size(); ++i) {
    auto out_blobdesc = bn2blobdesc_map.at(boxing_op->output_bns()[i]);
    std::vector<int64_t> output_shape_vec = {3, 17, 6, 6};
    if (i == 0) { output_shape_vec[0] = 4; }
    ASSERT_EQ(out_blobdesc->shape(), Shape(output_shape_vec));
    ASSERT_EQ(out_blobdesc->data_type(), data_type);
    ASSERT_EQ(out_blobdesc->has_data_id_field(), has_data_id_field);
  }

  // Test add clone box shape function
  boxing_conf->set_in_num(3);
  boxing_conf->set_out_num(1);
  boxing_conf->mutable_add_box();
  boxing_conf->mutable_clone_box();
  boxing_op = ConstructOp(op_conf);

  // do infer shape
  boxing_op->InferBlobDescs(fp, kModelParallel, 0, 1);

  // test results
  // output shape should be the same as input
  for (const std::string& bn : boxing_op->output_bns()) {
    BlobDesc* out_blobdesc = bn2blobdesc_map.at(bn);
    ASSERT_EQ(out_blobdesc->shape(), Shape(input_shape_vec2));
    ASSERT_EQ(out_blobdesc->data_type(), data_type);
    ASSERT_EQ(out_blobdesc->has_data_id_field(), has_data_id_field);
  }

  // Test concat clone shape function, this box has data_tmp_shape
  boxing_conf->set_in_num(4);
  boxing_conf->set_out_num(1);
  boxing_conf->mutable_concat_box()->set_axis(1);
  boxing_conf->mutable_clone_box();
  boxing_op = ConstructOp(op_conf);

  // do infer shape
  boxing_op->InferBlobDescs(fp, kModelParallel, 0, 1);

  // data_tmp_shape is {10, 17, 6, 6}, and the 17 = 4 + 4 + 4 + 5
  BlobDesc* data_tmp_blobdesc = bn2blobdesc_map.at(boxing_op->SoleDtbn());
  std::vector<int64_t> data_temp_shape_vec = {10, 17, 6, 6};
  ASSERT_EQ(data_tmp_blobdesc->shape(), Shape(data_temp_shape_vec));
  ASSERT_EQ(data_tmp_blobdesc->data_type(), data_type);
  ASSERT_EQ(data_tmp_blobdesc->has_data_id_field(), has_data_id_field);

  // test results
  // output shape should be the same as data_tmp_shape
  for (const std::string& bn : boxing_op->output_bns()) {
    BlobDesc* out_blobdesc = bn2blobdesc_map.at(bn);
    ASSERT_EQ(out_blobdesc->shape(), data_tmp_blobdesc->shape());
    ASSERT_EQ(out_blobdesc->data_type(), data_type);
    ASSERT_EQ(out_blobdesc->has_data_id_field(), has_data_id_field);
  }
}

TEST(BoxingOp, infer_blob_desc) {
#define MAKE_ENTRY(x, y) TestBoxingOp<OF_PP_PAIR_FIRST(x), y>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, ARITHMETIC_DATA_TYPE_SEQ,
                                   BOOL_SEQ)
}

}  // namespace oneflow
