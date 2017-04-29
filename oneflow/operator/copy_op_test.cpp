#include "operator/copy_op.h"
#include "operator/operator_manager.h"
#include "gtest/gtest.h"

namespace oneflow {

TEST(CopyOp, copy_3_3x4_shape) {
  OperatorConf op_conf;
  op_conf.set_name("copy_test");
  CopyOpConf* copy_conf = op_conf.mutable_copy_conf();
  copy_conf->set_copy_type(CopyOpConf::H2D);
  std::vector<std::string> copied_lbns = {
    "copy_lbn1", "copy_lbn2", "copy_lbn3"};
  for(std::string lbn : copied_lbns){
    copy_conf->add_copied_lbns(lbn);
  }
  auto copy_op = OpMgr::Singleton().ConstructOp(op_conf);

  std::vector<int64_t> shape_vec = {3, 4};
  for(std::string ibn : copy_op->input_bns()){
    copy_op->SetShapePtr(ibn, new Shape(shape_vec));
  }
  for(std::string obn : copy_op->output_bns()){
    copy_op->SetShapePtr(obn, new Shape);
  }
  copy_op->InferShape4ObAndDtbFromIb();

  for(int i = 0;i < copy_op->output_bns().size();i ++){
    Shape* input_shape_ptr = copy_op->GetShapePtr(
        copy_op->input_bns().at(i));
    Shape* output_shape_ptr = copy_op->GetShapePtr(
        copy_op->output_bns().at(i));
    ASSERT_TRUE(*input_shape_ptr == *output_shape_ptr);
    ASSERT_TRUE(input_shape_ptr != output_shape_ptr);
  }

}

} // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
