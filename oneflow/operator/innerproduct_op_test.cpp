#include "operator/innerproduct_op.h"
#include "operator/operator_manager.h"
#include "gtest/gtest.h"

namespace oneflow {

HaspMap<std::string, Shape*> bn_in_op2shape_ptr;

Shape* GetShapePtr4BnInOp(const std::string& bn){
  return bn_in_op2shape_ptr[bn];
}

TEST(InnerProductOp, modelparallel_innerproduct) {
  OperatorConf op_conf;
  op_conf.set_name("modelparallel_ip_test");
  op_conf.mutable_innerproduct_conf()->set_in("ip_in");
  op_conf.mutable_innerproduct_conf()->set_out("ip_out");
  op_conf.mutable_innerproduct_conf()->set_out_num(40);
  op_conf.mutable_innerproduct_conf()->set_axis(1);
  auto ip_op = OpMgr::Singleton().ConstructOp(op_conf);

  bn_in_op2shape_ptr.clear();
  std::vector<int64_t> shape_vec = {1000, 3, 256, 256};
  bn_in_op2shape_ptr[ip_op->SoleIbn()] = new Shape(shape_vec);
  bn_in_op2shape_ptr[ip_op->SoleObn()] = new Shape;
  bn_in_op2shape_ptr[ip_op->model_bns().at(0)] = new Shape;
  bn_in_op2shape_ptr[ip_op->model_bns().at(1)] = new Shape;
  bn_in_op2shape_ptr[ip_op->model_tmp_bns().at(0)] = new Shape;

  ip_op->InferShape4FwBlobs(GetShapePtr4BnInOp, kModelparallel, 3, 10);

  BalancedSplitter splitter(10, 4);
  int out_num = splitter.At(3).size();

  Shape* in_shape_ptr = GetShapePtr4BnInOp(ip_op->SoleIbn());
  Shape* out_shape_ptr = GetShapePtr4BnInOp(ip_op->SoleObn());
  CHECK_EQ(*out_shape_ptr, 
      Shape(std::vector<int64_t>(4, {1000, out_num, 1, 1})));
  Shape* weight_shape_ptr = GetShapePtr4BnInOp(ip_op->model_bns().at(0));
  CHECK_EQ(*weight_shape_ptr, 
      Shape(std::vector<int64_t>(4, {out_num, 3*256*256, 1, 1})));
  Shape* bias_shape_ptr = GetShapePtr4BnInOp(ip_op->model_bns().at(1));
  CHECK_EQ(*bias_shape_ptr, 
      Shape(std::vector<int64_t>(4, {1, out_num, 1, 1})));
  Shape* bias_multiplier_shape_ptr = 
    GetShapePtr4BnInOp(ip_op->model_tmp_bns().at(0));
  CHECK_EQ(*bias_multiplier_shape_ptr, 
      Shape(std::vector<int64_t>(4, {1000, 1, 1, 1})));
}

TEST(InnerProductOp, dataparallel_innerproduct) {
  OperatorConf op_conf;
  op_conf.set_name("dataparallel_ip_test");
  op_conf.mutable_innerproduct_conf()->set_in("ip_in");
  op_conf.mutable_innerproduct_conf()->set_out("ip_out");
  op_conf.mutable_innerproduct_conf()->set_out_num(40);
  op_conf.mutable_innerproduct_conf()->set_axis(1);
  auto ip_op = OpMgr::Singleton().ConstructOp(op_conf);

  bn_in_op2shape_ptr.clear();
  std::vector<int64_t> shape_vec = {1000, 3, 256, 256};
  bn_in_op2shape_ptr[ip_op->SoleIbn()] = new Shape(shape_vec);
  bn_in_op2shape_ptr[ip_op->SoleObn()] = new Shape;
  bn_in_op2shape_ptr[ip_op->model_bns().at(0)] = new Shape;
  bn_in_op2shape_ptr[ip_op->model_bns().at(1)] = new Shape;
  bn_in_op2shape_ptr[ip_op->model_tmp_bns().at(0)] = new Shape;

  ip_op->InferShape4FwBlobs(GetShapePtr4BnInOp, kDataparallel, 3, 10);

  Shape* in_shape_ptr = GetShapePtr4BnInOp(ip_op->SoleIbn());
  Shape* out_shape_ptr = GetShapePtr4BnInOp(ip_op->SoleObn());
  CHECK_EQ(*out_shape_ptr, 
      Shape(std::vector<int64_t>(4, {1000, 40, 1, 1})));
  Shape* weight_shape_ptr = GetShapePtr4BnInOp(ip_op->model_bns().at(0));
  CHECK_EQ(*weight_shape_ptr, 
      Shape(std::vector<int64_t>(4, {40, 3*256*256, 1, 1})));
  Shape* bias_shape_ptr = GetShapePtr4BnInOp(ip_op->model_bns().at(1));
  CHECK_EQ(*bias_shape_ptr, 
      Shape(std::vector<int64_t>(4, {1, 40, 1, 1})));
  Shape* bias_multiplier_shape_ptr = 
    GetShapePtr4BnInOp(ip_op->model_tmp_bns().at(0));
  CHECK_EQ(*bias_multiplier_shape_ptr, 
      Shape(std::vector<int64_t>(4, {1000, 1, 1, 1})));
}


} // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
