#include "oneflow/core/kernel/conv2d_kernel.h"
#include "oneflow/core/kernel/kernel_test_common.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobFunc() {
  using KTC = KTCommon<device_type, T>;
  auto bn2blob = new HashMap<std::string, Blob*>;
  (*bn2blob)["in"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({1, 1, 3, 3}), GetDataType<T>::val, false),
      {1, -1, 2, 3, 1, -1, 2, 1, -2});
  (*bn2blob)["col_buf"] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({1, 4, 4}), GetDataType<T>::val, false));
  (*bn2blob)["filter"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2, 4}), GetDataType<T>::val, false),
      {0.1f, 0.2f, 0.2f, 0.4f, -0.3f, 0.1f, 0.1f, 0.2f});
  (*bn2blob)["out"] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({1, 2, 2, 2}), GetDataType<T>::val, false));
  (*bn2blob)["bias"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2}), GetDataType<T>::val, false), {0, 0});
  (*bn2blob)[GenDiffBn("out")] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({1, 2, 2, 2}), GetDataType<T>::val, false),
      {1, 3, 2, 2, 2, 1, 1, 1});
  (*bn2blob)[GenDiffBn("in")] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({1, 1, 3, 3}), GetDataType<T>::val, false));
  (*bn2blob)[GenDiffBn("filter")] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({2, 4}), GetDataType<T>::val, false));
  (*bn2blob)[GenDiffBn("bias")] = KTC::CreateBlobWithRandomVal(
      new BlobDesc(Shape({2}), GetDataType<T>::val, false));
  (*bn2blob)["bias_multiplier"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({4}), GetDataType<T>::val, false), {1, 1, 1, 1});
  (*bn2blob)["expected_out"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({1, 2, 2, 2}), GetDataType<T>::val, false),
      {0.9f, 0.1f, 1.3f, -0.7f, 0.1f, 0.4f, -0.4f, -0.7f});
  (*bn2blob)["expected_bias_diff"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2}), GetDataType<T>::val, false), {2, 1.25f});
  (*bn2blob)["expected_in_diff"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({1, 1, 3, 3}), GetDataType<T>::val, false),
      {-0.5f, 0.4f, 0.7f, 0.3f, 1.9f, 1.9f, 0.5f, 1.5f, 1.0f});
  (*bn2blob)["expected_filter_diff"] = KTC::CreateBlobWithSpecifiedVal(
      new BlobDesc(Shape({2, 4}), GetDataType<T>::val, false),
      {1.5f, 1.25f, 3, -1, 1.25f, 0, 2.5f, 0});
  return [bn2blob](const std::string& bn) { return bn2blob->at(bn); };
}

template<DeviceType device_type, typename T>
Kernel* BuildConv2dKernel() {
  OperatorConf op_conf;
  op_conf.set_name("conv2d_test");
  op_conf.mutable_conv2d_conf()->set_in("conv2d/in");
  op_conf.mutable_conv2d_conf()->set_out("conv2d/out");
  op_conf.mutable_conv2d_conf()->set_out_num(1);
  op_conf.mutable_conv2d_conf()->set_pad_h(0);
  op_conf.mutable_conv2d_conf()->set_pad_w(0);
  op_conf.mutable_conv2d_conf()->set_kernel_size_h(2);
  op_conf.mutable_conv2d_conf()->set_kernel_size_w(2);
  op_conf.mutable_conv2d_conf()->set_stride_h(1);
  op_conf.mutable_conv2d_conf()->set_stride_w(1);
  op_conf.mutable_conv2d_conf()->set_dilation_h(1);
  op_conf.mutable_conv2d_conf()->set_dilation_w(1);
  auto conv2d_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  conv2d_op->ToProto(&op_proto);
  auto conv_kernel = new Conv2dKernel<device_type, T>();
  conv_kernel->InitFromOpProto(op_proto);
  return conv_kernel;
}

template<DeviceType device_type, typename T>
void TestConv2dKernel() {
  JobConf job_conf;
  job_conf.set_DefaultDataType(GetDataType<T>::val);
  JobDesc::Singleton()->InitFromJobConf(job_conf);
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);
  auto BnInOp2BlobFunc = BuildBnInOp2BlobFunc<device_type, T>();
  auto conv_kernel = BuildConv2dKernel<device_type, T>();
  conv_kernel->Forward(ctx, BnInOp2BlobFunc);
  conv_kernel->Backward(ctx, BnInOp2BlobFunc);
  SyncStream<device_type>(&ctx);
  KTCommon<device_type, T>::CheckResult(BnInOp2BlobFunc, "out", "expected_out");
  KTCommon<device_type, T>::CheckResult(BnInOp2BlobFunc, GenDiffBn("in"),
                                        "expected_in_diff");
  KTCommon<device_type, T>::CheckResult(BnInOp2BlobFunc, GenDiffBn("filter"),
                                        "expected_filter_diff");
  KTCommon<device_type, T>::CheckResult(BnInOp2BlobFunc, GenDiffBn("bias"),
                                        "expected_bias_diff");
}

}  // namespace

TEST(ConvKernel, conv_kernel_cpu) {
#define MAKE_ENTRY(x, y, z) test::TestConv2dKernel<x, OF_PP_PAIR_FIRST(y)>(z);
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, DEVICE_TYPE_SEQ,
                                   FLOATING_DATA_TYPE_SEQ, BOOL_SEQ)
}

}  // namespace test

}  // namespace oneflow
