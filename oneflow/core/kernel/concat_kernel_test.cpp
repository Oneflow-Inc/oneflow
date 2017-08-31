#include "oneflow/core/kernel/concat_kernel.h"
#include <random>
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename T>
Kernel* BuildConcatKernel() {
  OperatorConf op_conf;
  op_conf.set_name("concat_test");
  ConcatOpConf* concat_conf = op_conf.mutable_concat_conf();
  concat_conf->add_in("concat/in0");
  concat_conf->add_in("concat/in1");
  concat_conf->add_in("concat/in2");
  concat_conf->set_axis(1);
  concat_conf->set_out("concat_kernel_test");
  concat_conf->set_data_type(GetDataType<T>::val);
  auto concat_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  concat_op->ToProto(&op_proto);
  auto concat_kernel = new ConcatKernel<device_type, T>();
  concat_kernel->InitFromOpProto(op_proto);
  return concat_kernel;
}

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobMap() {
  using KTC = KTCommon<device_type, T>;

  auto bn2blob = new HashMap<std::string, Blob*>;
  BlobDesc* blob_desc212 =
      new BlobDesc(Shape({2, 1, 2}), GetDataType<T>::val, false);
  BlobDesc* blob_desc222 =
      new BlobDesc(Shape({2, 2, 2}), GetDataType<T>::val, false);
  BlobDesc* blob_desc242 =
      new BlobDesc(Shape({2, 4, 2}), GetDataType<T>::val, false);

  (*bn2blob)["in_0"] =
      KTC::CreateBlobWithSpecifiedVal(blob_desc212, {1, 2, 3, 4});
  (*bn2blob)["in_1"] = KTC::CreateBlobWithSpecifiedVal(
      blob_desc222, {5, 6, 7, 8, 9, 10, 11, 12});
  (*bn2blob)["in_2"] =
      KTC::CreateBlobWithSpecifiedVal(blob_desc212, {13, 14, 15, 16});
  (*bn2blob)["out"] = KTC::CreateBlobWithRandomVal(blob_desc242);
  (*bn2blob)["expected_out"] = KTC::CreateBlobWithSpecifiedVal(
      blob_desc242, {1, 2, 5, 6, 7, 8, 13, 14, 3, 4, 9, 10, 11, 12, 15, 16});
  (*bn2blob)[GenDiffBn("out")] = KTC::CreateBlobWithSpecifiedVal(
      blob_desc242, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  (*bn2blob)[GenDiffBn("in_0")] = KTC::CreateBlobWithRandomVal(blob_desc212);
  (*bn2blob)[GenDiffBn("in_1")] = KTC::CreateBlobWithRandomVal(blob_desc222);
  (*bn2blob)[GenDiffBn("in_2")] = KTC::CreateBlobWithRandomVal(blob_desc212);
  (*bn2blob)["expected_in_0_diff"] =
      KTC::CreateBlobWithSpecifiedVal(blob_desc212, {1, 2, 9, 10});
  (*bn2blob)["expected_in_1_diff"] = KTC::CreateBlobWithSpecifiedVal(
      blob_desc222, {3, 4, 5, 6, 11, 12, 13, 14});
  (*bn2blob)["expected_in_2_diff"] =
      KTC::CreateBlobWithSpecifiedVal(blob_desc212, {7, 8, 15, 16});

  return [bn2blob](const std::string& bn) { return bn2blob->at(bn); };
}

template<DeviceType device_type, typename T>
void TestConcatKernel() {
  using KTC = KTCommon<device_type, T>;
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);

  auto BnInOp2BlobFunc = BuildBnInOp2BlobMap<device_type, T>();
  auto concat_kernel = BuildConcatKernel<device_type, T>();

  concat_kernel->Forward(ctx, BnInOp2BlobFunc);
  concat_kernel->Backward(ctx, BnInOp2BlobFunc);
  SyncStream<device_type>(&ctx);

  KTC::CheckResult(BnInOp2BlobFunc, "out", "expected_out");
  KTC::CheckResult(BnInOp2BlobFunc, GenDiffBn("in_0"), "expected_in_0_diff");
  KTC::CheckResult(BnInOp2BlobFunc, GenDiffBn("in_1"), "expected_in_1_diff");
  KTC::CheckResult(BnInOp2BlobFunc, GenDiffBn("in_2"), "expected_in_2_diff");
}

}  // namespace

}  // namespace test

TEST(ConcatKernel, concat) {
#define MAKE_ENTRY(device_type, data_type_pair) \
  test::TestConcatKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, DEVICE_TYPE_SEQ,
                                   ALL_DATA_TYPE_SEQ)
}

}  // namespace oneflow
