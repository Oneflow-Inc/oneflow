#include "oneflow/core/kernel/clone_kernel.h"
#include <random>
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/kernel/kernel_test_common.h"

namespace oneflow {

namespace test {

namespace {

template<DeviceType device_type, typename T>
Kernel* BuildCloneKernel(int out_num) {
  OperatorConf op_conf;
  op_conf.set_name("clone_test");
  CloneOpConf* clone_conf = op_conf.mutable_clone_conf();
  clone_conf->set_out_num(out_num);
  clone_conf->set_lbn("clone_lbn");
  clone_conf->set_data_type(GetDataType<T>::val);
  auto clone_op = ConstructOp(op_conf);
  OperatorProto op_proto;
  clone_op->ToProto(&op_proto);
  auto clone_kernel = new CloneKernel<device_type, T>();
  clone_kernel->InitFromOpProto(op_proto);
  return clone_kernel;
}

template<DeviceType device_type, typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobFunc(int out_num) {
  auto blob_desc = new BlobDesc(Shape({1, 3, 2}), GetDataType<T>::val, false);

  using KTC = KTCommon<device_type, T>;

  auto bn2blob = new HashMap<std::string, Blob*>;
  (*bn2blob)["in"] = KTC::CreateBlobWithSameVal(blob_desc, 1);
  (*bn2blob)[GenDiffBn("in")] = KTC::CreateBlobWithRandomVal(blob_desc);
  (*bn2blob)["in_diff_expected"] =
      KTC::CreateBlobWithSameVal(blob_desc, 4 * out_num);
  for (size_t i = 0; i != out_num; ++i) {
    (*bn2blob)["out_" + std::to_string(i)] =
        KTC::CreateBlobWithRandomVal(blob_desc);
    (*bn2blob)["out_" + std::to_string(i) + "_diff"] =
        KTC::CreateBlobWithSameVal(blob_desc, 4);
  }
  return [bn2blob](const std::string& bn) { return bn2blob->at(bn); };
}

template<DeviceType device_type, typename T>
void TestCloneKernel(bool need_backward) {
  KernelCtx ctx;
  BuildKernelCtx<device_type>(&ctx);
  const int out_num = 3;
  auto BnInOp2BlobFunc = BuildBnInOp2BlobFunc<device_type, T>(out_num);
  auto clone_kernel = BuildCloneKernel<device_type, T>(out_num);

  clone_kernel->Forward(ctx, BnInOp2BlobFunc);
  if (need_backward) { clone_kernel->Backward(ctx, BnInOp2BlobFunc); }
  SyncStream<device_type>(&ctx);

  for (size_t i = 0; i != out_num; ++i) {
    KTCommon<device_type, T>::CheckResult(BnInOp2BlobFunc, "in",
                                          "out_" + std::to_string(i));
  }
  if (need_backward) {
    KTCommon<device_type, T>::CheckResult(BnInOp2BlobFunc, GenDiffBn("in"),
                                          "in_diff_expected");
  }
}

}  // namespace

}  // namespace test

TEST(CloneKernel, clone) {
#define MAKE_ENTRY(device_type, type_pair)                         \
  test::TestCloneKernel<device_type, OF_PP_PAIR_FIRST(type_pair)>( \
      IsFloatingPoint(OF_PP_PAIR_SECOND(type_pair)));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, DEVICE_TYPE_SEQ,
                                   ALL_DATA_TYPE_SEQ)
}

}  // namespace oneflow
