#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void CopyHdTestCase(OpKernelTestCase<device_type>* test_case,
                    const std::string& job_type,
                    const std::string& forward_or_backward,
                    const std::string& h2d) {
  test_case->set_is_train(job_type == "train");
  test_case->set_is_forward(forward_or_backward == "forward");
  test_case->mut_op_conf()->mutable_copy_hd_conf();
  CopyHdOpConf* copy_hd_conf = op_conf.mutable_copy_hd_conf();
  CopyHdOpConf::Type hd_type =
      (h2d == "h2d" ? CopyHdOpConf::H2D : CopyHdOpConf::D2H);
  copy_hd_conf->set_type(hd_type);
  auto SetBlobGpuDeviceType = [&](const std::string& name) {
    test_case->SetBlobSpecializedDeviceType(name, DeviceType::kGPU);
    test_case->SetBlobSpecializedDeviceType(GetDiffBn(name), DeviceType::kGPU);
  };
  SetBlobGpuDeviceType(hd_type == CopyHdOpConf::H2D ? "out" : "in");
  BlobDesc* blob_desc =
      new BlobDesc(Shape({3, 4, 5, 6}), GetDataType<T>::value, false);
  test_case->template RandomInitBlob<T>("in", blob_desc);
  test_case->set_initiation_before_backward([test_case]() {
    Blob* out = test_case->bn_in_op2blob().at("out");
    test_case->mut_bn_in_op2blob()->emplace(GenDiffBn("out"), out);
  });
}

TEST_CPU_ONLY_OPKERNEL(CopyHdTestCase, FLOATING_DATA_TYPE_SEQ, (train)(predict),
                       (forward)(backward));

}  // namespace test

}  // namespace oneflow

namespace oneflow {

namespace test {

namespace {

template<typename T>
std::function<Blob*(const std::string&)> BuildBnInOp2BlobFunc(
    CopyHdOpConf::Type hd_type) {
  BlobDesc* blob_desc =
      new BlobDesc(Shape({3, 4, 5, 6}), GetDataType<T>::value, false);

  auto bn2blob = new HashMap<std::string, Blob*>;

  if (hd_type == CopyHdOpConf::H2D) {
    (*bn2blob)["in"] =
        KTCommon<DeviceType::kCPU, T>::CreateBlobWithRandomVal(blob_desc);
    (*bn2blob)["out"] =
        KTCommon<DeviceType::kGPU, T>::CreateBlobWithRandomVal(blob_desc);
    (*bn2blob)[GenDiffBn("in")] =
        KTCommon<DeviceType::kCPU, T>::CreateBlobWithRandomVal(blob_desc);
  } else {
    (*bn2blob)["in"] =
        KTCommon<DeviceType::kGPU, T>::CreateBlobWithRandomVal(blob_desc);
    (*bn2blob)["out"] =
        KTCommon<DeviceType::kCPU, T>::CreateBlobWithRandomVal(blob_desc);
    (*bn2blob)[GenDiffBn("in")] =
        KTCommon<DeviceType::kGPU, T>::CreateBlobWithRandomVal(blob_desc);
  }
  (*bn2blob)["out_diff"] = (*bn2blob)["out"];
  return [bn2blob](const std::string& bn) { return bn2blob->at(bn); };
}

template<typename T>
Kernel* BuildCopyHdKernel(CopyHdOpConf::Type hd_type) {
  OperatorConf op_conf;
  op_conf.set_name("copy_hd_test");
  CopyHdOpConf* copy_hd_conf = op_conf.mutable_copy_hd_conf();
  copy_hd_conf->set_type(hd_type);
  auto copy_hd_op = ConstructOp(op_conf);

  OperatorProto op_proto;
  copy_hd_op->ToProto(&op_proto);
  auto copy_hd_kernel = new CopyHdKernel();
  copy_hd_kernel->InitFromOpProto(op_proto);
  return copy_hd_kernel;
}

template<typename T>
void TestCopyHdKernel(CopyHdOpConf::Type hd_type) {
  KernelCtx ctx;
  BuildKernelCtx<DeviceType::kGPU>(&ctx);
  auto BnInOp2BlobFunc = BuildBnInOp2BlobFunc<T>(hd_type);
  auto copy_hd_kernel = BuildCopyHdKernel<T>(hd_type);

  copy_hd_kernel->Forward(ctx, BnInOp2BlobFunc);
  copy_hd_kernel->Backward(ctx, BnInOp2BlobFunc);
  SyncStream<DeviceType::kGPU>(&ctx);
  if (hd_type == CopyHdOpConf::H2D) {
    KTCommon<DeviceType::kCPU, T>::CheckResult(BnInOp2BlobFunc, "in",
                                               "in_diff");
  } else {
    KTCommon<DeviceType::kGPU, T>::CheckResult(BnInOp2BlobFunc, "in",
                                               "in_diff");
  }
}

}  // namespace

}  // namespace test

TEST(CopyHdKernel, copy_d2h) {
#define COPY_TYPE_SEQ (CopyHdOpConf::D2H)(CopyHdOpConf::H2D)
#define MAKE_ENTRY(x, y) test::TestCopyHdKernel<OF_PP_PAIR_FIRST(x)>(y);
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, ALL_DATA_TYPE_SEQ, COPY_TYPE_SEQ)
}

}  // namespace oneflow
