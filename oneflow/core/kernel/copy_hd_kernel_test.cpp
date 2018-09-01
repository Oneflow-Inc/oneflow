#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void CopyLocalTestCase(OpKernelTestCase* test_case, const std::string& job_type,
                       const std::string& h2d) {
  test_case->set_default_device_type(DeviceType::kGPU);
  test_case->set_is_train(job_type == "train");
  test_case->set_is_forward(true);
  CopyLocalOpConf* copy_local_conf = test_case->mut_op_conf()->mutable_copy_local_conf();
  CopyLocalOpConf::Type hd_type = (h2d == "h2d" ? CopyLocalOpConf::H2D : CopyLocalOpConf::D2H);
  copy_local_conf->set_type(hd_type);
  test_case->SetBlobSpecializedDeviceType((hd_type == CopyLocalOpConf::H2D ? "out" : "in"),
                                          DeviceType::kGPU);

  BlobDesc* blob_desc = new BlobDesc(Shape({3, 4, 5, 6}), GetDataType<T>::value, false, false, 1);
  test_case->RandomInitBlob<T>("in", blob_desc);
  test_case->ForwardCheckBlobWithAnother<T>("out", blob_desc, "in", true);
}

TEST_CPU_ONLY_OPKERNEL(CopyLocalTestCase, POD_DATA_TYPE_SEQ, (train)(predict), (h2d)(d2h));

}  // namespace test

}  // namespace oneflow
