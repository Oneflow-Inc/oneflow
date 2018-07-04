#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void CloneTestCase(OpKernelTestCase* clone_test_case, const std::string& job_type,
                   const std::string& forward_or_backward) {
  const size_t copies = 3;
  clone_test_case->set_is_train(job_type == "train");
  clone_test_case->set_is_forward(forward_or_backward == "forward");
  CloneOpConf* clone_conf = clone_test_case->mut_op_conf()->mutable_clone_conf();
  clone_conf->set_out_num(copies);
  clone_conf->set_lbn("clone_lbn");

  auto blob_desc = new BlobDesc(Shape({1, 3, 2}), GetDataType<T>::value, false, false, 1);
  clone_test_case->template InitBlob<T>("in", blob_desc, {1, 2, 3, 4, 5, 6});
  for (size_t i = 0; i < copies; ++i) {
    clone_test_case->template ForwardCheckBlob<T>("out_" + std::to_string(i), blob_desc,
                                                  {1, 2, 3, 4, 5, 6});
    clone_test_case->template InitBlob<T>("out_" + std::to_string(i) + "_diff", blob_desc,
                                          {6, 5, 4, 3, 2, 1});
  }
  clone_test_case->template BackwardCheckBlob<T>(GenDiffBn("in"), blob_desc, {18, 15, 12, 9, 6, 3});
}

TEST_CPU_AND_GPU_OPKERNEL(CloneTestCase, FLOATING_DATA_TYPE_SEQ, (train)(predict),
                          (forward)(backward));

}  // namespace test

}  // namespace oneflow
