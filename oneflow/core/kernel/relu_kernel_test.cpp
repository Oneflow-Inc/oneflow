#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void ReluTestCase(OpKernelTestCase* relu_test_case, const std::string& job_type,
                  const std::string& forward_or_backward) {
  relu_test_case->set_is_train(job_type == "train");
  relu_test_case->set_is_forward(forward_or_backward == "forward");
  relu_test_case->mut_op_conf()->mutable_relu_conf();

  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 8}), GetDataType<T>::value, false, false, 1);
  relu_test_case->template InitBlob<T>("in", blob_desc,
                                       {1, -1, -2, 2, 0, 5, -10, 100});
  relu_test_case->template InitBlob<T>(GenDiffBn("out"), blob_desc,
                                       {-8, 7, -6, 5, -4, 3, -2, 1});
  relu_test_case->template ForwardCheckBlob<T>("out", blob_desc,
                                               {1, 0, 0, 2, 0, 5, 0, 100});
  relu_test_case->template BackwardCheckBlob<T>(GenDiffBn("in"), blob_desc,
                                                {-8, 0, 0, 5, 0, 3, 0, 1});
}
TEST_CPU_AND_GPU_OPKERNEL(ReluTestCase, FLOATING_DATA_TYPE_SEQ,
                          (train)(predict), (forward)(backward));

DiffKernelImplTestCase* DiffReluKernelImpl(const std::string& job_type,
                                           const std::string& fw_or_bw,
                                           const std::string& cpp_type) {
  auto* test_case =
      new DiffKernelImplTestCase(job_type == "train", fw_or_bw == "forward",
                                 DataType4CppTypeString(cpp_type));
  test_case->mut_op_conf()->mutable_relu_conf();
  test_case->SetBlobNames({"in"}, {"out"}, {GenDiffBn("out")},
                          {GenDiffBn("in")});
  test_case->SetInputBlobDesc("in", Shape({1, 8}),
                              DataType4CppTypeString(cpp_type));
  return test_case;
}
TEST_DIFF_KERNEL_IMPL(DiffReluKernelImpl, (train)(predict), (forward)(backward),
                      OF_PP_SEQ_MAP(OF_PP_PAIR_FIRST, FLOATING_DATA_TYPE_SEQ));

}  // namespace test

}  // namespace oneflow
