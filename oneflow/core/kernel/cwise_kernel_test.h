#ifndef ONEFLOW_CORE_KERNEL_CWISE_KERNEL_TEST_H_
#define ONEFLOW_CORE_KERNEL_CWISE_KERNEL_TEST_H_

#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<OperatorConf::OpTypeCase op_type_case>
void SetCWiseOpConf(OperatorConf* op_conf, int input_cnt) {
  UNIMPLEMENTED();
}

template<OperatorConf::OpTypeCase op_type_case>
DiffKernelImplTestCase* DiffCWiseKernelImpl(const std::string& job_type,
                                            const std::string& fw_or_bw,
                                            const std::string& cpp_type,
                                            const std::string& input_cnt_str) {
  DataType data_type = DataType4CppTypeString(cpp_type);
  auto* test_case = new DiffKernelImplTestCase(
      job_type == "train", fw_or_bw == "forward", data_type);
  int input_cnt = std::stoi(input_cnt_str);
  SetCWiseOpConf<op_type_case>(test_case->mut_op_conf(), input_cnt);
  std::list<std::string> input_bns;
  std::list<std::string> input_diff_bns;
  FOR_RANGE(int, i, 0, input_cnt) {
    const auto& bn_in_op = std::string("in_") + std::to_string(i);
    test_case->SetInputBlobDesc(bn_in_op, Shape({2, 8}), data_type);
    input_bns.push_back(bn_in_op);
    input_diff_bns.push_back(GenDiffBn(bn_in_op));
  }
  test_case->SetBlobNames(input_bns, {"out"}, {GenDiffBn("out")},
                          input_diff_bns);
  return test_case;
}

}  // namespace test

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CWISE_KERNEL_TEST_H_
