#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

namespace {

std::string& ExpectedBlobName(const std::string& name) {
  return name + "_$expected$";
}

}

void OpKernelTestCaseBuilder::InitBlob(const std::string& name, std::unique_ptr<Blob>&& blob) {
  CHECK(mut_bn_in_op2blob()->emplace_back(name, blob).second);
}

void OpKernelTestCaseBuilder::ForwardAssertEqBlob(const std::string& name, std::unique_ptr<Blob>&& blob) {
  mut_forward_asserted_blob()->push_back(name);
  CHECK(mut_bn_in_op2blob()->emplace_back(ExpectedBlobName(name), blob).second);
}

void OpKernelTestCaseBuilder::BackwardAssertEqBlob(const std::string& name, std::unique_ptr<Blob>&& blob) {
  mut_backward_asserted_blob()->push_back(name);
  CHECK(mut_bn_in_op2blob()->emplace_back(ExpectedBlobName(name), blob).second);
}

std::function<Blob*(const std::string&)>
OpKernelTestCase::MakeGetterBnInOp2Blob() const {
  auto bn_in_op2blob = bn_in_op2blob_;
  return [bn_in_op2blob](const std::string& bn_in_op) {
    return  (*bn_in_op2blob)[bn_in_op].get();
  }
}

void OpKernelTestCase::Build() {
  OpKernelTestCaseBuilder opk_test_case_builder(this);
  Build(opk_test_case_builder);
}

std::function<BlobDesc*(const std::string&)>
OpKernelTestCase::MakeGetterBnInOp2BlobDesc() const {
  auto bn_in_op2blob_desc = bn_in_op2blob_desc_;
  return [bn_in_op2blob_desc](const std::string& bn_in_op) {
    return  (*bn_in_op2blob_desc)[bn_in_op].get();
  }
}


}

}
