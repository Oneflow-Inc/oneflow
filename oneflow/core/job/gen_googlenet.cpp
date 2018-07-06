#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/job/init_op_conf.h"
#define DATA_FORMAT "channels_first"
#define CONCAT_AXIS 1
namespace oneflow {

std::string ConvBnReluBlock(const std::string& name, const std::string& in, int filters,
                            const std::vector<int> kernel_size, int strides,
                            const std::string& padding) {
  std::string op_out;
  op_out = Conv2D(name + "_conv", in, filters, padding, DATA_FORMAT, kernel_size, strides, 1, false,
                  kNone);

  op_out = BatchNorm(name + "_bn", op_out, kRelu);

  // op_out = Relu(name + "_relu", op_out);

  return op_out;
}

std::string InceptionBlock1(const std::string& name, const std::string& in, const int num1x1,
                            const std::vector<int> num5x5, const std::vector<int> num3x3d,
                            const int numPool) {
  std::string out, branch1x1, branch5x5;
  std::string branch3x3d;
  std::string branchPool;
  // 1x1 Convolution
  branch1x1 = ConvBnReluBlock(name + "_b1x1", in, num1x1, {1, 1}, 1, "same");

  // 5x5 Convolution
  out = ConvBnReluBlock(name + "_b5x5-1", in, num5x5[0], {1, 1}, 1, "same");
  branch5x5 = ConvBnReluBlock(name + "_b5x5", out, num5x5[1], {5, 5}, 1, "same");

  // Double 3x3 Convolution
  out = ConvBnReluBlock(name + "_b3x3d-1", in, num3x3d[0], {1, 1}, 1, "same");
  out = ConvBnReluBlock(name + "_b3x3d-2", out, num3x3d[1], {3, 3}, 1, "same");
  branch3x3d = ConvBnReluBlock(name + "_b3x3d", out, num3x3d[2], {3, 3}, 1, "same");

  // Average Pooling
  out = AveragePooling2D(name + "_pool_bPool", in, 3, 1, "same", DATA_FORMAT);
  branchPool = ConvBnReluBlock(name + "_bPool", out, numPool, {1, 1}, 1, "same");
  out = Concat(name + "_concat", {branch1x1, branch5x5, branch3x3d, branchPool}, CONCAT_AXIS);
  return out;
}

std::string InceptionBlock2(const std::string& name, const std::string& in, const int num3x3,
                            const std::vector<int> num3x3d) {
  std::string out;
  std::string branch3x3, branch3x3d;
  // 3x3 Convolution
  branch3x3 = ConvBnReluBlock(name + "_b3x3", in, num3x3, {3, 3}, 2, "valid");

  // Double 3x3 Convolution
  out = ConvBnReluBlock(name + "_b3x3d-1", in, num3x3d[0], {1, 1}, 1, "same");
  out = ConvBnReluBlock(name + "_b3x3d-2", out, num3x3d[1], {3, 3}, 1, "same");
  branch3x3d = ConvBnReluBlock(name + "_b3x3d", out, num3x3d[2], {3, 3}, 2, "valid");

  // Max Pooling
  out = MaxPooling2D(name + "_pool", in, 3, 2, "valid", DATA_FORMAT);
  out = Concat(name + "_concat", {branch3x3, branch3x3d, out}, CONCAT_AXIS);
  return out;
}

std::string InceptionBlock3(const std::string& name, const std::string& in, const int num1x1,
                            const std::vector<int> num7x7, const std::vector<int> num7x7d,
                            const int numPool) {
  std::string out, branch1x1, branch7x7;
  std::string branch7x7d;
  std::string branchPool;
  // 1x1 Convolution
  branch1x1 = ConvBnReluBlock(name + "_b1x1", in, num1x1, {1, 1}, 1, "same");

  // 7x7 Convolution
  out = ConvBnReluBlock(name + "_b7x7-1", in, num7x7[0], {1, 1}, 1, "same");
  out = ConvBnReluBlock(name + "_b7x7-2", out, num7x7[1], {1, 7}, 1, "same");
  branch7x7 = ConvBnReluBlock(name + "_b7x7", out, num7x7[2], {7, 1}, 1, "same");

  // Double 7x7 Convolution
  out = ConvBnReluBlock(name + "_b7x7d-1", in, num7x7d[0], {1, 1}, 1, "same");
  out = ConvBnReluBlock(name + "_b7x7d-2", out, num7x7d[1], {7, 1}, 1, "same");
  out = ConvBnReluBlock(name + "_b7x7d-3", out, num7x7d[2], {1, 7}, 1, "same");
  out = ConvBnReluBlock(name + "_b7x7d-4", out, num7x7d[3], {7, 1}, 1, "same");
  branch7x7d = ConvBnReluBlock(name + "_b7x7d", out, num7x7d[4], {1, 7}, 1, "same");

  // Average Pooling
  out = AveragePooling2D(name + "_pool_bPool", in, 3, 1, "same", DATA_FORMAT);
  branchPool = ConvBnReluBlock(name + "_bPool", out, numPool, {1, 1}, 1, "same");
  out = Concat(name + "_concat", {branch1x1, branch7x7, branch7x7d, branchPool}, CONCAT_AXIS);
  return out;
}

std::string InceptionBlock4(const std::string& name, const std::string& in,
                            const std::vector<int> num3x3, const std::vector<int> num7x7_3x3) {
  std::string out;
  std::string branch3x3, branch7x7_3x3;
  // 3x3 Convolution
  out = ConvBnReluBlock(name + "_b3x3-1", in, num3x3[0], {1, 1}, 1, "same");
  branch3x3 = ConvBnReluBlock(name + "_b3x3", out, num3x3[1], {3, 3}, 2, "valid");

  // 7x7 3x3 Convolution
  out = ConvBnReluBlock(name + "_b7x7-3x3-1", in, num7x7_3x3[0], {1, 1}, 1, "same");
  out = ConvBnReluBlock(name + "_b7x7-3x3-2", out, num7x7_3x3[1], {1, 7}, 1, "same");
  out = ConvBnReluBlock(name + "_b7x7-3x3-3", out, num7x7_3x3[2], {7, 1}, 1, "same");
  branch7x7_3x3 = ConvBnReluBlock(name + "_b7x7-3x3", out, num7x7_3x3[3], {3, 3}, 2, "valid");

  // Max Pooling
  out = MaxPooling2D(name + "_pool", in, 3, 2, "valid", DATA_FORMAT);
  out = Concat(name + "_concat", {branch3x3, branch7x7_3x3, out}, CONCAT_AXIS);
  return out;
}

std::string InceptionBlock5(const std::string& name, const std::string& in, const int num1x1,
                            const std::vector<int> num3x3, const std::vector<int> num3x3_3x3,
                            const int numPool) {
  std::string out, branch1x1;
  std::string branch3x3, branch3x3_2, branch3x3_3;
  std::string branch3x3_3x3, branch3x3_3x3_2, branch3x3_3x3_3;
  // 1x1 Convolution
  branch1x1 = ConvBnReluBlock(name + "_b1x1", in, num1x1, {1, 1}, 1, "same");
  // 3x3 Convolution
  out = ConvBnReluBlock(name + "_b3x3-1", in, num3x3[0], {1, 1}, 1, "same");
  branch3x3_2 = ConvBnReluBlock(name + "_b3x3-2", out, num3x3[1], {1, 3}, 1, "same");
  branch3x3_3 = ConvBnReluBlock(name + "_b3x3-3", out, num3x3[2], {3, 1}, 1, "same");
  branch3x3 = Concat(name + "_concat3x3", {branch3x3_2, branch3x3_3}, CONCAT_AXIS);

  // 3x3 3x3 Convolution
  out = ConvBnReluBlock(name + "_b3x3-3x3-1", in, num3x3_3x3[0], {1, 1}, 1, "same");
  out = ConvBnReluBlock(name + "_b3x3-3x3-2", out, num3x3_3x3[1], {3, 3}, 1, "same");
  branch3x3_3x3_2 = ConvBnReluBlock(name + "_b3x3_3x3_3", out, num3x3_3x3[2], {1, 3}, 1, "same");
  branch3x3_3x3_3 = ConvBnReluBlock(name + "_b3x3_3x3_4", out, num3x3_3x3[3], {3, 1}, 1, "same");
  branch3x3_3x3 = Concat(name + "_concat3x3-3x3", {branch3x3_3x3_2, branch3x3_3x3_3}, CONCAT_AXIS);

  // Average Pooling
  out = AveragePooling2D(name + "_pool", in, 3, 1, "same", DATA_FORMAT);
  out = ConvBnReluBlock(name + "_bPool", out, numPool, {1, 1}, 1, "same");
  out = Concat(name + "_concat", {branch1x1, branch3x3, branch3x3_3x3, out}, CONCAT_AXIS);
  return out;
}

void InceptionV3Model() {
  std::string out;
  std::string mixed8;

  // 299 x 299 x 3
  out = ConvBnReluBlock("conv1", "feature/out", 32, {3, 3}, 2, "valid");
  // 149 x 149 x 32
  out = ConvBnReluBlock("conv2", out, 32, {3, 3}, 1, "valid");
  // 147 x 147 x 32
  out = ConvBnReluBlock("conv3", out, 64, {3, 3}, 1, "same");
  // 147 x 147 x 64
  out = MaxPooling2D("pool1", out, 3, 2, "valid", DATA_FORMAT);
  // 73 x 73 x 64
  out = ConvBnReluBlock("conv4", out, 80, {1, 1}, 1, "valid");
  // 73 x 73 x 80
  out = ConvBnReluBlock("conv5", out, 192, {3, 3}, 1, "valid");
  // 71 x 71 x 192
  out = MaxPooling2D("pool2", out, 3, 2, "valid", DATA_FORMAT);
  // 35 x 35 x 192

  // Inception Blocks
  out = InceptionBlock1("mixed1", out, 64, {48, 64}, {64, 96, 96}, 32);
  // 35 x 35 x 256
  out = InceptionBlock1("mixed2", out, 64, {48, 64}, {64, 96, 96}, 64);
  // 35 x 35 x 288
  out = InceptionBlock1("mixed3", out, 64, {48, 64}, {64, 96, 96}, 64);
  // 35 x 35 x 288
  out = InceptionBlock2("mixed4", out, 384, {64, 96, 96});
  // 17 x 17 x 768
  out = InceptionBlock3("mixed5", out, 192, {128, 128, 192}, {128, 128, 128, 128, 192}, 192);
  // 17 x 17 x 768
  out = InceptionBlock3("mixed6", out, 192, {160, 160, 192}, {160, 160, 160, 160, 192}, 192);
  // 17 x 17 x 768
  out = InceptionBlock3("mixed7", out, 192, {128, 128, 192}, {160, 160, 160, 160, 192}, 192);
  // 17 x 17 x 768
  mixed8 = InceptionBlock3("mixed8", out, 192, {128, 128, 192}, {192, 192, 192, 192, 192}, 192);
  // 17 x 17 x 768
  out = InceptionBlock4("mixed9", mixed8, {192, 320}, {192, 192, 192, 192});
  // 8 x 8 x 1280
  out = InceptionBlock5("mixed10", out, 320, {384, 384, 384}, {448, 384, 384, 384}, 192);
  // 8 x 8 x 2048
  out = InceptionBlock5("mixed11", out, 320, {384, 384, 384}, {448, 384, 384, 384}, 192);
  // 8 x 8 x 2048

  // Prediction
  out = AveragePooling2D("pool3", out, 8, 1, "valid", DATA_FORMAT);
  // 1 x 1 x 2048
  out = Dropout("drop", out, 0.5);
  // 1 x 1 x 2048
  out = FullyConnected("fc1000", out, 1000);

  // Auxiliary
  // 17 x 17 x 768
  out = AveragePooling2D("auxPool", mixed8, 5, 3, "valid", DATA_FORMAT);
  // 5 x 5 x 768
  out = ConvBnReluBlock("auxConv1", out, 128, {1, 1}, 1, "same");
  // 5 x 5 x 128
  out = ConvBnReluBlock("auxConv2", out, 768, {5, 5}, 1, "valid");
  // 1 x 1 x 768
  out = FullyConnected("auxFc", out, 1024);
}

void GenGoogLeNet() {
  Global<JobConf1>::New();
  InitPlacementGroup();
  LOG(INFO) << "Create GoogLeNet/Inception V3.";
  InceptionV3Model();
  PrintProtoToTextFile(Global<JobConf1>::Get()->net(), "./googlenet.prototxt");
  PrintProtoToTextFile(Global<JobConf1>::Get()->placement(), "./googlenet_placement.prototxt");
  Global<JobConf1>::Delete();
}
}  // namespace oneflow

// DEFINE_int32(groups, 1, "groups number 1 or 2");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::GenGoogLeNet();
  return 0;
}
