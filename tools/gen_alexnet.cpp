#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/job/init_op_conf.h"

namespace oneflow {

std::string AlexNetConv2DBlock(const std::string& name, const std::string in, const int filters,
                               const std::string& padding = "same",
                               const std::string data_format = "channels_last",
                               const int kernel_size = 3, const int strides = 1,
                               const int dilation_rate = 1, const bool use_bias = true,
                               bool with_lrn = false, bool with_pooling = false) {
  std::string op_name, op_out;
  op_out = Conv2D(name, in, filters, padding, data_format, {kernel_size, kernel_size}, strides,
                  dilation_rate, use_bias);

  op_name = "relu_" + name;
  op_out = Relu(op_name, op_out);

  if (with_lrn) {
    op_name = "lrn_" + name;
    op_out = LocalResponseNormalization(op_name, op_out, 2, 2.0, 1e-4,
                                        0.75);  // add lrn support here
  }

  if (with_pooling) {
    op_name = "pool_" + name;
    op_out = MaxPooling2D(op_name, op_out, 3, 2);
  }

  return op_out;
}

std::string AlexNetFeature(const std::string& in) {
  std::string op_out = in;
  // features
  op_out = AlexNetConv2DBlock("conv1", op_out, 64 /*filter number*/, "same", "channels_last",
                              11 /*kernel_size*/, 4 /*stride*/, 1 /*dilation_rate*/,
                              true /*use_bias*/, true /*with_lrn*/, true /*with_pooling*/);
  op_out = AlexNetConv2DBlock("conv2", op_out, 192 /*filter number*/, "same", "channels_last",
                              5 /*kernel_size*/, 1 /*stride*/, 1 /*dilation_rate*/,
                              true /*use_bias*/, true /*with_lrn*/, true /*with_pooling*/);

  op_out = AlexNetConv2DBlock("conv3", op_out, 384 /*filter number*/, "same", "channels_last",
                              3 /*kernel_size*/, 1 /*stride*/, 1 /*dilation_rate*/,
                              true /*use_bias*/, false /*with_lrn*/, false /*with_pooling*/);

  op_out = AlexNetConv2DBlock("conv4", op_out, 256 /*filter number*/, "same", "channels_last",
                              3 /*kernel_size*/, 1 /*stride*/, 1 /*dilation_rate*/,
                              true /*use_bias*/, false /*with_lrn*/, false /*with_pooling*/);

  op_out = AlexNetConv2DBlock("conv5", op_out, 256 /*filter number*/, "same", "channels_last",
                              3 /*kernel_size*/, 1 /*stride*/, 1 /*dilation_rate*/,
                              true /*use_bias*/, false /*with_lrn*/, true /*with_pooling*/);
  return op_out;
}

std::string Classifier(const std::string& in, const int num_classes, bool with_dropout = false,
                       int first_fc_layer_num = 6) {
  std::string op_out, name;
  if (with_dropout) {
    name = "drop_" + std::to_string(first_fc_layer_num);
    op_out = Dropout(name, in, 0.5);
  } else {
    op_out = in;
  }
  // classifier
  // nn.Dropout(),
  // nn.Linear(256 * 6 * 6, 4096)
  name = "fc" + std::to_string(first_fc_layer_num);
  op_out = FullyConnected(name, op_out, 4096);
  // nn.ReLU(inplace=True),
  name = "relu_" + name;
  op_out = Relu(name, op_out);
  // nn.Dropout(),
  if (with_dropout) {
    name = "drop_" + std::to_string(first_fc_layer_num + 1);
    op_out = Dropout(name, op_out, 0.5);
  }
  // nn.Linear(4096, 4096),
  name = "fc" + std::to_string(first_fc_layer_num + 1);
  op_out = FullyConnected(name, op_out, 4096);
  // nn.ReLU(inplace=True),
  name = "relu_" + name;
  op_out = Relu(name, op_out);
  // nn.Linear(4096, num_classes),
  name = "fc" + std::to_string(first_fc_layer_num + 2);
  op_out = FullyConnected(name, op_out, num_classes);
  return op_out;
}

void GenAlexNet(const int groups) {
  Global<JobConf1>::New();
  InitPlacementGroup();
  LOG(INFO) << "Create AlexNet. groups = " << groups;
  std::string op_out;
  op_out = AlexNetFeature("feature");
  op_out = Classifier(op_out, 1000, true);
  op_out = Softmax("prob", op_out);
  PrintProtoToTextFile(Global<JobConf1>::Get()->net(), "./alexnet.prototxt");
  PrintProtoToTextFile(Global<JobConf1>::Get()->placement(), "./alexnet_placement.prototxt");
  Global<JobConf1>::Delete();
}
}  // namespace oneflow

DEFINE_int32(groups, 1, "groups number 1 or 2");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::GenAlexNet(FLAGS_groups);
  return 0;
}
