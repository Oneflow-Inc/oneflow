#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/job/init_op_conf.h"
#define DATA_FORMAT "channels_first"

namespace oneflow {
// conv + batch_norm + relu
std::string Conv2DBlock(bool use_relu, const std::string& name, const std::string& in,
                        const int filters, const std::string& padding = "same",
                        const std::string& data_format = DATA_FORMAT, const int kernel_size = 3,
                        const int strides = 1, const int dilation_rate = 1,
                        const bool use_bias = false) {
  std::string op_name, op_out;
  op_out = Conv2D(name, in, filters, padding, data_format, {kernel_size, kernel_size}, strides,
                  dilation_rate, use_bias);
  if (use_relu) {
    op_name = "bn_" + name;
    op_out = BatchNorm(op_name, op_out, kRelu, 1, 0.997,
                       0.0000101);  // out of last op is the input of this
  } else {
    op_name = "bn_" + name;
    op_out = BatchNorm(op_name, op_out, kNone, 1, 0.997,
                       0.0000101);  // out of last op is the input of this
  }

  return op_out;
}

// one resnet contains 4 residual blocks, id from 2 to 5
// res id: residual block id, from 2 to 5.
// residual block name: res(res id)
// one residual block contains some building blocks
// building block id in residual block.
// building_block_name: res*_bb(building block id)
// e.g. res2a(caffe) -> res2_bb1(of), res3c -> res3_bb3
// one building block contains 3 conv blocks(>50), 2 conv blocks(18, 34)
// conv_block_name: res*_bb*_b1a/b/c(main branch), res*_bb*_b2(shortcut branch)
std::string BuildingBlock(const std::string& res_block_name, int building_block_id,
                          const std::string& in, int filter1_2, int filter3,
                          bool down_sampling = true) {
  std::string op_out, building_block_name, name;
  std::string b2_out = in;
  int stride = 1;
  if (building_block_id == 1 && down_sampling) stride = 2;
  building_block_name = res_block_name + "_bb" + std::to_string(building_block_id);

  // shortcup branch - b2
  if (building_block_id == 1) {
    name = building_block_name + "_b2";
    b2_out = Conv2DBlock(false /*no relu*/, name, in, filter3, "same", DATA_FORMAT, 1, stride);
  }

  // main branch - b1
  name = building_block_name + "_b1a";
  op_out = Conv2DBlock(true, name, in, filter1_2, "same", DATA_FORMAT, 1, stride);
  name = building_block_name + "_b1b";
  op_out = Conv2DBlock(true, name, op_out, filter1_2, "same", DATA_FORMAT, 3, 1);
  name = building_block_name + "_b1c";
  op_out = Conv2DBlock(false /*no relu*/, name, op_out, filter3, "same", DATA_FORMAT, 1, 1);
  // element wise sum
  std::vector<std::string> v = {op_out, b2_out};
  name = building_block_name + "_add";
  op_out = Add(name, v, kRelu);

  return op_out;
}

std::string ResidualBlock(int res_block_id, int building_block_num, const std::string& in,
                          int filter1_2, int filter3, bool down_sampling = true) {
  std::string res_block_name = "res" + std::to_string(res_block_id);
  std::string op_out = in;
  bool ds;
  for (int i = 0; i < building_block_num; ++i) {
    if (i == 0 && down_sampling && res_block_id > 2)
      ds = true;
    else
      ds = false;
    op_out = BuildingBlock(res_block_name, i + 1, op_out, filter1_2, filter3, ds);
  }
  return op_out;
}

std::string ResidualBlocks(const int layer_num, const std::string& in) {
  std::string op_out = in;
  // layer number -> building block number array(residual block 2, 3, 4, 5)
  HashMap<int, std::vector<int>> layer_num2bb_num = {
      {50, {3, 4, 6, 3}}, {101, {3, 4, 23, 3}}, {152, {3, 8, 36, 3}},
  };

  // filter num of each residual block
  int res_block_filter_num[4][2] = {
      {64, 256}, {128, 512}, {256, 1024}, {512, 2048},
  };

  std::vector<int> bb_num = layer_num2bb_num[layer_num];
  for (int i = 0; i < 4; i++) {
    op_out = ResidualBlock(i + 2, bb_num[i], op_out, res_block_filter_num[i][0],
                           res_block_filter_num[i][1], i > 0);
  }
  return op_out;
}

void FindAndReplace(std::string& source, std::string const& find, std::string const& replace) {
  for (std::string::size_type i = 0; (i = source.find(find, i)) != std::string::npos;) {
    source.replace(i, find.length(), replace);
    i += replace.length();
  }
}

void DLNet2csv(const DLNetConf net) {
  PersistentOutStream out_stream(LocalFS(), "./resnet.csv");
  out_stream << "name,type,inputs,outputs,params\n";
  std::string str;
  for (const OperatorConf& cur_op_conf : net.op()) {
    // std::string str;
    // google::protobuf::TextFormat::PrintToString(cur_op_conf, &str);
    // LOG(INFO) << str;
    out_stream << cur_op_conf.name() << ",";
    // out_stream << std::to_string(cur_op_conf.op_type_case()) << ",";
    // out_stream << cur_op_conf.in() << ",";
    // out_stream << cur_op_conf.out() << ",";
    google::protobuf::TextFormat::PrintToString(cur_op_conf, &str);
    FindAndReplace(str, "{", "");
    FindAndReplace(str, "}", "");
    FindAndReplace(str, "\n", ",");
    out_stream << str;
    out_stream << "\n";
  }
}
void DLNet2Dot(const DLNetConf net) {
  // LogicalGraph::NewSingleton(net);
  // LogicalGraph::DeleteSingleton();
}
void GenResNet(const int layer_num) {
  Global<JobConf1>::New();
  InitPlacementGroup();
  std::string op_out;
  op_out = Conv2DBlock(true, "conv1", "transpose/out", 64, "same", DATA_FORMAT, 7, 2, 1);
  op_out = MaxPooling2D("pool1", op_out, 3, 2, "same", DATA_FORMAT);
  op_out = ResidualBlocks(layer_num, op_out);
  op_out = AveragePooling2D("pool5", op_out, 7, 1, "valid", DATA_FORMAT);
  op_out = FullyConnected("fc1000", op_out, 1000);
  // op_out = Softmax("prob", op_out);
  PrintProtoToTextFile(Global<JobConf1>::Get()->net(), "./resnet.prototxt");
  PrintProtoToTextFile(Global<JobConf1>::Get()->placement(), "./resnet_placement.prototxt");
  Global<JobConf1>::Delete();
  // DLNet2csv(resnet);
  // DLNet2Dot(resnet);
}

}  // namespace oneflow

DEFINE_int32(layer_num, 50, "ResNet layer number:50, 101, 152");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  oneflow::GenResNet(FLAGS_layer_num);
  return 0;
}
