#include "context/net_descriptor.h"
#include "proto_io.h"

namespace caffe {
NetDescriptor::NetDescriptor(const caffe::NetParameter& net_param)
  : layer_num_(0) {
  net_name_ = net_param.name();
  layer_num_ = net_param.layer_size();
  for (int32_t id = 0; id < layer_num_; ++id) {
    caffe::LayerProto layer_proto = net_param.layer(id);
    LayerInfo layer_info;
    layer_info.name = layer_proto.name();
    layer_info.type = layer_proto.type();
    layer_infos_.push_back(layer_info);
    layer_protos_.push_back(layer_proto);
  }

  is_train_ = false;
  if (net_param.has_state()) {
    auto&& state = net_param.state();
    if (state.has_phase()) {
      is_train_ = true;
    }
  }
}
NetDescriptor::~NetDescriptor() {}
std::string NetDescriptor::net_name() const {
  return net_name_;
}
int32_t NetDescriptor::layer_num() const {
  return layer_num_;
}
std::string NetDescriptor::layer_name(int32_t id) const {
  return layer_infos_[id].name;
}
std::string NetDescriptor::layer_type(int32_t id) const {
  return layer_infos_[id].type;
}
bool NetDescriptor::is_train() const {
  return is_train_;
}
caffe::LayerProto NetDescriptor::layer_proto(int32_t id) const {
  return layer_protos_[id];
}
}  // namespace caffe
