#include <cstdint>
#include <vector>
#include <string>
#include "layers/net_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"
#include "math/math_util.h"
#include "net/network.h"

namespace caffe {
template <typename Dtype>
void NetLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new NetParam<Dtype>();
  NetProto net_proto;
  ParseProtoFromStringOrDie(proto_param_, &net_proto);
  CHECK(net_proto.has_in_num());
  param->in_num_ = net_proto.in_num();
  CHECK(net_proto.has_out_num());
  param->out_num_ = net_proto.out_num();
  CHECK(net_proto.has_forward_is_sender());
  param->forward_is_sender_ = net_proto.forward_is_sender();
  param_ = param;
}
template <typename Dtype>
void NetLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(NetData, data, data_param);
  GET_CONCRETE_POINTER(NetParam, param, param_);
  param->out_num_ = param->in_num_;
  int64_t envelope_count = 0;
  for (int32_t i = 0; i < param->in_num_; ++i) {
    data->out[i]->set_shape(data->in[i]->shape());
    envelope_count += data->in[i]->shape().count();
  }
  Shape envelope_shape;
  envelope_shape.Reshape({ 1, envelope_count });
  data->in_envelope->set_shape(envelope_shape);
  data->out_envelope->set_shape(envelope_shape);
}
template <typename Dtype>
void NetLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(NetData, data, data_param);
  GET_CONCRETE_POINTER(NetModel, model, model_param);
  GET_CONCRETE_POINTER(NetParam, param, param_);
  // Use ctx, data and model
  if (param->forward_is_sender_) {
    // Forward, out_net -> send {in -> out}
    for (int32_t i = 0; i < data->in.size(); ++i) {
      int64_t elem_num = data->in[i]->shape().count();
      caffe_copy(elem_num, data->in[i]->data(), data->out[i]->mutable_data());
    }
  } else {
    // Forward, in_net -> receive {in_envelope -> out_envelope}
    Network* network = GetNdspiRDMAInstance();  // Used for inter-node messaging
    // Read from in_envelope at remote machine to out_envelope at local memory
    MemoryDescriptor* remote_mem_descriptor
      = reinterpret_cast<MemoryDescriptor*>(data->in_envelope);
    NetworkMemory* local_network_memory
      = reinterpret_cast<NetworkMemory*>(data->out_envelope);
    network->Read(remote_mem_descriptor, local_network_memory);
  }
}
template <typename Dtype>
void NetLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const {
  GET_CONCRETE_POINTER(NetData, data, data_param);
  GET_CONCRETE_POINTER(NetModel, model, model_param);
  GET_CONCRETE_POINTER(NetParam, param, param_);
  // Use ctx, data and model
  if (!param->forward_is_sender_) {
    // backward, in_net -> send {out -> in}
    for (int32_t i = 0; i < data->out.size(); ++i) {
      if (!data->channel_is_enabled(i)) continue;
      int64_t elem_num = data->out[i]->shape().count();
      caffe_copy(elem_num, data->out[i]->data(), data->in[i]->mutable_data());
    }
  } else {
    // backward, out_net -> receive {out_envelope -> in_envelope}
    Network* network = GetNdspiRDMAInstance();  // Used for inter-node messaging
    // Read from out_envelope at remote machine to in_envelope at local memory
    MemoryDescriptor* remote_mem_descriptor
      = reinterpret_cast<MemoryDescriptor*>(data->out_envelope);
    NetworkMemory* local_network_memory
      = reinterpret_cast<NetworkMemory*>(data->in_envelope);
    network->Read(remote_mem_descriptor, local_network_memory);
  }
}
INSTANTIATE_LAYER_FUNCS(NetLayer);
INSTANTIATE_CLASS(NetLayer);
REGISTER_LAYER_CLASS(Net);
}  // namespace caffe
