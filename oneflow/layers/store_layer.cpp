#include <cstdint>
#include <vector>
#include <string>
#include "layers/store_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"
#include "math/math_util.h"
#include "common/split_util.h"
#include "context/one.h"
#include "io/stream.h"
#include "memory/blob.h"
#include "context/config_parser.h"
#include "common/str_util.h"
#include "context/solver_descriptor.h"

namespace caffe {
template <typename Dtype>
void StoreLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new StoreParam<Dtype>();

  StoreProto store_proto;
  ParseProtoFromStringOrDie(proto_param_, &store_proto);

  param->stop_ = store_proto.stop();
  auto n1 = store_proto.store_layer_names_size();
  for (int i = 0; i < n1; ++i) {
    param->store_layer_names.push_back(store_proto.store_layer_names(i));
  }
  auto n2 = store_proto.store_layer_shapes_size();
  for (int i = 0; i < n2; ++i) {
    param->store_layer_shapes.push_back(store_proto.store_layer_shapes(i));
  }
  CHECK(n1 == n2);
  auto n3 = store_proto.layer_seek_pos_size();
  for (int i = 0; i < n3; ++i) {
    param->layer_seek_pos.push_back(store_proto.layer_seek_pos(i));
  }
  CHECK(n1 == n3);
  param_ = param;
}

template <typename Dtype>
void StoreLayer<Dtype>::InitFromInputShape(
  DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(StoreData, data, data_param);
  GET_CONCRETE_POINTER(StoreParam, param, param_);
  auto model_param = param->mutable_model_param();
  // Set the blobs' shape in param_
  std::vector<int64_t> input_shape;
  int64_t store_size = 0;
  for (auto store_layer_shape : param->store_layer_shapes)
    store_size += store_layer_shape;
  input_shape.assign({ store_size, 1 });
  data->in->set_shape(input_shape);
  param_->mutable_data_param()->AlignBlobShapes(*data_param);
}

template <typename Dtype>
void StoreLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const{
  GET_CONCRETE_POINTER(StoreData, data, data_param);
  GET_CONCRETE_POINTER(StoreModel, model, model_param);
  GET_CONCRETE_POINTER(StoreParam, param, param_);

  auto solver_descriptor_ = caffe::TheOne<Dtype>::config_parser()->solver_descriptor();
  std::string train_net_path = solver_descriptor_->train_net();
  io::Stream* in_stream = io::FileSystem::
    GetInstance(train_net_path)->OpenForWrite(train_net_path, true);

  std::vector<std::string> blob_names = param->mutable_data_param()->blob_names();
  Blob<Dtype>* store_blob 
    = param->mutable_data_param()->GetBlob(blob_names[0]);

  auto acceptor = [&](
    const std::string& blobName, const std::string& data, const int64_t pos) {
    in_stream->Seek(pos);
    in_stream->Put(blobName, data);
    if (param->stop_) in_stream->Done();
  };
  const std::vector<std::string>& store_layer_names
    = param->store_layer_names;
  const std::vector<int64_t>& store_layer_shapes
    = param->store_layer_shapes;
  const std::vector<int64_t>& layer_seek_pos
    = param->layer_seek_pos;
  reinterpret_cast<const Blob<Dtype>*>(store_blob)->Serialize(
    acceptor, store_layer_names, store_layer_shapes, layer_seek_pos);

}

template <typename Dtype>
void StoreLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const{
  GET_CONCRETE_POINTER(StoreData, data, data_param);
  GET_CONCRETE_POINTER(StoreModel, model, model_param);
  GET_CONCRETE_POINTER(StoreParam, param, param_);
}
INSTANTIATE_CLASS(StoreLayer);
REGISTER_LAYER_CLASS(Store);
}  // namespace caffe
