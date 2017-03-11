#include <cstdint>
#include <string>
#include <vector>
#include "layers/load_partialmodel_layer.h"
#include "common/common.h"
#include "layers/layer_factory.h"
#include "memory/blob.h"
#include "context/one.h"
#include "context/config_parser.h"
#include "io/stream.h"
#include "common/str_util.h"
#include "context/solver_descriptor.h"

namespace caffe {
template <typename Dtype>
void LoadPartialModelLayer<Dtype>::InitParamFromProto() {
  CHECK(param_ == nullptr);
  auto param = new LoadPartialModelParam<Dtype>();
  LoadPartialModelProto load_partialmodel_proto;
  ParseProtoFromStringOrDie(proto_param_, &load_partialmodel_proto);
  auto n1 = load_partialmodel_proto.load_layer_names_size();
  for (int i = 0; i < n1; ++i) {
    param->load_layer_names.push_back(load_partialmodel_proto.load_layer_names(i));
  }
  auto n2 = load_partialmodel_proto.load_layer_shapes_size();
  for (int i = 0; i < n2; ++i) {
    param->load_layer_shapes.push_back(load_partialmodel_proto.load_layer_shapes(i));
  }
  CHECK(n1 == n2);
  param_ = param;
}
template <typename Dtype>
void LoadPartialModelLayer<Dtype>::InitFromInputShape(
    DataParam<Dtype>* data_param) {
  GET_CONCRETE_POINTER(LoadPartialModelData, data, data_param);
  GET_CONCRETE_POINTER(LoadPartialModelParam, param, param_);
  CHECK(param->load_size_ > 0)
    << "Please call |SetLoadSize| firstly to set load_size.";

  std::vector<int64_t> output_shape;
  int64_t load_size = 0;
  for (auto load_layer_shape : param->load_layer_shapes)
    load_size += load_layer_shape;
  output_shape.assign({ load_size, 1 });
  data->out->set_shape(output_shape);
  param_->mutable_data_param()->AlignBlobShapes(*data_param);

}
template <typename Dtype>
void LoadPartialModelLayer<Dtype>::Forward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const{
  GET_CONCRETE_POINTER(LoadPartialModelData, data, data_param);
  GET_CONCRETE_POINTER(LoadPartialModelModel, model, model_param);
  GET_CONCRETE_POINTER(LoadPartialModelParam, param, param_);
  std::vector<std::string> blob_names = param->mutable_data_param()->blob_names();
  Blob<Dtype>* out_blob = param->mutable_data_param()->GetBlob(blob_names[0]);
  auto solver_descriptor_ = caffe::TheOne<Dtype>::config_parser()->solver_descriptor();
  const std::string train_net_path = solver_descriptor_->train_net();
  if (train_net_path.empty()) {

  } else {
    // stream classes should be registered by their names.
    io::Stream* in_stream = io::Stream::CreateForRead(train_net_path, false);
    CHECK(in_stream, "Cannot open train net file: ", train_net_path);
    if (in_stream->AtEnd()) return;
    else {
      size_t offset = in_stream->Tell();
      in_stream->Seek(offset);
      auto layer_num_in_segment = param->load_layer_names.size();
      size_t prev_offset = 0;
      for (int i = 0; i < layer_num_in_segment; ++i) {
        if (in_stream->Valid()) {
          in_stream->ReadOneStep();
          BlobProto proto;
          CHECK(proto.ParseFromString(in_stream->value()));
          CHECK(out_blob->Deserialize(
            prev_offset,
            proto,
            param->load_layer_names[i],
            param->load_layer_shapes[i]));
          prev_offset += param->load_layer_shapes[i];
        }
      }
    }
  }
}

template <typename Dtype>
void LoadPartialModelLayer<Dtype>::Backward(const ContextParam& ctx,
  DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const{
  GET_CONCRETE_POINTER(LoadPartialModelData, data, data_param);
  GET_CONCRETE_POINTER(LoadPartialModelModel, model, model_param);
    // Use ctx, data and model
}
INSTANTIATE_CLASS(LoadPartialModelLayer);
REGISTER_LAYER_CLASS(LoadPartialModel);
}