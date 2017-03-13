#ifndef _LAYER_LOAD_PARTIALMODEL_LAYER_H_
#define _LAYER_LOAD_PARTIALMODEL_LAYER_H_

#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class LoadPartialModelData : public DataParam<Dtype> {
 public:
   Blob<Dtype>* out{ nullptr };
   explicit LoadPartialModelData(const std::string layer_name) {
     DATA_REGISTER_BLOB(layer_name, out, BlobType::kOutput);
   }
};

template <typename Dtype>
class LoadPartialModelModel : public ModelParam<Dtype> {
 public:
   explicit LoadPartialModelModel(const std::string layer_name) {}
};

template <typename Dtype>
class LoadPartialModelParam : public LayerParam<Dtype> {
 public:
   int32_t load_size_{ 0 };
   std::vector<std::string> load_layer_names;
   std::vector<int64_t> load_layer_shapes;
   LoadPartialModelParam() {}
};

template <typename Dtype>
class LoadPartialModelLayer : public BaseLayer<Dtype> {
 public:
   explicit LoadPartialModelLayer(const std::string& layer_name,
     const std::string& proto_param) : BaseLayer(layer_name, proto_param) {}

   DataParam<Dtype>* CreateDataParam() const override {
     return new LoadPartialModelData<Dtype>(layer_name_);
   }

   ModelParam<Dtype>* CreateModelParam() const override {
     return new LoadPartialModelModel<Dtype>(layer_name_);
   }

   void InitParamFromProto() override;
   void InitFromInputShape(DataParam<Dtype>* data_param) override;
   void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
     ModelParam<Dtype>* model_param) const override;
   void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
     ModelParam<Dtype>* model_param) const override;
 private:
   //std::map<std::string, int> blob_names_indices_;
   LoadPartialModelLayer(const LoadPartialModelLayer& other) = delete;
   LoadPartialModelLayer&
     operator= (const LoadPartialModelLayer& other) = delete;
  };
}  // namespace caffe
#endif