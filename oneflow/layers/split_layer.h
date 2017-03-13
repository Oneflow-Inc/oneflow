#ifndef _LAYERS_SPLIT_LAYER_H_
#define _LAYERS_SPLIT_LAYER_H_
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class SplitData : public DataParam<Dtype> {
public:
  Blob<Dtype>* in{ nullptr };
  Blob<Dtype>* in_diff{ nullptr };
  std::vector<Blob<Dtype>*> out;
  std::vector<Blob<Dtype>*> out_diff;
  explicit SplitData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, in, BlobType::kInput);
    DATA_REGISTER_BLOB(layer_name, in_diff, BlobType::kInDiff);
  }
  // You can get the outpub blob with the name "0/out", "1/out" ...
  void SetOutputNum(const std::string& layer_name, int32_t out_num) {
    out.resize(out_num, nullptr);
    out_diff.resize(out_num, nullptr);
    for (int32_t idx = 0; idx < out_num; ++idx) {
      DATA_REGISTER_ARRAY_BLOB(layer_name, out, idx, BlobType::kOutput);
      DATA_REGISTER_ARRAY_BLOB(layer_name, out_diff, idx, BlobType::kOutDiff);
    }
  }
};

template <typename Dtype>
class SplitModel : public ModelParam<Dtype> {
public:
  explicit SplitModel(const std::string& layer_name) {}
};

template <typename Dtype>
class SplitParam : public LayerParam<Dtype> {
public:
  // Init from proto
  int32_t out_num_;
  explicit SplitParam() {}

  void SetOutputNum(const std::string& layer_name, int32_t out_num) {
    out_num_ = out_num;
    CHECK_GT(out_num_, 1);
    GET_CONCRETE_POINTER(SplitData, prototype_data, prototype_data_);
    prototype_data->SetOutputNum(layer_name, out_num_);
  }
};

template <typename Dtype>
class SplitLayer : public BaseLayer<Dtype> {
public:
  /*
  proto_param should be a string in the following format:
  "out_num: 2"
  which indicats there are 2 output blobs.
  */
  explicit SplitLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {}

  DataParam<Dtype>* CreateDataParam() const override {
    SplitData<Dtype>* data = new SplitData<Dtype>(layer_name_);
    GET_CONCRETE_POINTER(SplitParam, param, param_);
    CHECK_GT(param->out_num_, 1);
    data->SetOutputNum(layer_name_, param->out_num_);
    return data;
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new SplitModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  SplitLayer(const SplitLayer& other) = delete;
  SplitLayer& operator=(const SplitLayer& other) = delete;
};

}  // namespace caffe
#endif  // _LAYERS_SPLIT_LAYER_H_
