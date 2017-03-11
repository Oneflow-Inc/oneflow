#include "task/task_param.h"
#include "common/common.h"
#include "dag/node_meta.h"
#include "dag/task_dag.h"
#include "dag/dag_iterator.h"
#include "task/task_param_creator.h"

namespace caffe {
template <typename Dtype>
TaskParam<Dtype>::TaskParam(const TaskParamCreator<Dtype>* task_param_creator)
  : task_param_creator_(task_param_creator) {
  Init();
}

template <typename Dtype>
TaskParam<Dtype>::~TaskParam() {
  for (auto& param : data_params_) {
    delete param;
  }
  for (auto& param : model_params_) {
    delete param;
  }
}

template <typename Dtype>
void TaskParam<Dtype>::set_blob(int32_t index, Blob<Dtype>* blob) {
  *(blob_pptrs_[index]) = blob;
}

template <typename Dtype>
const std::vector<DataParam<Dtype>*>& TaskParam<Dtype>::data_params() const {
  return data_params_;
}

template <typename Dtype>
const std::vector<ModelParam<Dtype>*>& TaskParam<Dtype>::model_params() const {
  return model_params_;
}

template <typename Dtype>
void TaskParam<Dtype>::Init() {
  // Firstly, create data_params_ and model_params_
  auto& ordered_layers = task_param_creator_->ordered_layers();
  int32_t layer_num = ordered_layers.size();

  for (auto& layer : ordered_layers) {
    data_params_.push_back(layer->CreateDataParam());
    model_params_.push_back(layer->CreateModelParam());
  }

  // Secondly, create and initialize blob_pptrs_
  auto& layer_blobs_in_execution
    = task_param_creator_->layer_blobs_in_execution();
  int32_t layer_blob_num = layer_blobs_in_execution.size();
  blob_pptrs_.resize(layer_blob_num);
  int32_t blob_count = 0;
  for (int32_t i = 0; i < layer_num; ++i) {
    auto& data_name_to_blob_pptrs = data_params_[i]->name_to_blob_pptr();
    for (auto& name_to_blob_pptr : data_name_to_blob_pptrs) {
      auto& layer_blob = name_to_blob_pptr.first;
      auto index = task_param_creator_->index_of_layer_blob(layer_blob);
      CHECK(index < layer_blob_num);
      blob_pptrs_[index] = name_to_blob_pptr.second;
      ++blob_count;
    }
    auto& model_name_to_blob_pptrs = model_params_[i]->name_to_blob_pptr();
    for (auto& name_to_blob_pptr : model_name_to_blob_pptrs) {
      auto& layer_blob = name_to_blob_pptr.first;
      auto index = task_param_creator_->index_of_layer_blob(layer_blob);
      CHECK(index < layer_blob_num);
      blob_pptrs_[index] = name_to_blob_pptr.second;
      ++blob_count;
    }
  }
  CHECK(blob_count == layer_blob_num);
}
INSTANTIATE_CLASS(TaskParam);
}  // namespace caffe
