#include "task/task_param.h"
#include "common/common.h"
#include "task/task_param_creator.h"

namespace oneflow {
template <typename Dtype>
TaskParam<Dtype>::TaskParam(const TaskParamCreator<Dtype>* task_param_creator)
  : task_param_creator_(task_param_creator) {
  Init();
}

template <typename Dtype>
TaskParam<Dtype>::~TaskParam() {
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
  //FixMe xiaoshu
}
INSTANTIATE_CLASS(TaskParam);
}  // namespace oneflow
