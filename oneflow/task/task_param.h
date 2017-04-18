#ifndef _TASK_PARAM_H_
#define _TASK_PARAM_H_
#include <cstdint>
#include <vector>

namespace oneflow {
template <typename Dtype>
class DataParam;

template <typename Dtype>
class ModelParam;

template <typename Dtype>
class Blob;

template <typename Dtype>
class TaskParamCreator;

template <typename Dtype>
class TaskParam {
 public:
   explicit TaskParam(const TaskParamCreator<Dtype>* task_param_creator);
   ~TaskParam();

  void set_blob(int32_t index, Blob<Dtype>* blob);
  const std::vector<DataParam<Dtype>*>& data_params() const;
  const std::vector<ModelParam<Dtype>*>& model_params() const;

 private:
   const TaskParamCreator<Dtype>* task_param_creator_;
   std::vector<DataParam<Dtype>*> data_params_;
   std::vector<ModelParam<Dtype>*> model_params_;
   std::vector<Blob<Dtype>**> blob_pptrs_;

   void Init();
};
}  // namespace oneflow
#endif  // _TASK_PARAM_H_
