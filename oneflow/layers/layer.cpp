#include "layers/layer.h"
#include "common/common.h"

namespace caffe {
template <typename Dtype>
void Layer<Dtype>::Setup(
  const std::vector<const std::shared_ptr<BlobMeta>>& inputs,
  std::vector<std::shared_ptr<BlobMeta>>* outputs) {
  SetParameterNames();
  AddShapes();
  LayerSetup(inputs, outputs);
  Reshape(inputs, outputs);
}

template <typename Dtype>
void Layer<Dtype>::AddShapes() {
  for (auto& it : parameter_names_) {
    shape_dict_.insert({ it, new Shape });
  }
}

template <typename Dtype>
void Layer<Dtype>::AllocateModel(void* data_ptr,
  const std::unique_ptr<DeviceRegister<Dtype>>& device_register,
  std::string name, int64_t global_id) {
  for (auto& it : shape_dict_) {
    device_register->add_blob(
      it.first + name,
      data_ptr,
      get_shape(it.first),
      MemoryType::kDeviceMemory,
      global_id);
    data_ptr = static_cast<char*>(data_ptr)
      +it.second->count() * sizeof(Dtype);
  }
}

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
