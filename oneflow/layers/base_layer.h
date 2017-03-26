#ifndef _LAYER_LAYER2_H_
#define _LAYER_LAYER2_H_
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <glog/logging.h>

#include "common/shape.h"
#include "common/str_util.h"
#include "memory/blob.h"
#include "layers/layer_util.h"
#include "device/device_alternate.h"
#include "common/cudnn_utils.h"
namespace oneflow {

struct ContextParam {
  cublasHandle_t cublas_handle;
  cudaStream_t cuda_stream;
  cudnnHandle_t cudnn_handle;
};

template <typename Dtype>
class DataModelBaseParam {
public:
  DataModelBaseParam() {}
  virtual ~DataModelBaseParam() {
    if (own_blob_) {
      for (auto& pair : name_to_blob_ptr_) {
        if (pair.second) {
          delete pair.second;
          pair.second = nullptr;
        }
      }
    }
  }
  // Get all the <name, blob> pair in an DataModelBaseParam object
  const std::unordered_map<std::string, Blob<Dtype>**>& name_to_blob_pptr()
    const {
    return name_to_blob_pptr_;
  }

  std::vector<std::string> blob_names() const {
    return blob_names_;
  }
  // Copy the blob shapes from another DataModelBaseParam object.
  // Ensure both the blobs in |other| and |this| object exist.
  void AlignBlobShapes(const DataModelBaseParam<Dtype>& other) {
    for (auto& pair : other.name_to_blob_ptr_) {
      auto& name = pair.first;
      Blob<Dtype>* other_blob_ptr = pair.second;
      CHECK_NOTNULL(other_blob_ptr);
      CHECK(name_to_blob_ptr_.count(name) > 0);
      Blob<Dtype>* this_blob_ptr = name_to_blob_ptr_[name];
      CHECK_NOTNULL(this_blob_ptr);
      this_blob_ptr->set_shape(other_blob_ptr->shape());
    }
  }
  // Allocate empty blobs for all the blobs in |name_to_blob_|.
  // Please call this function only when all the blobs have been registered
  // either by REGISTER_BLOB or REGISTER_ARRAY_BLOB. In other words,
  // never call this function in constructor: since we call REGISTER_BLOB in
  // constructor to register blobs into |name_to_blob_|.
  // For some particular layers, e.g., SplitLayer or ConcatLayer, besides
  // REGISTER_BLOB in constructor, we also call REGISTER_ARRAY_BLOB in
  // |SetInputNum| or |SetOutputNum| to register array blobs in |name_to_blob_|.
  void AllocateEmptyBlobs() {
    for (auto& pair : name_to_blob_pptr_) {
      Blob<Dtype>** blob_address_ptr = pair.second;
      if (*blob_address_ptr != nullptr) {
        delete *blob_address_ptr;
        *blob_address_ptr = nullptr;
        name_to_blob_ptr_[pair.first] = nullptr;
      }
      *blob_address_ptr = new Blob<Dtype>();
      name_to_blob_ptr_[pair.first] = *blob_address_ptr;
    }
    own_blob_ = true;
  }

  bool CheckBlobNull(const std::string& blob_name) {
    CHECK(name_to_blob_pptr_.count(blob_name) > 0);
    Blob<Dtype>** old_blob_address_ptr = name_to_blob_pptr_[blob_name];
    return *old_blob_address_ptr == nullptr;
  }
  bool HasBlob(const std::string& blob_name) {
    return name_to_blob_pptr_.count(blob_name) > 0;
  }
  Blob<Dtype>* GetBlob(const std::string& blob_name) {
    CHECK(name_to_blob_pptr_.count(blob_name) > 0);
    Blob<Dtype>** old_blob_address_ptr = name_to_blob_pptr_[blob_name];
    return *old_blob_address_ptr;
  }

  void SetBlob(const std::string& blob_name, Blob<Dtype>* blob_ptr) {
    // TODO(jiyuan): handle the blob in array,
    // if the blob name is "layer_name/data/1/in"

    CHECK_NOTNULL(blob_ptr);
    CHECK(name_to_blob_pptr_.count(blob_name) > 0);
    Blob<Dtype>** old_blob_address_ptr = name_to_blob_pptr_[blob_name];
    // Release old blob if it is non-null
    if (*old_blob_address_ptr != nullptr && own_blob_) {
      // NOTE(jiyuan): if own_blob_ is false, we will not release the old blob
      delete *old_blob_address_ptr;
      *old_blob_address_ptr = nullptr;
      name_to_blob_ptr_[blob_name] = nullptr;
    }
    *old_blob_address_ptr = blob_ptr;
    name_to_blob_ptr_[blob_name] = blob_ptr;
  }

  void SetShape(const std::string& blob_name, const Shape& shape) {
    CHECK(name_to_blob_pptr_.count(blob_name) > 0);
    Blob<Dtype>** blob_address_ptr = name_to_blob_pptr_[blob_name];
    Blob<Dtype>* blob_ptr = *blob_address_ptr;
    CHECK_NOTNULL(blob_ptr);
    blob_ptr->set_shape(shape);
  }

  Shape GetShape(const std::string& blob_name) const {
    auto name_to_blob_it = name_to_blob_pptr_.find(blob_name);
    CHECK(name_to_blob_it != name_to_blob_pptr_.end());
    Blob<Dtype>** blob_address_ptr = name_to_blob_it->second;
    Blob<Dtype>* blob_ptr = *blob_address_ptr;
    CHECK_NOTNULL(blob_ptr);
    return blob_ptr->shape();
  }

  bool HasBlob(const std::string& blob_name) const {
    return name_to_blob_pptr_.count(blob_name) > 0;
  }
protected:
  // |name_to_blob_pptr_| is used to control where the |Blob<Dtype>*| points to
  std::unordered_map<std::string, Blob<Dtype>**> name_to_blob_pptr_;
  // |name_to_blob_ptr_| is used in destructor if necessary. We use both ptr
  // and pptr to work around the case described below.
  // int** in = new int*[3](); // new int*[3]();  // in[0] == in[1] == in[2] == 0x0000000000000000
  // in[0] = new int(1);        // in[0] = 0x000000fc150f0038{1}
  // in[1] = new int(2);        // in[1] = 0x000000fc150f0028{2}
  // in[2] = new int(3);        // in[2] = 0x000000fc150f0018{3}
  // int *tmp = in[0];
  // std::unordered_map<std::string, int**> dict;
  // dict.insert({ "0", &in[0] });  // {"0",0x000000fc15102640{0x000000fc150f0038{1}}}
  // dict.insert({ "1", &in[1] });  // {"1",0x000000fc15102648{0x000000fc150f0028{2}}}
  // dict.insert({ "2", &in[2] });  // {"2",0x000000fc15102650{0x000000fc150f0018{3}}}
  // delete[]in;                    // dict["0"] -> {"0",0x000000fc15102640{0x0000000000000000{?}}}
  //                                // called in destructor of derived class
  // // std::cout << **(dict["0"]) << std::endl;
  // std::cout << **(dict["1"]) << std::endl;
  // std::cout << **(dict["2"]) << std::endl;
  std::unordered_map<std::string, Blob<Dtype>*> name_to_blob_ptr_;
  // In most cases, the blobs are allocated outside DataModelBaseParam, i.e., the
  // object does not own the memories of the blobs, thus no need to release the
  // resources in destructor. In rare cases, we need to call |AllocateEmptyBlobs|,
  // therefore, the object owns the empty blobs and need to release them in the
  // destructor. In summary, the default value of |own_blob_| is false, it
  // becomes true only when users call |AllocateEmptyBlobs|.
  bool own_blob_{ false };
  std::vector<std::string> blob_names_;
private:
  DataModelBaseParam(const DataModelBaseParam& other) = delete;
  DataModelBaseParam& operator=(const DataModelBaseParam& other) = delete;
};

/*
NOTE(jiyuan): DataParam and ModelParam are very similar.
They can use the same class. However, we explicitly distinguish them with
DataParam and ModelParam classes to force the interfaces to do type-checking:
  BaseLayer.Forward(const ContextParam&, DataParam*, ModelParam*)
  BaseLayer.Backward(const ContextParam&, DataParam*, ModelParam*)
To reuse their common implementations, we let both DataParam and ModelParam
inherit from the DataModelBaseParam.
*/
// NOTE(Chonglin): kModel for blobs storing real layer parameters
// kTemp for blobs as temp buffer in layer
enum class BlobType {
  kInput = 0,
  kInDiff,
  kOutput,
  kOutDiff,
  kOther,

  kModel,
  kModelDiff,
  kTemp
};
/*
COMMENT(jiyuan):
1, Should we add a diff blob for each kInput and kOutput blob by default?
Otherwise, we need to take care of diff blob during registration, setup, creation.
2, Is it true that each kInput and kOutput blob needs a diff correspondence?
3, In inference-only job, the diff blobs are obviously unnecessary.
4, Does the boxing, copy-like layer need diff blob?
*/
// Data related parameters
template <typename Dtype>
class DataParam : public DataModelBaseParam<Dtype> {
public:
  DataParam() {}

  // NOTE: not the full name of inputs/outputs, but only the suffixes.
  std::vector<std::string> GetInputVars() const { return input_vars_; }
  std::vector<std::string> GetInputDiffs() const { return input_diffs_; }
  std::vector<std::string> GetOutputVars() const { return output_vars_; }
  std::vector<std::string> GetOutputDiffs() const { return output_diffs_; }
  std::vector<std::string> GetOtherVars() const { return other_vars_; }
  bool channel_is_enabled(int32_t i) const { return enable_channel[i]; }

protected:
  // |inputs_| and |outputs_| contain the variable names, they are solely used
  // for DataParam, but no for ModelParam
  std::vector<std::string> input_vars_;
  std::vector<std::string> input_diffs_;
  std::vector<std::string> output_vars_;
  std::vector<std::string> output_diffs_;
  std::vector<std::string> other_vars_;
  std::vector<bool> enable_channel;  // Used only in backward direction
};
// Model related parameters
template <typename Dtype>
class ModelParam : public DataModelBaseParam<Dtype> {
public:
  ModelParam() {}

  std::vector<std::string> GetModelVars() const { return model_vars_; }
  std::vector<std::string> GetModelDiffs() const { return model_diffs_; }
  std::vector<std::string> GetTempVars() const { return temp_vars_; }

protected:
  std::vector<std::string> model_vars_;
  std::vector<std::string> model_diffs_;
  std::vector<std::string> temp_vars_;
};

template <typename Dtype>
class BaseLayer;

// Independent parameter
template <typename Dtype>
class LayerParam {
protected:
  DataParam<Dtype>* prototype_data_{ nullptr };
  ModelParam<Dtype>* prototype_model_{ nullptr };
  //const BaseLayer<Dtype>& layer_;

public:
  explicit LayerParam() { }
  virtual ~LayerParam() {
    // Deleting a nullptr is safe
    delete prototype_data_;
    delete prototype_model_;
  }
  void CreateDataAndModelParam(const BaseLayer<Dtype>& layer) {
    CHECK(prototype_data_ == nullptr);
    prototype_data_ = layer.CreateDataParam();
    CHECK(prototype_model_ == nullptr);
    prototype_model_ = layer.CreateModelParam();
  }
  void AllocateEmptyBlobs() {
    CHECK_NOTNULL(prototype_data_);
    prototype_data_->AllocateEmptyBlobs();
    CHECK_NOTNULL(prototype_model_);
    prototype_model_->AllocateEmptyBlobs();
  }

  const DataParam<Dtype>* data_param() const {
    CHECK_NOTNULL(prototype_data_);
    return prototype_data_;
  }
  const ModelParam<Dtype>* model_param() const {
    CHECK_NOTNULL(prototype_model_);
    return prototype_model_;
  }
  DataParam<Dtype>* mutable_data_param() {
    CHECK_NOTNULL(prototype_data_);
    return prototype_data_;
  }
  ModelParam<Dtype>* mutable_model_param() {
    CHECK_NOTNULL(prototype_model_);
    return prototype_model_;
  }
  std::vector<std::string> GetInputVars() const {
    return prototype_data_->GetInputVars();
  }
  std::vector<std::string> GetInputDiffs() const {
    return prototype_data_->GetInputDiffs();
  }
  std::vector<std::string> GetOutputVars() const {
    return prototype_data_->GetOutputVars();
  }
  std::vector<std::string> GetOutputDiffs() const {
    return prototype_data_->GetOutputDiffs();
  }
  std::vector<std::string> GetOtherVars() const {
    return prototype_data_->GetOtherVars();
  }
  std::vector<std::string> GetModelVars() const {
    return prototype_model_->GetModelVars();
  }
  std::vector<std::string> GetModelDiffs() const {
    return prototype_model_->GetModelDiffs();
  }
  std::vector<std::string> GetTempVars() const {
    return prototype_model_->GetTempVars();
  }

  std::vector<std::string> blob_names() const {
    std::vector<std::string> blob_names = prototype_data_->blob_names();
    auto model_blob_names = prototype_model_->blob_names();
    for (auto& model_blob_name : model_blob_names) {
      blob_names.push_back(model_blob_name);
    }
    return blob_names;
  }

  Shape GetBlobShape(const std::string& blob_name) const {
    if (prototype_data_->HasBlob(blob_name)) {
      return prototype_data_->GetShape(blob_name);
    } else if (prototype_model_->HasBlob(blob_name)) {
      return prototype_model_->GetShape(blob_name);
    } else {
      LOG(FATAL) << "The blob can not be found";
    }
  }
};

// Follow the below steps to initialize and use a layer (user-customized
// operator).
template <typename Dtype>
class BaseLayer {
public:
  // Step 1:
  // Pass the layer name and proto string to constructor.
  BaseLayer(const std::string& layer_name, const std::string& proto_param)
    : layer_name_(layer_name), proto_param_(proto_param) {}
  virtual ~BaseLayer() { delete param_; }

  // Step 2:
  // The initialization from proto string is type-specific, so please fill the
  // details in the derived layer type.
  virtual void InitParamFromProto() = 0;

  // Step 3:
  // Complete the initialization that could be done when we have proto string.
  void InitParam() {
    // Init members in |param_| depending on the |proto_param_|;
    InitParamFromProto();
    // Create the prototyped DataParam and ModelParam;
    param_->CreateDataAndModelParam(*this);
    // Allocate empty blobs in DataParam and ModelParam; Note that we don't need
    // the blobs but need the shapes of blobs, so all the blobs are empty.
    param_->AllocateEmptyBlobs();
  }


  // Step 4:
  // Do the initialization if we know input data's shape (in |data_param|).
  // It must infer the output data's shape (in data_param) and the internal
  // model's shape (in model_param) and internal data's shape.
  virtual void InitFromInputShape(DataParam<Dtype>* data_param) = 0;

  // Step 5:
  // Fill the forward and backward computation.
  virtual void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const = 0;
  virtual void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const = 0;

  // The following functions provide user the capability of dynamically creating
  // appropriate layer-type-specific DataParam and ModelParam.
  // NOTE: Before call |CreateDataParam| or |CreateModelParam|, you must call
  // |InitParamFromProto| or |InitParam|.
  virtual DataParam<Dtype>* CreateDataParam() const = 0;
  virtual ModelParam<Dtype>* CreateModelParam() const = 0;

  const std::string& layer_name() const { return layer_name_; }
  bool IsElemWise() const { return is_elem_wise_; }

  // Useful only for layers who may be model-parallelized, please override this
  // function if necessary. Call it between calling InitFromProto and calling
  // InitFromInputShape(). See example in innerproduct_layer.h
  virtual void SetModelParallelismSize(int32_t split_num, int32_t index) {}

  Shape GetBlobShape(const std::string& blob_name) const {
    return param_->GetBlobShape(blob_name);
  }

  const DataParam<Dtype>* GetDataParam() const {
    return param_->data_param();
  }
  const ModelParam<Dtype>* GetModelParam() const {
    return param_->model_param();
  }
  std::vector<std::string> GetInputVars() const {
    return param_->GetInputVars();
  }
  std::vector<std::string> GetInputDiffs() const {
    return param_->GetInputDiffs();
  }
  std::vector<std::string> GetOutputVars() const {
    return param_->GetOutputVars();
  }
  std::vector<std::string> GetOutputDiffs() const {
    return param_->GetOutputDiffs();
  }
  std::vector<std::string> GetOtherVars() const {
    return param_->GetOtherVars();
  }
  std::vector<std::string> GetModelVars() const {
    return param_->GetModelVars();
  }
  std::vector<std::string> GetModelDiffs() const {
    return param_->GetModelDiffs();
  }
  std::vector<std::string> GetTempVars() const {
    return param_->GetTempVars();
  }
  std::vector<std::string> blob_names() const {
    return param_->blob_names();
  }

  bool has_BP() const {
    auto input_vars = param_->GetInputVars();
    auto input_diffs = param_->GetInputDiffs();
    auto output_vars = param_->GetOutputVars();
    auto output_diffs = param_->GetOutputDiffs();
    bool input_has_diff
      = strings::has_diff_correspondence(input_vars, input_diffs);
    bool output_has_diff
      = strings::has_diff_correspondence(output_vars, output_diffs);
    if (input_has_diff && output_has_diff) {
      // General layer, such as convolution layer
      return true;
    }
    if (input_has_diff && !output_has_diff) {
      // Loss layer
      return true;
    }
    // Data provider layer
    CHECK(!input_has_diff && !output_has_diff);
    return false;
  }
protected:
  // Every layer has a member to store all the required parameters in forward
  // and backward computation. The most important parameters you can find include
  // the shapes of input/output data, and the shapes of model's weights.
  LayerParam<Dtype>* param_{ nullptr };
  // Memorize the layer's name. The layer's name also acts as the prefix of the
  // blob names inside this layer.
  std::string layer_name_;
  // Memorize the raw content of layer proto in a string
  std::string proto_param_;
  // The model-parallelism of element-wise operation needs special treatment.
  // Set true if the layer is element-wise operation.
  bool is_elem_wise_{ false };

private:
  BaseLayer(const BaseLayer& other) = delete;
  BaseLayer& operator=(const BaseLayer& other) = delete;
};
}
#endif  // _LAYER_LAYER2_H_
