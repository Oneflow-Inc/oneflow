#ifndef _LAYER_LAYER_UTIL_H_
#define _LAYER_LAYER_UTIL_H_
#include "common/blob_name_converter.h"

#define REGISTER_BLOB(layer_name, prefix, blob_variable)                       \
  CHECK(!layer_name.empty());                                                  \
  std::string blob_##blob_variable                                             \
    = strings::full_blob_name_in_layer(layer_name, prefix + #blob_variable);   \
  CHECK_EQ(this->name_to_blob_pptr_.count(blob_##blob_variable), 0);           \
  CHECK_EQ(this->name_to_blob_pptr_.insert(                                    \
                {blob_##blob_variable, &blob_variable}).second, true);         \
  CHECK_EQ(this->name_to_blob_ptr_.count(blob_##blob_variable), 0);            \
  CHECK_EQ(this->name_to_blob_ptr_.insert(                                     \
                {blob_##blob_variable, blob_variable}).second, true);          \
  this->blob_names_.push_back(blob_##blob_variable)

#define REGISTER_ARRAY_BLOB(layer_name, prefix, variable_suffix, index)           \
  CHECK(!layer_name.empty());                                                     \
  std::string variable_##variable_suffix                                          \
    = prefix + strings::blob_variable_name_in_array(index, #variable_suffix);     \
  std::string full_##variable_suffix                                              \
    = strings::full_blob_name_in_layer(layer_name, variable_##variable_suffix);   \
  CHECK_EQ(this->name_to_blob_pptr_.count(full_##variable_suffix), 0);            \
  CHECK_EQ(this->name_to_blob_pptr_.insert(                                       \
                {full_##variable_suffix, &variable_suffix[index]}).second, true); \
  CHECK_EQ(this->name_to_blob_ptr_.count(full_##variable_suffix), 0);             \
  CHECK_EQ(this->name_to_blob_ptr_.insert(                                        \
                {full_##variable_suffix, variable_suffix[index]}).second, true);  \
  blob_names_.push_back(full_##variable_suffix)

#define MODEL_REGISTER_BLOB(layer_name, blob_variable, blob_type)                        \
  std::string prefix_of_##blob_variable = "model/";                                      \
  REGISTER_BLOB(layer_name, prefix_of_##blob_variable, blob_variable);                   \
  std::string prefix_##blob_variable = prefix_of_##blob_variable + #blob_variable;       \
  if (blob_type == BlobType::kModel) { model_vars_.push_back(prefix_##blob_variable); }  \
  if (blob_type == BlobType::kTemp) { temp_vars_.push_back(prefix_##blob_variable); }

#define MODEL_REGISTER_ARRAY_BLOB(layer_name, blob_variable, index, blob_type)                        \
  {                                                                                                   \
  std::string prefix_of_##blob_variable_##index = "model/";                                           \
  REGISTER_ARRAY_BLOB(layer_name, prefix_of_##blob_variable_##index, blob_variable, index);           \
  std::string prefix_##blob_variable_##index                                                          \
   = prefix_of_##blob_variable_##index + strings::blob_variable_name_in_array(index, #blob_variable); \
  if (blob_type == BlobType::kModel) {                                                                \
    model_vars_.push_back(prefix_##blob_variable_##index);                                            \
  }                                                                                                   \
  if (blob_type == BlobType::kTemp) {                                                                 \
    temp_vars_.push_back(prefix_##blob_variable_##index);                                             \
  }                                                                                                   \
  }

// DataParam needs to 
#define DATA_REGISTER_BLOB(layer_name, blob_variable, blob_type)                                \
  std::string prefix_of_##blob_variable = "data/";                                              \
  REGISTER_BLOB(layer_name, prefix_of_##blob_variable, blob_variable);                          \
  std::string prefix_##blob_variable = prefix_of_##blob_variable + #blob_variable;              \
  if (blob_type == BlobType::kInput) { this->input_vars_.push_back(prefix_##blob_variable); }         \
  if (blob_type == BlobType::kOutput) { this->output_vars_.push_back(prefix_##blob_variable); }       \
  if (blob_type == BlobType::kOther) { this->other_vars_.push_back(prefix_##blob_variable); }         \
  if (blob_type == BlobType::kInDiff) { this->input_diffs_.push_back(prefix_##blob_variable); }       \
  if (blob_type == BlobType::kOutDiff) { this->output_diffs_.push_back(prefix_##blob_variable); }

#define DATA_REGISTER_ARRAY_BLOB(layer_name, blob_variable, index, blob_type)                               \
  {                                                                                                         \
  std::string prefix_of_##blob_variable_##index = "data/";                                                  \
  REGISTER_ARRAY_BLOB(layer_name, prefix_of_##blob_variable_##index, blob_variable, index);                 \
  std::string prefix_##blob_variable_##index                                                                \
    = prefix_of_##blob_variable_##index + strings::blob_variable_name_in_array(index, #blob_variable);      \
  if (blob_type == BlobType::kInput) {                                                                      \
    input_vars_.push_back(prefix_##blob_variable_##index);                                                  \
  }                                                                                                         \
  if (blob_type == BlobType::kOutput) {                                                                     \
    output_vars_.push_back(prefix_##blob_variable_##index);                                                 \
  }                                                                                                         \
  if (blob_type == BlobType::kOther) {                                                                      \
    other_vars_.push_back(prefix_##blob_variable_##index);                                                  \
  }                                                                                                         \
  if (blob_type == BlobType::kInDiff) {                                                                     \
    input_diffs_.push_back(prefix_##blob_variable_##index);                                                 \
  }                                                                                                         \
  if (blob_type == BlobType::kOutDiff) {                                                                    \
    output_diffs_.push_back(prefix_##blob_variable_##index);                                                \
  }                                                                                                         \
  }

#define GET_CONCRETE_POINTER(sub_type, sub_name, base_name)                    \
  sub_type<Dtype>* sub_name = dynamic_cast<sub_type<Dtype>*> (base_name);      \
  CHECK_NOTNULL(sub_name)
#endif  // _LAYER_LAYER_UTIL_H_
