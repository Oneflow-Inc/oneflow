#ifndef _BLOB_NAME_CONVERTER_H_
#define _BLOB_NAME_CONVERTER_H_
#include <vector>
#include <string>
#include <cstdint>
namespace oneflow {
namespace strings {
// The following two functions define the rules for conversion between a data 
// blob's name and its corresponding diff blob's name (if it has).
std::string get_diff_blob_name(const std::string& data_blob_name);
std::string get_data_blob_name(const std::string& diff_blob_name);

// Construct a blob's full name inside a particular layer by combining the layer
// name and the variable suffix (e.g., "out", "out_diff")
std::string full_blob_name_in_layer(const std::string& layer_name,
  const std::string& var_suffix);
// Construct a blob's full name inside a particular dag by combining the layer
// name and the blob's alias suffix specified in config file (e.g., "out" in 
// prototxt)
std::string full_blob_name_in_dag(const std::string& layer_name,
  const std::string& alias_suffix);

// Compose a blob's variable name in a array
// Various names relating to a particular blob
// "conv1/0/in":  the full blob name
// "0/in":  the blob variable name
// 0: the index of the blob in the array
// "in": the blob variable name's suffix
std::string blob_variable_name_in_array(
  int32_t idx, const std::string& var_suffix);
// Whether the blob is an element in a blob array according to its variable 
// suffix (without 'data/' or 'model/' prefix). If |blob_var_name| has a "/"
// inside, return true.
bool is_array_blob(const std::string& blob_var_name);
// If a blob is an element of a blob array, get its index in the array and the 
// variable name's suffix from the blob variable name.
// The input argument |blob_var_name| does not have the 'data/' or 'model/' prefix
void parse_blob_variable_name_in_array(const std::string& blob_var_name,
  int32_t *idx, std::string *var_suffix);
// Remove the prefix "data/" or "model/" from the blob_variable name
std::string remove_prefix_of_blob_variable_name(const std::string& blob_var_name);

bool has_diff_correspondence(const std::vector<std::string>& blob_names,
  const std::vector<std::string>& diff_names);
}
}
#endif  // _BLOB_NAME_CONVERTER_H_
