#include "common/blob_name_converter.h"
#include <string>
#include <algorithm>
#include <glog/logging.h>
#include "common/str_util.h"

namespace caffe {
namespace strings {
const std::string kSuffixOfDiff = "_diff";
std::string get_diff_blob_name(const std::string& data_blob_name) {
  CHECK(!EndsWith(data_blob_name, kSuffixOfDiff));
  return data_blob_name + kSuffixOfDiff;
}
std::string get_data_blob_name(const std::string& diff_blob_name) {
  CHECK(EndsWith(diff_blob_name, kSuffixOfDiff));
  return RemoveSuffix(diff_blob_name, kSuffixOfDiff);
}
std::string full_blob_name_in_layer(const std::string& layer_name,
  const std::string& var_suffix) {
  return layer_name + "/" + var_suffix;
}
std::string full_blob_name_in_dag(const std::string& layer_name,
  const std::string& alias_suffix) {
  return layer_name + "/" + alias_suffix;
}
std::string blob_variable_name_in_array(
  int32_t idx, const std::string& var_suffix) {
  return std::to_string(idx) + "/" + var_suffix;
}
bool is_array_blob(const std::string& blob_var_name) {
  return blob_var_name.find('/') != std::string::npos;

  // there is 2 '/' in array variable name
  //size_t count = std::count(blob_var_name.begin(), blob_var_name.end(), '/');
  //return count == 2;
}
void parse_blob_variable_name_in_array(const std::string& blob_var_name,
  int32_t *idx, std::string *var_suffix) {
  //// remove the prefix "data/" or "model/" firstly.
  //std::string blob_var_without_prefix
  //  = remove_prefix_of_blob_variable_name(blob_var_name);
  auto pos = blob_var_name.find('/');
  CHECK(pos != std::string::npos);
  std::string idx_str = blob_var_name.substr(0, pos);
  *idx = std::stoi(idx_str);
  *var_suffix = blob_var_name.substr(pos + 1);
  return;
}
std::string remove_prefix_of_blob_variable_name(const std::string& blob_var_name) {
  auto pos = blob_var_name.find('/');
  CHECK(pos != std::string::npos);
  return blob_var_name.substr(pos + 1);
}
}  // end namespace strings
}  // end namespace caffe