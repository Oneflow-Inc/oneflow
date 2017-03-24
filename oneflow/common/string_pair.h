#ifndef _DAG_STAGE_SEGMENT_PAIR_H_
#define _DAG_STAGE_SEGMENT_PAIR_H_
#include <string>
#include <functional>
// See http://en.cppreference.com/w/cpp/utility/hash
// http://stackoverflow.com/questions/17016175/c-unordered-map-using-a-custom-class-type-as-the-key
namespace caffe {
struct StringPair {
  StringPair() {}
  StringPair(const std::string& first_name, const std::string& second_name)
  : first(first_name), second(second_name) {}
  bool operator==(const StringPair& other) const {
    return first == other.first && second == other.second;
  }
  std::string first;
  std::string second;
};
}  // caffe
namespace std {
template <>
struct hash<caffe::StringPair> {
  typedef caffe::StringPair argument_type;
  typedef std::size_t result_type;
  result_type operator()(argument_type const& s) const {
    result_type const h1(std::hash<std::string>()(s.first));
    result_type const h2(std::hash<std::string>()(s.second));
    return h1 ^ (h2 << 1);
  }
};
}  // namespace std
#endif  // _DAG_STAGE_SEGMENT_PAIR_H_
