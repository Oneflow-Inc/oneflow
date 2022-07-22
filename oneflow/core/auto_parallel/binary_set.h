/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_AUTO_PARALLEL_BINARY_SET_H_
#define ONEFLOW_CORE_AUTO_PARALLEL_BINARY_SET_H_

#include <cstdlib>
#include <unordered_map>
#include <vector>

// log_2_ index only support 32-bit int. Don't know why.
// Don't have any other bugs for unsigned int.
#define kBinarySetEntryType unsigned int

namespace oneflow {
namespace auto_parallel {

class BinarySet {
 public:
  BinarySet() {}
  explicit BinarySet(int32_t size_of_set);

  // Initialization
  void Initialize(int32_t size_of_set);
  // Clear all the elements in the set
  void Clear();
  // Check if i-th element in this subset
  int32_t CheckExistence(int32_t i) const;
  // Add i-th element into this subset
  void AddEntry(int32_t i);
  // Take i-th element out from this subset
  void DeleteEntry(int32_t i);
  // Get the union with another subset and store it into u
  void UnionTo(const BinarySet& bs, BinarySet& u);
  // If this binary set intersects another one
  bool IfIntersect(const BinarySet& bs) const;
  // Get the intersection with another subset and store it into i
  void IntersectionTo(const BinarySet& bs, BinarySet& i) const;
  // Count number of elements in this subset
  int32_t Total() const;
  // Output all the elements in the subset
  void OutPut(std::vector<int32_t>& out) const;
  // Output all the elements in the subset
  void QuickOutPut(std::vector<int32_t>& out) const;
  // Add elements of input into this subset
  void AddEntries(std::vector<int32_t>& in);
  // If two binary sets are equal to each other
  bool operator==(const BinarySet& rhs) const;

  inline int32_t GetSizeOfSet() const { return size_of_set_; };

 private:
  friend struct BinarySetHasher;
  // binary_set_values_ contains a vector of 64-bit or 32-bit int.
  // Each bit means whether an entry is in the set
  std::vector<kBinarySetEntryType> binary_set_values_;

  int32_t size_of_set_ = -1;

  // total bits of the entry type in vector binary_set_values_.
  static const int32_t bit_entry_type_ = 8 * sizeof(kBinarySetEntryType);
  // A static function for initialization of log_2_ mapping
  static std::unordered_map<kBinarySetEntryType, int32_t> InitLog2();
  // Take log2 of a integer value: 2^n -> n.
  static const std::unordered_map<kBinarySetEntryType, int32_t> log_2_;
};

struct BinarySetHasher {
  std::size_t operator()(const BinarySet& bs) const {
    using std::hash;
    using std::size_t;

    size_t h = 0;
    for (int i = 0; i < bs.binary_set_values_.size(); i++) {
      h ^= (hash<kBinarySetEntryType>()(bs.binary_set_values_[i]) << i);
    }
    return h;
  };
};

}  // namespace auto_parallel
}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PARALLEL_BINARY_SET_H_
