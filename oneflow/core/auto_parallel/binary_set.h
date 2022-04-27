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
#ifndef BINARY_SET_H_
#define BINARY_SET_H_

#include <cstdlib>
#include <unordered_map>
#include <vector>
// log_2 index only support 32-bit int. Don't know why. Don't have any other bugs for unsigned long
// long int.
#define BinarySetEntryType unsigned int

namespace oneflow {
namespace auto_parallel {

class BinarySet {
 public:
  // BinarySetValues contains a vector of 64-bit or 32-bit int.
  // Each bit means whether an entry is in the set
  std::vector<unsigned long long int> BinarySetValues;

  int32_t SizeOfSet;

  // total bits of the entry type in vector BinarySetValues.
  static const int32_t bit_of_BinarySetEntryType = 8 * sizeof(BinarySetEntryType);
  // Take log2 of a integer value: 2^n -> n.
  static std::unordered_map<BinarySetEntryType, int32_t> log_2;
  // A static function for initialization of log_2 mapping
  static std::unordered_map<BinarySetEntryType, int32_t> Initialize_log2();

  BinarySet() {}
  BinarySet(int32_t size_of_set);

  // Initialization
  void Initialize(int32_t size_of_set);
  // Clear all the elements in the set
  void Clear();
  // Check if i-th element in this subset
  int32_t CheckExistency(int32_t i) const;
  // Add i-th element into this subset
  void AddEntry(int32_t i);
  // Take i-th element out from this subset
  void DeleteEntry(int32_t i);
  // Get the union with another subset and store it into u
  void UnionTo(BinarySet& bs, BinarySet& u);
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
  void AddEntrys(std::vector<int32_t>& in);
  // If two binary sets are equal to each other
  bool operator==(const BinarySet& rhs) const;
};

struct BinarySetHasher {
  std::size_t operator()(const BinarySet& bs) const {
    using std::hash;
    using std::size_t;

    size_t h = 0;
    for (int i = 0; i < bs.BinarySetValues.size(); i++) {
      h ^= (hash<BinarySetEntryType>()(bs.BinarySetValues[i]) << i);
    }
    return h;
  };
};

}  // namespace auto_parallel
}  // namespace oneflow

#endif  // BINARY_SET_H_
