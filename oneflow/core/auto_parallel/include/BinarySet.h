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
#include <vector>

namespace Algorithm {
class BinarySet {
 public:
  // BinarySetValues contains a vector of 64-bit int.
  // Each bit means whether an entry is in the set
  std::vector<unsigned long long int> BinarySetValues;

  int32_t SizeOfSet;

  BinarySet() {}
  BinarySet(int32_t size_of_set);

  // Initialization
  void Initialize(int32_t size_of_set);
  // Check if i-th element in this subset
  int32_t CheckExistency(int32_t i);
  // Add i-th element into this subset
  void AddEntry(int32_t i);
  // Take i-th element out from this subset
  void DeleteEntry(int32_t i);
  // Get the union with another subset and store it into u
  void UnionTo(BinarySet &bs, BinarySet &u);
  // Get the intersection with another subset and store it into i
  void IntersectionTo(BinarySet &bs, BinarySet &i);
  // Count number of elements in this subset
  int32_t Total();
  // Output all the elements in the subset
  void OutPut(std::vector<int32_t> &out);
  // Add elements of input into this subset
  void AddEntrys(std::vector<int32_t> &in);
};

}  // namespace Algorithm

#endif  // BINARY_SET_H_