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
#include "binary_set.h"

namespace oneflow {
namespace auto_parallel {

// A static function for initialization of log_2 mapping
std::unordered_map<BinarySetEntryType, int32_t> BinarySet::Initialize_log2() {
  std::unordered_map<BinarySetEntryType, int32_t> log_2_;
  for (int32_t i = 0; i < BinarySet::bit_of_BinarySetEntryType; i++) {
    log_2_[(BinarySetEntryType)(1 << i)] = i;
  }
  return log_2_;
}

// Initialization of log_2 mapping
std::unordered_map<BinarySetEntryType, int32_t> BinarySet::log_2 = BinarySet::Initialize_log2();

// Constructor
BinarySet::BinarySet(int32_t size_of_set) : SizeOfSet(size_of_set) {
  int32_t k = (size_of_set - 1) / bit_of_BinarySetEntryType + 1;
  BinarySetValues.resize(k, 0);
}

// Initialization if needed
void BinarySet::Initialize(int32_t size_of_set) {
  SizeOfSet = size_of_set;
  int32_t k = (size_of_set - 1) / bit_of_BinarySetEntryType + 1;
  BinarySetValues.resize(k, 0);
}

// Check if i-th element in this subset
int32_t BinarySet::CheckExistency(int32_t i) {
  int32_t k = i / bit_of_BinarySetEntryType;
  int32_t j = i % bit_of_BinarySetEntryType;
  return BinarySetValues[k] >> j & 1;
}

// Add i-th element into this subset
void BinarySet::AddEntry(int32_t i) {
  int32_t k = i / bit_of_BinarySetEntryType;
  int32_t j = i % bit_of_BinarySetEntryType;
  BinarySetValues[k] |= 1 << j;
}
// Take i-th element out from this subset
void BinarySet::DeleteEntry(int32_t i) {
  int32_t k = i / bit_of_BinarySetEntryType;
  int32_t j = i % bit_of_BinarySetEntryType;
  BinarySetValues[k] &= ~(1 << j);
}
// Get the union with another subset and store it into u
void BinarySet::UnionTo(BinarySet& bs, BinarySet& u) {
  for (int32_t k = 0; k < BinarySetValues.size(); k++) {
    u.BinarySetValues[k] = BinarySetValues[k] | bs.BinarySetValues[k];
  }
}
// Get the intersection with another subset and store it into i
void BinarySet::IntersectionTo(BinarySet& bs, BinarySet& i) {
  for (int32_t k = 0; k < BinarySetValues.size(); k++) {
    i.BinarySetValues[k] = BinarySetValues[k] & bs.BinarySetValues[k];
  }
}
// Count number of elements in this subset
int32_t BinarySet::Total() {
  int32_t t = 0;
  for (int32_t k = 0; k < BinarySetValues.size(); k++) {
    BinarySetEntryType bsv = BinarySetValues[k];
    bsv = (bsv & 0x5555555555555555) + ((bsv >> 1) & 0x5555555555555555);
    bsv = (bsv & 0x3333333333333333) + ((bsv >> 2) & 0x3333333333333333);
    bsv = (bsv & 0x0F0F0F0F0F0F0F0F) + ((bsv >> 4) & 0x0F0F0F0F0F0F0F0F);
    bsv = (bsv & 0x00FF00FF00FF00FF) + ((bsv >> 8) & 0x00FF00FF00FF00FF);
    bsv = (bsv & 0x0000FFFF0000FFFF) + ((bsv >> 16) & 0x0000FFFF0000FFFF);
    // bsv = (bsv & 0x00000000FFFFFFFF) + ((bsv >> 32) & 0x00000000FFFFFFFF);
    t += int32_t(bsv);
  }
  return t;
}

// Output all the elements in the subset
void BinarySet::OutPut(std::vector<int32_t>& out) {
  out.clear();
  for (int32_t i = 0; i < SizeOfSet; i++) {
    if (CheckExistency(i)) { out.emplace_back(i); }
  }
}

// Output all the elements in the subset
void BinarySet::QuickOutPut(std::vector<int32_t>& out) {
  out.clear();
  for (int32_t i = 0; i < BinarySetValues.size(); i++) {
    BinarySetEntryType x = BinarySetValues[i];
    BinarySetEntryType y;
    while (x) {
      y = x;
      x &= x - 1;
      out.emplace_back(i * BinarySet::bit_of_BinarySetEntryType + log_2[y - x]);
    }
  }
}

// Add elements of input into this subset
void BinarySet::AddEntrys(std::vector<int32_t>& in) {
  for (int32_t i = 0; i < in.size(); i++) { AddEntry(i); }
}

// If two binary sets are equal to each other
bool BinarySet::operator==(const BinarySet& rhs) const {
  if (SizeOfSet != rhs.SizeOfSet) { return false; }
  for (int32_t i = 0; i < BinarySetValues.size(); i++) {
    if (BinarySetValues[i] != rhs.BinarySetValues[i]) { return false; }
  }
  return true;
}

}  // namespace auto_parallel
}  // namespace oneflow
