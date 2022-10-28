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
#include "oneflow/core/auto_parallel/binary_set.h"

namespace oneflow {
namespace auto_parallel {

namespace {
// A static function for initialization of log_2 mapping
std::unordered_map<BinarySetEntryType, int32_t> InitLog2() {
  std::unordered_map<BinarySetEntryType, int32_t> log_2;
  for (int32_t i = 0; i < 8 * sizeof(BinarySetEntryType); i++) {
    log_2[static_cast<BinarySetEntryType>(1 << i)] = i;
  }
  return log_2;
}

// Initialization of log_2 mapping
// Take log2 of a integer value: 2^n -> n.
const std::unordered_map<BinarySetEntryType, int32_t> log_2 = InitLog2();

}  // namespace

// Constructor
BinarySet::BinarySet(int32_t size_of_set) : size_of_set_(size_of_set) {
  int32_t k = (size_of_set - 1) / bit_entry_type_ + 1;
  binary_set_values_.resize(k, 0);
}

// Initialization if needed
void BinarySet::Initialize(int32_t size_of_set) {
  size_of_set_ = size_of_set;
  int32_t k = (size_of_set - 1) / bit_entry_type_ + 1;
  binary_set_values_.resize(k, 0);
}

// Clear all the elements in the set
void BinarySet::Clear() { binary_set_values_.assign(binary_set_values_.size(), 0); }

// Check if i-th element in this subset
bool BinarySet::CheckExistence(int32_t i) const {
  int32_t k = i / bit_entry_type_;
  int32_t j = i % bit_entry_type_;
  return bool((binary_set_values_[k] >> j) & 1);
}

// Add i-th element into this subset
void BinarySet::AddEntry(int32_t i) {
  int32_t k = i / bit_entry_type_;
  int32_t j = i % bit_entry_type_;
  binary_set_values_[k] |= (1 << j);
}
// Take i-th element out from this subset
void BinarySet::DeleteEntry(int32_t i) {
  int32_t k = i / bit_entry_type_;
  int32_t j = i % bit_entry_type_;
  binary_set_values_[k] &= ~(1 << j);
}
// Get the union with another subset and store it into u
void BinarySet::UnionTo(const BinarySet& bs, BinarySet& u) {
  for (int32_t k = 0; k < binary_set_values_.size(); k++) {
    u.binary_set_values_[k] = binary_set_values_[k] | bs.binary_set_values_[k];
  }
}
// If this binary set intersects another one
bool BinarySet::IfIntersect(const BinarySet& bs) const {
  int32_t min_bs_size = std::min(binary_set_values_.size(), bs.binary_set_values_.size());
  for (int32_t k = 0; k < min_bs_size; k++) {
    if (binary_set_values_[k] & bs.binary_set_values_[k]) { return true; }
  }
  return false;
}
// Get the intersection with another subset and store it into i
void BinarySet::IntersectionTo(const BinarySet& bs, BinarySet& i) const {
  int32_t min_bs_size = std::min(binary_set_values_.size(), bs.binary_set_values_.size());
  if (min_bs_size > i.binary_set_values_.size()) { i.binary_set_values_.resize(min_bs_size, 0); }
  for (int32_t k = 0; k < binary_set_values_.size(); k++) {
    i.binary_set_values_[k] = binary_set_values_[k] & bs.binary_set_values_[k];
  }
}
// Count number of elements in this subset
int32_t BinarySet::Total() const {
  int32_t t = 0;
  for (int32_t k = 0; k < binary_set_values_.size(); k++) {
    BinarySetEntryType bsv = binary_set_values_[k];
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
void BinarySet::Output(std::vector<int32_t>& out) const {
  out.clear();
  for (int32_t i = 0; i < size_of_set_; i++) {
    if (CheckExistence(i)) { out.emplace_back(i); }
  }
}

// Output all the elements in the subset
void BinarySet::QuickOutput(std::vector<int32_t>& out) const {
  out.clear();
  for (int32_t i = 0; i < binary_set_values_.size(); i++) {
    BinarySetEntryType x = binary_set_values_[i];
    BinarySetEntryType y = 0;
    while (x) {
      y = x;
      x &= x - 1;
      out.emplace_back(i * BinarySet::bit_entry_type_ + log_2.find(y - x)->second);
    }
  }
}

// Add elements of input into this subset
void BinarySet::AddEntries(std::vector<int32_t>& in) {
  for (int32_t i : in) { AddEntry(i); }
}

// If two binary sets are equal to each other
bool BinarySet::operator==(const BinarySet& rhs) const {
  if (size_of_set_ != rhs.size_of_set_) { return false; }
  for (int32_t i = 0; i < binary_set_values_.size(); i++) {
    if (binary_set_values_[i] != rhs.binary_set_values_[i]) { return false; }
  }
  return true;
}

}  // namespace auto_parallel
}  // namespace oneflow
