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

#ifndef ONEFLOW_MAYBE_UTILITY_H_
#define ONEFLOW_MAYBE_UTILITY_H_

#include <cstddef>
#include <functional>

namespace oneflow {

namespace maybe {

// unlike std::nullopt in c++17, the NullOptType is used in both Variant and Optional,
// so it is more like both std::nullopt and std::monostate (in c++17),
// the advantage of this unification is a more unifed experience,
// i.e. `return NullOpt` can be used in both Variant and Optional context
struct NullOptType {
  explicit constexpr NullOptType() = default;

  bool operator==(NullOptType) const { return true; }
  bool operator!=(NullOptType) const { return false; }
  bool operator<(NullOptType) const { return false; }
  bool operator>(NullOptType) const { return false; }
  bool operator<=(NullOptType) const { return true; }
  bool operator>=(NullOptType) const { return true; }
};

constexpr const std::size_t NullOptHash = -3333;

constexpr NullOptType NullOpt{};

struct InPlaceT {
  explicit constexpr InPlaceT() = default;
};

constexpr InPlaceT InPlace;

template<typename T>
struct InPlaceTypeT {
  explicit constexpr InPlaceTypeT() = default;
};

template<typename T>
constexpr InPlaceTypeT<T> InPlaceType;

template<std::size_t I>
struct InPlaceIndexT {
  explicit constexpr InPlaceIndexT() = default;
};

template<std::size_t I>
constexpr InPlaceIndexT<I> InPlaceIndex;

template<class T>
constexpr void HashCombine(std::size_t& seed, const T& v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

}  // namespace maybe

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::maybe::NullOptType> {
  size_t operator()(oneflow::maybe::NullOptType) const noexcept {
    return oneflow::maybe::NullOptHash;
  }
};

}  // namespace std

#endif  // ONEFLOW_MAYBE_UTILITY_H_
