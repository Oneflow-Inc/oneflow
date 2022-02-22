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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_ATTRS_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_ATTRS_H_

#include <string>
#include <vector>

#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/attr_value.h"

namespace oneflow {

using user_op::AttrVal;

class OpInterpCtx;

class OpAttrs {
 public:
  explicit OpAttrs(const OpInterpCtx* ctx) : ctx_(ctx) {}

  size_t count(const std::string& attr_name) const;

  template<typename T>
  Maybe<const T&> at(const std::string& attr_name) {
    return AttrValueCast<T>(*JUST(this->at(attr_name)));
  }
  Maybe<AttrVal> at(const std::string& attr_name) const;
  Maybe<AttrVal> operator[](const std::string& attr_name) const;

  class const_iterator {
   public:
    using bucket_iter = HashSet<std::string>::const_iterator;
    using reference = const std::pair<std::string, std::shared_ptr<AttrVal>>&;
    using pointer = const std::pair<std::string, std::shared_ptr<AttrVal>>*;

    const_iterator() = default;
    const_iterator(bucket_iter pos, bucket_iter limit, const OpAttrs* self)
        : pos_(pos), limit_(limit), self_(self) {
      CHECK_JUST(UpdateKV());
    }
    reference operator*() const { return kv_; }
    pointer operator->() const { return &kv_; }

    const_iterator& operator++() {
      pos_++;
      CHECK_JUST(UpdateKV());
      return *this;
    }
    bool operator==(const const_iterator& x) const { return pos_ == x.pos_ && self_ == x.self_; }
    bool operator!=(const const_iterator& x) const { return !(*this == x); }

   private:
    Maybe<void> UpdateKV() {
      if (pos_ != limit_) {
        kv_.first = *pos_;
        kv_.second = JUST(self_->at(*pos_));
      }
      return Maybe<void>::Ok();
    }

    bucket_iter pos_;
    bucket_iter limit_;
    const OpAttrs* self_;
    std::pair<std::string, std::shared_ptr<AttrVal>> kv_;
  };

  const_iterator begin() const;
  const_iterator end() const;

  bool operator==(const OpAttrs& other) const;

 private:
  const OpInterpCtx* ctx_;
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::OpAttrs> {
  size_t operator()(const oneflow::OpAttrs& attrs) const;
};

}  // namespace std

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_ATTRS_H_
