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
#ifndef ONEFLOW_CORE_FRAMEWORK_ATTR_MAP_H_
#define ONEFLOW_CORE_FRAMEWORK_ATTR_MAP_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

namespace user_op {
class AttrVal;
}
class AttrValue;
class MutableAttrMap;

// Make sure AttrName2AttrVal is a ordered map.
using AttrName2AttrVal = std::map<std::string, std::shared_ptr<const user_op::AttrVal>>;

class AttrName2AttrValWrapper {
 public:
  AttrName2AttrValWrapper(const std::shared_ptr<const AttrName2AttrVal>& attrs);
  AttrName2AttrValWrapper(const AttrName2AttrValWrapper&) = default;
  AttrName2AttrValWrapper(AttrName2AttrValWrapper&&) = default;
  ~AttrName2AttrValWrapper() = default;

  size_t size() const { return attrs_->size(); }
  bool empty() const { return attrs_->empty(); }

  AttrName2AttrValWrapper& operator=(const AttrName2AttrValWrapper& other) {
    attrs_ = other.attrs_;
    hash_value_ = other.hash_value_;
    return *this;
  }

  bool operator==(const AttrName2AttrValWrapper& other) const;

  using const_iterator = typename AttrName2AttrVal::const_iterator;
  const_iterator begin() const { return attrs_->begin(); }
  const_iterator end() const { return attrs_->end(); }

  const_iterator find(const std::string& attr_name) const { return attrs_->find(attr_name); }

  size_t hash_value() const { return hash_value_; }

 private:
  std::shared_ptr<const AttrName2AttrVal> attrs_;
  size_t hash_value_;
};

class AttrMap final {
 public:
  AttrMap();
  explicit AttrMap(const std::shared_ptr<const AttrName2AttrVal>& attrs);

  using value_type = typename AttrName2AttrVal::value_type;
  AttrMap(std::initializer_list<value_type> init);

  AttrMap(const MutableAttrMap& other);  // without coping AttrVal

  AttrMap(const AttrMap&) = default;
  AttrMap(AttrMap&&) = default;
  ~AttrMap() = default;

  AttrMap& operator=(const AttrMap& other);

  bool operator==(const AttrMap& other) const;

  template<typename T>
  Maybe<const T&> GetAttr(const std::string& attr_name) const;

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(const std::string& attr_name) const;

  size_t size() const { return attrs_.size(); }
  bool empty() const { return attrs_.empty(); }

  using const_iterator = typename AttrName2AttrVal::const_iterator;
  const_iterator begin() const { return attrs_.begin(); }
  const_iterator end() const { return attrs_.end(); }

  const_iterator find(const std::string& attr_name) const { return attrs_.find(attr_name); }

  size_t hash_value() const { return attrs_.hash_value(); }

 private:
  AttrName2AttrValWrapper attrs_;
};

class UserOpConf;
AttrMap MakeAttrMapFromUserOpConf(const UserOpConf& user_op_conf);

class ComposedAttrMap final {
 public:
  ComposedAttrMap(const ComposedAttrMap&) = default;
  ComposedAttrMap(ComposedAttrMap&&) = default;
  ComposedAttrMap(const AttrMap& base) : base_(base) {}
  ComposedAttrMap(const AttrMap& prior, const AttrMap& base) : prior_(prior), base_(base) {}

  template<typename T>
  Maybe<const T&> GetAttr(const std::string& attr_name) const;

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(const std::string& attr_name) const;

  void ResetPrior(const AttrMap& prior) { prior_ = prior; }
  void ResetBase(const AttrMap& base) { base_ = base; }

  std::string ToString() const;

 private:
  AttrMap prior_;
  AttrMap base_;
};

class MutableAttrMap : public std::map<std::string, std::shared_ptr<user_op::AttrVal>> {
 public:
  using std::map<std::string, std::shared_ptr<user_op::AttrVal>>::map;

  template<typename T>
  Maybe<void> SetAttr(const std::string& attr_name, const T& attr_val);
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::AttrName2AttrValWrapper> final {
  size_t operator()(const oneflow::AttrName2AttrValWrapper& attr_name2attr_val) const {
    return attr_name2attr_val.hash_value();
  }
};

template<>
struct hash<oneflow::AttrMap> final {
  size_t operator()(const oneflow::AttrMap& attr_map) const { return attr_map.hash_value(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_FRAMEWORK_ATTR_MAP_H_
