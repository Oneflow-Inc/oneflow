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

namespace oneflow {

namespace cfg {
class AttrValue;
}
namespace user_op {
class AttrVal;
}
class AttrValue;
class MutableAttrMap;
class MutableCfgAttrMap;

using AttrName2AttrVal = HashMap<std::string, std::shared_ptr<const user_op::AttrVal>>;

const std::shared_ptr<const AttrName2AttrVal>& EmptyAttrName2AttrVal();

class AttrMap final {
 public:
  AttrMap() : attrs_(EmptyAttrName2AttrVal()) {}
  explicit AttrMap(const std::shared_ptr<const AttrName2AttrVal>& attrs) : attrs_(attrs) {}

  using value_type = typename AttrName2AttrVal::value_type;
  AttrMap(std::initializer_list<value_type> init);

  AttrMap(const MutableAttrMap& other);  // without coping AttrVal
  AttrMap(const MutableCfgAttrMap& other);

  AttrMap(const AttrMap&) = default;
  AttrMap(AttrMap&&) = default;
  ~AttrMap() = default;

  AttrMap& operator=(const AttrMap& other) {
    attrs_ = other.attrs_;
    return *this;
  }

  template<typename T>
  Maybe<const T&> GetAttr(const std::string& attr_name) const;

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(const std::string& attr_name) const;

  size_t size() const { return attrs_->size(); }
  bool empty() const { return attrs_->empty(); }

  using const_iterator = typename AttrName2AttrVal::const_iterator;
  const_iterator begin() const { return attrs_->begin(); }
  const_iterator end() const { return attrs_->end(); }

  const_iterator find(const std::string& attr_name) const { return attrs_->find(attr_name); }

 private:
  std::shared_ptr<const AttrName2AttrVal> attrs_;
};

class UserOpConf;
AttrMap MakeAttrMapFromUserOpConf(const UserOpConf& user_op_conf);

class ComposedAttrMap final {
 public:
  ComposedAttrMap(const AttrMap& base) : base_(base) {}
  ComposedAttrMap(const AttrMap& prior, const AttrMap& base) : prior_(prior), base_(base) {}

  template<typename T>
  Maybe<const T&> GetAttr(const std::string& attr_name) const;

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(const std::string& attr_name) const;

  void ResetPrior(const AttrMap& prior) { prior_ = prior; }
  void ResetBase(const AttrMap& base) { base_ = base; }

 private:
  AttrMap prior_;
  AttrMap base_;
};

class MutableAttrMap : public HashMap<std::string, std::shared_ptr<user_op::AttrVal>> {
 public:
  using HashMap<std::string, std::shared_ptr<user_op::AttrVal>>::HashMap;

  template<typename T>
  Maybe<void> SetAttr(const std::string& attr_name, const T& attr_val);
};

class MutableCfgAttrMap : public HashMap<std::string, std::shared_ptr<cfg::AttrValue>> {
 public:
  using HashMap<std::string, std::shared_ptr<cfg::AttrValue>>::HashMap;

  template<typename T>
  Maybe<void> SetAttr(const std::string& attr_name, const T& attr_val);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_ATTR_MAP_H_
