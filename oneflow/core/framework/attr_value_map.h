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
#ifndef ONEFLOW_CORE_FRAMEWORK_ATTR_VALUE_MAP_H_
#define ONEFLOW_CORE_FRAMEWORK_ATTR_VALUE_MAP_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/user_op_attr.cfg.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

class MutableAttrValueMap;
class MutableCfgAttrValueMap;

using AttrName2AttrVal = HashMap<std::string, std::shared_ptr<const AttrVal>>;

class AttrValueMap {
 public:
  explicit AttrValueMap(const std::shared_ptr<const AttrName2AttrVal>& attrs) : attrs_(attrs) {}

  // without coping AttrVal.
  explicit AttrValueMap(const MutableAttrValueMap& other);
  explicit AttrValueMap(const MutableCfgAttrValueMap& other);

  AttrValueMap(const AttrValueMap&) = default;
  AttrValueMap(AttrValueMap&&) = default;
  ~AttrValueMap() = default;

  template<typename T>
  Maybe<const T&> GetAttr(const std::string& attr_name) const;

  using const_iterator = AttrName2AttrVal::const_iterator;
  const_iterator begin() const { return attrs_->begin(); }
  const_iterator end() const { return attrs_->end(); }

  const_iterator find(const std::string& attr_name) const { return attrs_->find(attr_name); }

 private:
  std::shared_ptr<const AttrName2AttrVal> attrs_;
};

class ComposedAttrValueMap final {
 public:
  ComposedAttrValueMap(const AttrValueMap& base) : base_(base) {}
  ComposedAttrValueMap(const AttrValueMap& prior, const AttrValueMap& base)
      : prior_(prior), base_(base) {}

  template<typename T>
  Maybe<const T&> GetAttr(const std::string& attr_name) const;

  void ResetPrior(const AttrValueMap& prior) { prior_ = prior; }
  void ResetBase(const AttrValueMap& base) { base_ = base; }

 private:
  AttrValueMap prior_;
  AttrValueMap base_;
};

class MutableAttrValueMap : public HashMap<std::string, std::shared_ptr<AttrVal>> {
 public:
  using HashMap<std::string, std::shared_ptr<AttrVal>>::HashMap;

  template<typename T>
  Maybe<void> SetAttr(const std::string& attr_name, const T& attr_val);
};

class MutableCfgAttrValueMap : public HashMap<std::string, std::shared_ptr<cfg::AttrValue>> {
 public:
  using HashMap<std::string, std::shared_ptr<cfg::AttrValue>>::HashMap;

  template<typename T>
  Maybe<void> SetAttr(const std::string& attr_name, const T& attr_val);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_ATTR_VALUE_MAP_H_
