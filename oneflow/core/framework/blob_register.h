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
#ifndef ONEFLOW_CORE_FRAMEWORK_BLOB_REGISTER_H_
#define ONEFLOW_CORE_FRAMEWORK_BLOB_REGISTER_H_

#include <map>
#include "oneflow/core/framework/object.h"

namespace oneflow {

namespace compatible_py {

class BlobRegister;

class RegisteredBlobAccess {
 public:
  RegisteredBlobAccess(const std::string& blob_name,
                       const std::shared_ptr<BlobRegister>& blob_register,
                       const std::shared_ptr<BlobObject>& blob_object);
  ~RegisteredBlobAccess();

  int64_t reference_counter() const;
  std::shared_ptr<BlobObject> blob_object() const;
  void increase_reference_counter();
  int64_t decrease_reference_counter();

  std::shared_ptr<BlobRegister> blob_register() const;

 private:
  std::string blob_name_;
  std::shared_ptr<BlobRegister> blob_register_;
  int64_t reference_counter_;
  std::shared_ptr<BlobObject> blob_object_;
};

class BlobRegister : public std::enable_shared_from_this<BlobRegister> {
 public:
  BlobRegister(const std::function<void(std::shared_ptr<BlobObject>)>& release);
  ~BlobRegister() = default;

  std::shared_ptr<RegisteredBlobAccess> OpenRegisteredBlobAccess(
      const std::string& blob_name, const std::shared_ptr<BlobObject>& blob_object);

  void CloseRegisteredBlobAccess(const std::string& blob_name);

  std::shared_ptr<std::map<std::string, std::shared_ptr<BlobObject>>> blob_name2object() const;

  bool HasObject4BlobName(const std::string& blob_name) const;

  std::shared_ptr<BlobObject> GetObject4BlobName(const std::string& blob_name) const;

  void SetObject4BlobName(const std::string& blob_name, const std::shared_ptr<BlobObject>& obj);

  void TrySetObject4BlobName(const std::string& blob_name, const std::shared_ptr<BlobObject>& obj);

  void ClearObject4BlobName(const std::string& blob_name);

  void TryClearObject4BlobName(const std::string& blob_name);

  void ForceReleaseAll();

 private:
  std::shared_ptr<std::map<std::string, std::shared_ptr<BlobObject>>> blob_name2object_;
  std::shared_ptr<std::map<std::string, std::shared_ptr<RegisteredBlobAccess>>> blob_name2access_;
  std::function<void(std::shared_ptr<BlobObject>)> release_;
};

}  // namespace compatible_py

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_BLOB_REGISTER_H_
