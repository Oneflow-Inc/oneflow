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
#include "oneflow/core/framework/blob_register.h"

namespace oneflow {

namespace compatible_py {

RegisteredBlobAccess::RegisteredBlobAccess(const std::string& blob_name,
                                           const std::shared_ptr<BlobRegister>& blob_register,
                                           const std::shared_ptr<BlobObject>& blob_object)
    : blob_name_(blob_name), blob_register_(blob_register), reference_counter_(0) {
  if (!blob_object) {
    blob_object_ = blob_register->GetObject4BlobName(blob_name);
  } else {
    blob_object_ = blob_object;
    blob_register_->SetObject4BlobName(blob_name, blob_object);
  }
}

RegisteredBlobAccess::~RegisteredBlobAccess() { blob_register_->ClearObject4BlobName(blob_name_); }

int64_t RegisteredBlobAccess::reference_counter() const { return reference_counter_; }
std::shared_ptr<BlobObject> RegisteredBlobAccess::blob_object() const { return blob_object_; }
void RegisteredBlobAccess::increase_reference_counter() {
  reference_counter_ = reference_counter_ + 1;
}
int64_t RegisteredBlobAccess::decrease_reference_counter() {
  reference_counter_ = reference_counter_ - 1;
  return reference_counter_;
}

std::shared_ptr<BlobRegister> RegisteredBlobAccess::blob_register() const { return blob_register_; }

BlobRegister::BlobRegister(const std::function<void(std::shared_ptr<BlobObject>)>& release)
    : blob_name2object_(std::make_shared<std::map<std::string, std::shared_ptr<BlobObject>>>()),
      blob_name2access_(
          std::make_shared<std::map<std::string, std::shared_ptr<RegisteredBlobAccess>>>()),
      release_(release) {}

std::shared_ptr<RegisteredBlobAccess> BlobRegister::OpenRegisteredBlobAccess(
    const std::string& blob_name, const std::shared_ptr<BlobObject>& blob_object) {
  if (blob_name2access_->find(blob_name) == blob_name2access_->end()) {
    (*blob_name2access_)[blob_name] =
        std::make_shared<RegisteredBlobAccess>(blob_name, shared_from_this(), blob_object);
  }
  std::shared_ptr<RegisteredBlobAccess> access = blob_name2access_->at(blob_name);
  access->increase_reference_counter();
  return access;
}

void BlobRegister::CloseRegisteredBlobAccess(const std::string& blob_name) {
  if (blob_name2access_->find(blob_name) != blob_name2access_->end()) {
    if (blob_name2access_->at(blob_name)->decrease_reference_counter() == 0) {
      blob_name2access_->erase(blob_name);
    }
  }
}

std::shared_ptr<std::map<std::string, std::shared_ptr<BlobObject>>> BlobRegister::blob_name2object()
    const {
  return blob_name2object_;
}

bool BlobRegister::HasObject4BlobName(const std::string& blob_name) const {
  return blob_name2object_->find(blob_name) != blob_name2object_->end();
}

std::shared_ptr<BlobObject> BlobRegister::GetObject4BlobName(const std::string& blob_name) const {
  CHECK(blob_name2object_->find(blob_name) != blob_name2object_->end());
  return blob_name2object_->at(blob_name);
}

void BlobRegister::SetObject4BlobName(const std::string& blob_name,
                                      const std::shared_ptr<BlobObject>& obj) {
  CHECK(blob_name2object_->find(blob_name) == blob_name2object_->end());
  (*blob_name2object_)[blob_name] = obj;
}

void BlobRegister::TrySetObject4BlobName(const std::string& blob_name,
                                         const std::shared_ptr<BlobObject>& obj) {
  if (blob_name2object_->find(blob_name) == blob_name2object_->end()) {
    SetObject4BlobName(blob_name, obj);
  }
}

void BlobRegister::ClearObject4BlobName(const std::string& blob_name) {
  CHECK(HasObject4BlobName(blob_name)) << "blob_name " << blob_name << " not found";
  release_(blob_name2object_->at(blob_name));
  blob_name2object_->erase(blob_name);
}

void BlobRegister::TryClearObject4BlobName(const std::string& blob_name) {
  if (HasObject4BlobName(blob_name)) { ClearObject4BlobName(blob_name); }
}

void BlobRegister::ForceReleaseAll() {
  for (auto& pair : *blob_name2object_) {
    LOG(INFO) << "Forcely release blob " << (pair.first);
    pair.second->ForceReleaseAll();
  }
}

}  // namespace compatible_py

}  // namespace oneflow
