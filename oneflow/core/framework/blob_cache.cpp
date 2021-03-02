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
#include <mutex>
#include "oneflow/core/framework/blob_cache.h"

namespace oneflow {

namespace compatible_py {

namespace {

std::mutex& GlobalObjectId2BlobCacheMutex() {
  static std::mutex global_object_id2blob_cache_mutex;
  return global_object_id2blob_cache_mutex;
}

HashMap<int64_t, std::shared_ptr<BlobCache>>* GlobalObjectId2BlobCache() {
  static HashMap<int64_t, std::shared_ptr<BlobCache>> object_id2blob_cache;
  return &object_id2blob_cache;
}

}  // namespace

Maybe<EagerPhysicalBlobHeader> BlobCache::GetHeaderCache(
    const std::function<
        std::shared_ptr<EagerPhysicalBlobHeader>(const std::shared_ptr<BlobObject>&)>& Fetch) {
  if (!header_cache_) { header_cache_ = Fetch(blob_object_); }
  return header_cache_;
}

Maybe<BlobObject> BlobCache::GetCachedDelegateBlobObject(
    const std::shared_ptr<OpArgParallelAttribute>& op_arg_parallel_attr,
    const std::function<std::shared_ptr<BlobObject>(
        const std::shared_ptr<BlobObject>&, const std::shared_ptr<OpArgParallelAttribute>&)>&
        Fetch) {
  if (delegate_blob_object_.find(*op_arg_parallel_attr) == delegate_blob_object_.end()) {
    std::shared_ptr<BlobObject> delegate_blob_object = Fetch(blob_object_, op_arg_parallel_attr);
    delegate_blob_object_[*op_arg_parallel_attr] = delegate_blob_object;
  }
  return delegate_blob_object_.at(*op_arg_parallel_attr);
}

Maybe<BlobCache> FindOrCreateBlobCache(const std::shared_ptr<BlobObject>& blob_object) {
  int64_t object_id = blob_object->object_id();
  std::unique_lock<std::mutex> lock(GlobalObjectId2BlobCacheMutex());
  auto* object_id2blob_cache = GlobalObjectId2BlobCache();
  if (object_id2blob_cache->find(object_id) == object_id2blob_cache->end()) {
    (*object_id2blob_cache)[object_id] = std::make_shared<BlobCache>(blob_object);
  }
  return object_id2blob_cache->at(object_id);
}

Maybe<void> TryDisableBlobCache(const std::shared_ptr<BlobObject>& blob_object) {
  int64_t object_id = blob_object->object_id();
  std::unique_lock<std::mutex> lock(GlobalObjectId2BlobCacheMutex());
  auto* object_id2blob_cache = GlobalObjectId2BlobCache();
  if (object_id2blob_cache->find(object_id) != object_id2blob_cache->end()) {
    object_id2blob_cache->erase(object_id);
  }
  return Maybe<void>::Ok();
}

Maybe<BlobObject> FindOrCreateDelegateBlobObject(
    const std::function<std::shared_ptr<BlobObject>(
        const std::shared_ptr<BlobObject>&, const std::shared_ptr<OpArgParallelAttribute>&)>& Fetch,
    const std::shared_ptr<BlobObject>& x_blob_object,
    const std::shared_ptr<OpArgParallelAttribute>& op_arg_parallel_attr) {
  if ((*x_blob_object->op_arg_parallel_attr()) == (*op_arg_parallel_attr)) { return x_blob_object; }
  std::shared_ptr<BlobCache> blob_cache = JUST(FindOrCreateBlobCache(x_blob_object));
  return blob_cache->GetCachedDelegateBlobObject(op_arg_parallel_attr, Fetch);
}

Maybe<void> ClearAllBlobCache() {
  std::unique_lock<std::mutex> lock(GlobalObjectId2BlobCacheMutex());
  auto* object_id2blob_cache = GlobalObjectId2BlobCache();
  object_id2blob_cache->clear();
  return Maybe<void>::Ok();
}

}  // namespace compatible_py

}  // namespace oneflow
