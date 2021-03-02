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
#ifndef ONEFLOW_CORE_FRAMEWORK_BLOB_CACHE_H_
#define ONEFLOW_CORE_FRAMEWORK_BLOB_CACHE_H_

#include <functional>
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/eager_blob_util.h"
#include "oneflow/core/framework/op_arg_util.h"

namespace oneflow {

namespace compatible_py {

class BlobCache {
 public:
  BlobCache(const std::shared_ptr<BlobObject>& blob_object) : blob_object_(blob_object) {}
  ~BlobCache() = default;

  std::shared_ptr<BlobObject> blob_object() const { return blob_object_; }

  std::shared_ptr<EagerPhysicalBlobHeader> GetHeaderCache(
      const std::function<
          std::shared_ptr<EagerPhysicalBlobHeader>(const std::shared_ptr<BlobObject>&)>& Fetch);

  std::shared_ptr<BlobObject> GetCachedDelegateBlobObject(
      const std::shared_ptr<OpArgParallelAttribute>& op_arg_parallel_attr,
      const std::function<std::shared_ptr<BlobObject>(
          const std::shared_ptr<BlobObject>&, const std::shared_ptr<OpArgParallelAttribute>&)>&
          Fetch);

 private:
  std::shared_ptr<BlobObject> blob_object_;
  std::shared_ptr<EagerPhysicalBlobHeader> header_cache_;
  HashMap<OpArgParallelAttribute, std::shared_ptr<BlobObject>> delegate_blob_object_;
};

Maybe<BlobCache> FindOrCreateBlobCache(const std::shared_ptr<BlobObject>& blob_object);

Maybe<void> TryDisableBlobCache(const std::shared_ptr<BlobObject>& blob_object);

Maybe<BlobObject> FindOrCreateDelegateBlobObject(
    const std::function<std::shared_ptr<BlobObject>(
        const std::shared_ptr<BlobObject>&, const std::shared_ptr<OpArgParallelAttribute>&)>& Fetch,
    const std::shared_ptr<BlobObject>& x_blob_object,
    const std::shared_ptr<OpArgParallelAttribute>& op_arg_parallel_attr);

Maybe<void> ClearAllBlobCache();

}  // namespace compatible_py

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_BLOB_CACHE_H_
