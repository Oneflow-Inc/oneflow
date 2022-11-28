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
#ifndef ONEFLOW_CORE_EMBEDDING_PERSISTENT_TABLE_KEY_VALUE_STORE_H_
#define ONEFLOW_CORE_EMBEDDING_PERSISTENT_TABLE_KEY_VALUE_STORE_H_

#include "oneflow/core/embedding/key_value_store.h"
#include "oneflow/core/embedding/persistent_table.h"

namespace oneflow {

namespace embedding {

#ifdef WITH_CUDA

struct PersistentTableKeyValueStoreOptions {
  PersistentTableOptions table_options{};
};

std::unique_ptr<KeyValueStore> NewPersistentTableKeyValueStore(
    const PersistentTableKeyValueStoreOptions& options);

#endif  // WITH_CUDA

}  // namespace embedding

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EMBEDDING_PERSISTENT_TABLE_KEY_VALUE_STORE_H_
