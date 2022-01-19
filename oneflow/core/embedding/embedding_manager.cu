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
#include "oneflow/core/embedding/embedding_manager.h"
#include "oneflow/core/embedding/fixed_table_key_value_store.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/embedding/cached_key_value_store.h"

namespace oneflow {

namespace embedding {}  // namespace embedding

EmbeddingMgr::~EmbeddingMgr() {
  printf("Here is deconstructor \n"); 
  int device_id = 0; 
  OF_CUDA_CHECK(cudaGetDevice(&device_id));
  printf("Here is in device: %d \n", device_id);

  for(auto& pair: key_value_store_map_){
    printf("Enter deconstructor for loop! \n");
    std::cout<<"map key is: "<<pair.first.first<<"-"<<pair.first.second<<std::endl;
  }
  // for (auto& pair : key_value_store_map_) { pair.second->SaveSnapshot("index"); }
  
  /*	
  std::pair<std::string, int64_t> map_key = std::make_pair("EmbeddingTest", 0);
  std::string snapshot_name("zzk_test_save"); 
  auto it = key_value_store_map_.find(map_key);
  if (it != key_value_store_map_.end()) {
    printf("Success \n");
    it->second->SaveSnapshot(snapshot_name);
  } else {
    LOG(ERROR) << "Can not find embedding: EmbeddingTest-0";
    // LOG(ERROR) << "Can not find this embedding: " << embedding_name << "-" << parallel_id;
  }
  */ 
}

embedding::KeyValueStore* EmbeddingMgr::GetOrCreateKeyValueStore(
    const embedding::EmbeddingOptions& embedding_options, int64_t parallel_id,
    int64_t parallel_num) {
  const std::string& name = embedding_options.EmbeddingName();
  const uint32_t line_size = embedding_options.LineSize();
  int device_id = 0; 
  OF_CUDA_CHECK(cudaGetDevice(&device_id)); 
  // printf("Here is in device: %d \n", device_id); 
  /*
  std::cout<<"Name is: "<<name<<std::endl; 
  std::cout<<"Parallel id is: "<<parallel_id<<std::endl; 
  */ 

  std::pair<std::string, int64_t> map_key = std::make_pair(name, parallel_id);
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = key_value_store_map_.find(map_key);
  if (it != key_value_store_map_.end()) { 
    // printf("Find successfully"); 	  
    return it->second.get(); }

  printf("======== Cannot Find !======== \n"); 
  LOG(ERROR) << "Map Key is: "<<name<<"-"<<parallel_id; 

  std::unique_ptr<embedding::KeyValueStore> store;
  const std::string& path = embedding_options.FixedTablePath();
  const std::string& num_rank = std::to_string(parallel_num);
  const int32_t rank_id_suffix_length = num_rank.size();
  const std::string& rank_id = std::to_string(parallel_id);
  embedding::FixedTableKeyValueStoreOptions options{};
  options.table_options.path = path + "/" + std::string(rank_id_suffix_length - rank_id.size(), '0')
                               + rank_id + "-" + num_rank;
  options.table_options.value_size = line_size * GetSizeOfDataType(DataType::kFloat);
  options.table_options.key_size = GetSizeOfDataType(DataType::kInt64);
  options.max_query_length = 65536 * 26;
  options.table_options.physical_block_size = embedding_options.FixedTableBlockSize();
  options.table_options.num_blocks_per_chunk = embedding_options.FixedTableChunkSize();
  store = NewFixedTableKeyValueStore(options);

  if (embedding_options.L2CachePolicy() != "none") {
    embedding::CacheOptions cache_options{};
    cache_options.value_memory_kind = embedding::CacheOptions::MemoryKind::kHost;
    if (embedding_options.L2CachePolicy() == "lru") {
      cache_options.policy = embedding::CacheOptions::Policy::kLRU;
    } else if (embedding_options.L2CachePolicy() == "full") {
      cache_options.policy = embedding::CacheOptions::Policy::kFull;
    } else {
      UNIMPLEMENTED();
    }
    cache_options.max_query_length = 65536 * 26;
    cache_options.key_size = GetSizeOfDataType(DataType::kInt64);
    cache_options.value_size = GetSizeOfDataType(DataType::kFloat) * line_size;
    cache_options.capacity =
        embedding_options.L2CacheMemoryBudgetMb() * 1024 * 1024 / cache_options.value_size;
    std::unique_ptr<embedding::Cache> cache = embedding::NewCache(cache_options);
    LOG(ERROR) << "add L2 cache: " << embedding_options.L2CachePolicy() << " "
               << embedding_options.L2CacheMemoryBudgetMb();
    store = NewCachedKeyValueStore(std::move(store), std::move(cache));
  }
  if (embedding_options.L1CachePolicy() != "none") {
    embedding::CacheOptions cache_options{};
    cache_options.value_memory_kind = embedding::CacheOptions::MemoryKind::kDevice;
    if (embedding_options.L1CachePolicy() == "lru") {
      cache_options.policy = embedding::CacheOptions::Policy::kLRU;
    } else if (embedding_options.L1CachePolicy() == "full") {
      cache_options.policy = embedding::CacheOptions::Policy::kFull;
    } else {
      UNIMPLEMENTED();
    }
    cache_options.max_query_length = 65536 * 26;
    cache_options.key_size = GetSizeOfDataType(DataType::kInt64);
    cache_options.value_size = GetSizeOfDataType(DataType::kFloat) * line_size;
    cache_options.capacity =
        embedding_options.L1CacheMemoryBudgetMb() * 1024 * 1024 / cache_options.value_size;
    std::unique_ptr<embedding::Cache> cache = embedding::NewCache(cache_options);
    LOG(ERROR) << "add L1 cache: " << embedding_options.L1CachePolicy() << " "
               << embedding_options.L1CacheMemoryBudgetMb();
    store = NewCachedKeyValueStore(std::move(store), std::move(cache));
  }
  if (store->SnapshotExists("index")) { store->LoadSnapshot("index"); }
  auto pair = key_value_store_map_.emplace(map_key, std::move(store));
  LOG(ERROR) << "emplace kv into MAP \n"; 
  CHECK(pair.second);
  return pair.first->second.get();
}

embedding::KeyValueStore* EmbeddingMgr::GetKeyValueStore(const std::string& embedding_name,
                                                         int64_t parallel_id,
                                                         int64_t parallel_num) {
  std::pair<std::string, int64_t> map_key = std::make_pair(embedding_name, parallel_id);
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = key_value_store_map_.find(map_key);
  return it->second.get();
}

void EmbeddingMgr::CreateKeyValueStore(std::string embedding_options_string,
                                       int64_t parallel_id, int64_t parallel_num,
                                       uint64_t cuda_device_id) {
  // This function is used in Python, so it need to set CudaDeviceId manually.
  OF_CUDA_CHECK(cudaSetDevice(cuda_device_id));
  // todo(use json directly)
  embedding::EmbeddingOptions embedding_options(embedding_options_string); 
  const std::string& name = embedding_options.EmbeddingName();
  const uint32_t line_size = embedding_options.LineSize();
  
  std::cout<<"Name is: "<<name<<std::endl; 
  std::cout<<"Parallel id is: "<<parallel_id<<std::endl; 
  std::pair<std::string, int64_t> map_key = std::make_pair(name, parallel_id);

  std::unique_ptr<embedding::KeyValueStore> store;
  const std::string& path = embedding_options.FixedTablePath();
  const std::string& num_rank = std::to_string(parallel_num);
  const int32_t rank_id_suffix_length = num_rank.size();
  const std::string& rank_id = std::to_string(parallel_id);
  embedding::FixedTableKeyValueStoreOptions options{};
  options.table_options.path = path + "/" + std::string(rank_id_suffix_length - rank_id.size(), '0')
                               + rank_id + "-" + num_rank;
  options.table_options.value_size = line_size * GetSizeOfDataType(DataType::kFloat);
  options.table_options.key_size = GetSizeOfDataType(DataType::kInt64);
  options.max_query_length = 65536 * 26;
  options.table_options.physical_block_size = embedding_options.FixedTableBlockSize();
  options.table_options.num_blocks_per_chunk = embedding_options.FixedTableChunkSize();
  store = NewFixedTableKeyValueStore(options);

  if (embedding_options.L2CachePolicy() != "none") {
    embedding::CacheOptions cache_options{};
    cache_options.value_memory_kind = embedding::CacheOptions::MemoryKind::kHost;
    if (embedding_options.L2CachePolicy() == "lru") {
      cache_options.policy = embedding::CacheOptions::Policy::kLRU;
    } else if (embedding_options.L2CachePolicy() == "full") {
      cache_options.policy = embedding::CacheOptions::Policy::kFull;
    } else {
      UNIMPLEMENTED();
    }
    cache_options.max_query_length = 65536 * 26;
    cache_options.key_size = GetSizeOfDataType(DataType::kInt64);
    cache_options.value_size = GetSizeOfDataType(DataType::kFloat) * line_size;
    cache_options.capacity =
        embedding_options.L2CacheMemoryBudgetMb() * 1024 * 1024 / cache_options.value_size;
    std::unique_ptr<embedding::Cache> cache = embedding::NewCache(cache_options);
    LOG(ERROR) << "add L2 cache: " << embedding_options.L2CachePolicy() << " "
               << embedding_options.L2CacheMemoryBudgetMb();
    store = NewCachedKeyValueStore(std::move(store), std::move(cache));
  }
  if (embedding_options.L1CachePolicy() != "none") {
    embedding::CacheOptions cache_options{};
    cache_options.value_memory_kind = embedding::CacheOptions::MemoryKind::kDevice;
    if (embedding_options.L1CachePolicy() == "lru") {
      cache_options.policy = embedding::CacheOptions::Policy::kLRU;
    } else if (embedding_options.L1CachePolicy() == "full") {
      cache_options.policy = embedding::CacheOptions::Policy::kFull;
    } else {
      UNIMPLEMENTED();
    }
    cache_options.max_query_length = 65536 * 26;
    cache_options.key_size = GetSizeOfDataType(DataType::kInt64);
    cache_options.value_size = GetSizeOfDataType(DataType::kFloat) * line_size;
    cache_options.capacity =
        embedding_options.L1CacheMemoryBudgetMb() * 1024 * 1024 / cache_options.value_size;
    std::unique_ptr<embedding::Cache> cache = embedding::NewCache(cache_options);
    LOG(ERROR) << "add L1 cache: " << embedding_options.L1CachePolicy() << " "
               << embedding_options.L1CacheMemoryBudgetMb();
    store = NewCachedKeyValueStore(std::move(store), std::move(cache));
  }
  auto pair = key_value_store_map_.emplace(map_key, std::move(store));
  if (!pair.second) { LOG(ERROR) << "Create Embedding failed!"; }
  else{
    printf("Create Embedding Success! \n"); 
  }
}

void EmbeddingMgr::SaveSnapshot(const std::string& embedding_name, int64_t parallel_id,
                                const std::string& snapshot_name) {
  // std::pair<std::string, int64_t> map_key = std::make_pair(embedding_name, parallel_id);
  std::unique_lock<std::mutex> lock(mutex_);
  std::cout<<"Here enter SaveSnapshot function"<<std::endl; 
  OF_CUDA_CHECK(cudaSetDevice(0)); 

  for(auto& pair: key_value_store_map_){
    printf("Enter for loop! \n");  
    std::cout<<"map key is: "<<pair.first.first<<"-"<<pair.first.second<<std::endl; 
  }
  
  std::pair<std::string, int64_t> map_key = std::make_pair("EmbeddingTest", 0);
  std::string tmp_snapshot_name("zzk_test_save"); 
  auto it = key_value_store_map_.find(map_key);
  if (it != key_value_store_map_.end()) {
    printf("Success \n");
    it->second->SaveSnapshot(tmp_snapshot_name);
  } else {
    LOG(ERROR) << "Can not find embedding: EmbeddingTest-0";
  }
  
  OF_CUDA_CHECK(cudaDeviceSynchronize());   
  /* 
  auto it = key_value_store_map_.find(map_key);
  if (it != key_value_store_map_.end()) {
    printf("Success \n"); 
    it->second->SaveSnapshot(snapshot_name);
  } else {
    LOG(ERROR) << "Can not find this embedding: " << embedding_name << "-" << parallel_id;
  }
  */
}

void EmbeddingMgr::LoadSnapshot(const std::string& embedding_name, int64_t parallel_id,
                                const std::string& snapshot_name) {
  std::pair<std::string, int64_t> map_key = std::make_pair(embedding_name, parallel_id);
  auto it = key_value_store_map_.find(map_key);
  if (it != key_value_store_map_.end()) {
    if (it->second->SnapshotExists(snapshot_name)) {
      it->second->LoadSnapshot(snapshot_name);
    } else {
      LOG(ERROR) << "Here Exists Embedding name is: " << embedding_name << "-" << parallel_id
                 << " but no corresponding snapshot. ";
    }
  } else {
    LOG(ERROR) << "Can not find the embedding: " << embedding_name << "-" << parallel_id;
  }
}

}  // namespace oneflow
