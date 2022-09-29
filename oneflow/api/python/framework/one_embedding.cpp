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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/embedding/embedding_manager.h"
#include "oneflow/core/embedding/persistent_table.h"
#include "oneflow/core/embedding/hash_functions.cuh"
#include "oneflow/core/framework/dtype.h"

namespace py = pybind11;

namespace oneflow {

class OneEmbeddingHandler final {
 public:
  OneEmbeddingHandler(const std::string& key_value_store_option_string, int64_t local_rank_id,
                      int64_t rank_id, int64_t world_size)
      : local_rank_id_(local_rank_id), rank_id_(rank_id), world_size_(world_size) {
    embedding::KeyValueStoreOptions key_value_store_options(key_value_store_option_string);
    embedding_name_ = key_value_store_options.Name();
    CreateKeyValueStore(key_value_store_options);
  }

  void LoadSnapshot(const std::string& snapshot_name) {
#ifdef WITH_CUDA
    Singleton<embedding::EmbeddingManager>::Get()->LoadSnapshot(embedding_name_, local_rank_id_,
                                                                rank_id_, snapshot_name);
#else
    UNIMPLEMENTED() << "Only Support with CUDA";
#endif
  }

  void SaveSnapshot(const std::string& snapshot_name) {
#ifdef WITH_CUDA
    Singleton<embedding::EmbeddingManager>::Get()->SaveSnapshot(embedding_name_, local_rank_id_,
                                                                rank_id_, snapshot_name);
#else
    UNIMPLEMENTED() << "Only Support with CUDA";
#endif
  }

 private:
  void CreateKeyValueStore(const embedding::KeyValueStoreOptions& key_value_store_options) {
#ifdef WITH_CUDA
    Singleton<embedding::EmbeddingManager>::Get()->CreateKeyValueStore(
        key_value_store_options, local_rank_id_, rank_id_, world_size_);
#else
    UNIMPLEMENTED() << "Only Support with CUDA";
#endif
  }

  std::string embedding_name_;
  int64_t local_rank_id_;
  int64_t rank_id_;
  int64_t world_size_;
};

namespace embedding {

class PersistentTableWriter {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistentTableWriter);
  PersistentTableWriter() = default;
  virtual ~PersistentTableWriter() = default;

  virtual void Write(const py::array& keys, const py::array& values) = 0;
  virtual void Close() = 0;
};

template<typename Key, typename Value>
class PersistentTableWriterImpl : public PersistentTableWriter {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistentTableWriterImpl);
  PersistentTableWriterImpl(const std::vector<std::string>& paths, const std::string& snapshot_name,
                            uint32_t storage_dim, uint64_t target_chunk_size_mb,
                            uint16_t physical_block_size)
      : closed_(false), snapshot_name_(snapshot_name), storage_dim_(storage_dim) {
    tables_.resize(paths.size());
    for (size_t i = 0; i < paths.size(); ++i) {
      PersistentTableOptions options;
      options.path = paths[i];
      options.key_size = sizeof(Key);
      options.value_size = storage_dim * sizeof(Value);
      options.target_chunk_size_mb = target_chunk_size_mb;
      options.physical_block_size = physical_block_size;
      tables_[i] = NewPersistentTable(options);
    }
  }
  ~PersistentTableWriterImpl() override { CloseImpl(); }

  void Write(const py::array& keys, const py::array& values) override {
    pybind11::dtype::of<int32_t>().equal(pybind11::dtype::of<int64_t>());
    CHECK(!closed_) << "Write on closed table";
    CHECK_EQ(keys.ndim(), 1);
    CHECK_EQ(values.ndim(), 2);
    CHECK_EQ(keys.shape(0), values.shape(0));
    CHECK_EQ(values.shape(1), storage_dim_);
    CHECK(keys.dtype().equal(py::dtype::of<Key>()));
    CHECK(values.dtype().equal(py::dtype::of<Value>()));
    const size_t n = keys.size();
    std::vector<std::vector<Key>> keys_buffers(tables_.size());
    std::vector<std::vector<char>> values_buffers(tables_.size());
    for (size_t i = 0; i < n; ++i) {
      const Key key = *(reinterpret_cast<const Key*>(keys.template data(i)));
      const uint32_t shard = ShardingHash()(key) % tables_.size();
      keys_buffers[shard].push_back(key);
      const size_t values_offset = values_buffers[shard].size();
      values_buffers[shard].resize(values_offset + storage_dim_ * sizeof(Value));
      for (size_t j = 0; j < values.shape(1); ++j) {
        std::memcpy(values_buffers[shard].data() + values_offset + j * values.itemsize(),
                    values.template data(i, j), values.itemsize());
      }
    }
    for (size_t shard = 0; shard < tables_.size(); ++shard) {
      tables_[shard]->Put(keys_buffers[shard].size(), keys_buffers[shard].data(),
                          values_buffers[shard].data());
    }
  }

  void Close() override { CloseImpl(); }

 private:
  void CloseImpl() {
    if (!closed_) {
      for (auto& table : tables_) {
        table->SaveSnapshot(snapshot_name_);
        table.reset();
      }
    }
    closed_ = true;
  }

  bool closed_;
  std::string snapshot_name_;
  std::vector<std::unique_ptr<PersistentTable>> tables_;
  uint32_t storage_dim_;
};

template<typename Key>
std::shared_ptr<PersistentTableWriter> NewPersistentTableWriter(
    const std::vector<std::string>& paths, const std::string& snapshot_name,
    const Symbol<DType>& key_type, const Symbol<DType>& value_type, uint32_t storage_dim,
    uint64_t target_chunk_size_mb, uint16_t physical_block_size) {
  if (value_type->data_type() == DataType::kFloat) {
    return std::shared_ptr<PersistentTableWriter>(new PersistentTableWriterImpl<Key, float>(
        paths, snapshot_name, storage_dim, target_chunk_size_mb, physical_block_size));
  } else {
    UNIMPLEMENTED();
  }
}

std::shared_ptr<PersistentTableWriter> NewPersistentTableWriter(
    const std::vector<std::string>& paths, const std::string& snapshot_name,
    const Symbol<DType>& key_type, const Symbol<DType>& value_type, uint32_t storage_dim,
    uint64_t target_chunk_size_mb, uint16_t physical_block_size) {
  if (key_type->data_type() == DataType::kInt32) {
    return NewPersistentTableWriter<int32_t>(paths, snapshot_name, key_type, value_type,
                                             storage_dim, target_chunk_size_mb,
                                             physical_block_size);
  } else if (key_type->data_type() == DataType::kUInt32) {
    return NewPersistentTableWriter<uint32_t>(paths, snapshot_name, key_type, value_type,
                                              storage_dim, target_chunk_size_mb,
                                              physical_block_size);
  } else if (key_type->data_type() == DataType::kInt64) {
    return NewPersistentTableWriter<int64_t>(paths, snapshot_name, key_type, value_type,
                                             storage_dim, target_chunk_size_mb,
                                             physical_block_size);
  } else if (key_type->data_type() == DataType::kUInt64) {
    return NewPersistentTableWriter<uint64_t>(paths, snapshot_name, key_type, value_type,
                                              storage_dim, target_chunk_size_mb,
                                              physical_block_size);
  } else {
    UNIMPLEMENTED();
    return std::shared_ptr<embedding::PersistentTableWriter>(nullptr);
  }
}

class PersistentTableReader {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistentTableReader);
  PersistentTableReader() = default;
  virtual ~PersistentTableReader() = default;

  virtual std::tuple<py::object, py::object> Next() = 0;
  virtual void Close() = 0;
};

template<typename Key, typename Value>
class PersistentTableReaderImpl : public PersistentTableReader {
 public:
  constexpr static uint32_t kBatchSize = 65536;
  OF_DISALLOW_COPY_AND_MOVE(PersistentTableReaderImpl);
  PersistentTableReaderImpl(const std::vector<std::string>& paths, const std::string& snapshot_name,
                            uint32_t storage_dim, uint64_t target_chunk_size_mb,
                            uint16_t physical_block_size)
      : closed_(false),
        snapshot_name_(snapshot_name),
        storage_dim_(storage_dim),
        current_table_(0) {
    tables_.resize(paths.size());
    iterators_.resize(paths.size());
    for (size_t i = 0; i < paths.size(); ++i) {
      PersistentTableOptions options;
      options.path = paths[i];
      options.key_size = sizeof(Key);
      options.value_size = storage_dim * sizeof(Value);
      options.target_chunk_size_mb = target_chunk_size_mb;
      options.physical_block_size = physical_block_size;
      options.read_only = true;
      tables_[i] = NewPersistentTable(options);
      iterators_[i] =
          std::unique_ptr<PersistentTable::Iterator>(tables_[i]->ReadSnapshot(snapshot_name));
    }
    keys_buffer_.resize(kBatchSize);
    values_buffer_.resize(kBatchSize * storage_dim_);
  }
  ~PersistentTableReaderImpl() override { CloseImpl(); }

  std::tuple<py::object, py::object> Next() override {
    while (current_table_ < tables_.size()) {
      uint32_t n_result = 0;
      iterators_[current_table_]->Next(kBatchSize, &n_result, keys_buffer_.data(),
                                       values_buffer_.data());
      if (n_result != 0) {
        py::array_t<Key> keys_arr(py::array::ShapeContainer({n_result}));
        py::array_t<Value> values_arr(py::array::ShapeContainer({n_result, storage_dim_}));
        std::memcpy(keys_arr.mutable_data(), keys_buffer_.data(), n_result * sizeof(Key));
        std::memcpy(values_arr.mutable_data(), values_buffer_.data(),
                    n_result * storage_dim_ * sizeof(Value));
        return std::make_tuple(keys_arr, values_arr);
      } else {
        current_table_ += 1;
        continue;
      }
    }
    throw py::stop_iteration();
  }

  void Close() override { CloseImpl(); }

 private:
  void CloseImpl() {
    if (!closed_) {
      for (auto& table : tables_) { table.reset(); }
    }
    closed_ = true;
  }

  bool closed_;
  std::string snapshot_name_;
  std::vector<std::unique_ptr<PersistentTable>> tables_;
  std::vector<std::unique_ptr<PersistentTable::Iterator>> iterators_;
  uint32_t storage_dim_;
  size_t current_table_;
  std::vector<Key> keys_buffer_;
  std::vector<Value> values_buffer_;
};

template<typename Key>
std::shared_ptr<PersistentTableReader> NewPersistentTableReader(
    const std::vector<std::string>& paths, const std::string& snapshot_name,
    const Symbol<DType>& key_type, const Symbol<DType>& value_type, uint32_t storage_dim,
    uint64_t target_chunk_size_mb, uint16_t physical_block_size) {
  if (value_type->data_type() == DataType::kFloat) {
    return std::shared_ptr<PersistentTableReader>(new PersistentTableReaderImpl<Key, float>(
        paths, snapshot_name, storage_dim, target_chunk_size_mb, physical_block_size));
  } else {
    UNIMPLEMENTED();
  }
}

std::shared_ptr<PersistentTableReader> NewPersistentTableReader(
    const std::vector<std::string>& paths, const std::string& snapshot_name,
    const Symbol<DType>& key_type, const Symbol<DType>& value_type, uint32_t storage_dim,
    uint64_t target_chunk_size_mb, uint16_t physical_block_size) {
  if (key_type->data_type() == DataType::kInt32) {
    return NewPersistentTableReader<int32_t>(paths, snapshot_name, key_type, value_type,
                                             storage_dim, target_chunk_size_mb,
                                             physical_block_size);
  } else if (key_type->data_type() == DataType::kUInt32) {
    return NewPersistentTableReader<uint32_t>(paths, snapshot_name, key_type, value_type,
                                              storage_dim, target_chunk_size_mb,
                                              physical_block_size);
  } else if (key_type->data_type() == DataType::kInt64) {
    return NewPersistentTableReader<int64_t>(paths, snapshot_name, key_type, value_type,
                                             storage_dim, target_chunk_size_mb,
                                             physical_block_size);
  } else if (key_type->data_type() == DataType::kUInt64) {
    return NewPersistentTableReader<uint64_t>(paths, snapshot_name, key_type, value_type,
                                              storage_dim, target_chunk_size_mb,
                                              physical_block_size);
  } else {
    UNIMPLEMENTED();
    return std::shared_ptr<embedding::PersistentTableReader>(nullptr);
  }
}

}  // namespace embedding

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<OneEmbeddingHandler, std::shared_ptr<OneEmbeddingHandler>>(m, "OneEmbeddingHandler")
      .def(py::init([](const std::string& key_value_store_option_str, const int64_t local_rank_id,
                       const int64_t rank_id, const int64_t world_size) {
        return std::make_shared<OneEmbeddingHandler>(key_value_store_option_str, local_rank_id,
                                                     rank_id, world_size);
      }))
      .def("SaveSnapshot", &OneEmbeddingHandler::SaveSnapshot)
      .def("LoadSnapshot", &OneEmbeddingHandler::LoadSnapshot);

  py::class_<embedding::PersistentTableWriter, std::shared_ptr<embedding::PersistentTableWriter>>(
      m, "PersistentTableWriter")
      .def(py::init([](const std::vector<std::string>& paths, const std::string& snapshot_name,
                       const Symbol<DType>& key_type, const Symbol<DType>& value_type,
                       uint32_t storage_dim, uint64_t target_chunk_size_mb,
                       uint16_t physical_block_size) {
        return embedding::NewPersistentTableWriter(paths, snapshot_name, key_type, value_type,
                                                   storage_dim, target_chunk_size_mb,
                                                   physical_block_size);
      }))
      .def("__enter__", [](embedding::PersistentTableWriter* writer) { return writer; })
      .def("__exit__", [](embedding::PersistentTableWriter* writer, const py::object& exc_type,
                          const py::object& exc_val, const py::object& exc_tb) { writer->Close(); })
      .def("write", &embedding::PersistentTableWriter::Write)
      .def("close", &embedding::PersistentTableWriter::Close);

  py::class_<embedding::PersistentTableReader, std::shared_ptr<embedding::PersistentTableReader>>(
      m, "PersistentTableReader")
      .def(py::init([](const std::vector<std::string>& paths, const std::string& snapshot_name,
                       const Symbol<DType>& key_type, const Symbol<DType>& value_type,
                       uint32_t storage_dim, uint64_t target_chunk_size_mb,
                       uint16_t physical_block_size) {
        return embedding::NewPersistentTableReader(paths, snapshot_name, key_type, value_type,
                                                   storage_dim, target_chunk_size_mb,
                                                   physical_block_size);
      }))
      .def("__next__", &embedding::PersistentTableReader::Next)
      .def("__iter__", [](embedding::PersistentTableReader* reader) { return reader; })
      .def("__enter__", [](embedding::PersistentTableReader* reader) { return reader; })
      .def("__exit__", [](embedding::PersistentTableReader* reader, const py::object& exc_type,
                          const py::object& exc_val, const py::object& exc_tb) { reader->Close(); })
      .def("close", &embedding::PersistentTableReader::Close);
}

}  // namespace oneflow
