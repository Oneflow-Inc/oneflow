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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/tensor_slice_copier.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/cpu/cpu_stream.h"
#include "oneflow/core/ep/include/device_manager_registry.h"

namespace oneflow {

namespace {

template<typename T>
void InitializeWithConf(const InitializerConf& conf, const uint32_t random_seed, Blob* blob) {
  KernelUtil<DeviceType::kCPU, T>::InitializeWithConf(nullptr, conf, random_seed, blob);
}

struct InitializeWithConfUtil final {
#define MAKE_INITIALIZE_SWITCH_ENTRY(func_name, T) func_name<T>
  DEFINE_STATIC_SWITCH_FUNC(void, InitializeWithConf, MAKE_INITIALIZE_SWITCH_ENTRY,
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ));
#undef MAKE_INITIALIZE_SWITCH_ENTRY
};

NdSbp GetNdSbp(const KernelConf& kernel_conf, const std::string& bn_in_op) {
  const auto& nd_sbp_map = kernel_conf.op_attribute().nd_sbp_signature().bn_in_op2nd_sbp();
  const auto& it = nd_sbp_map.find(bn_in_op);
  CHECK(it != nd_sbp_map.end());
  return NdSbp(it->second);
}

class OnDemandHostBlob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OnDemandHostBlob);
  explicit OnDemandHostBlob(const Blob* like) {
    Shape shape;
    like->shape().ToShape(&shape);
    blob_desc_.reset(new BlobDesc(shape, like->data_type()));
    Init();
  }
  explicit OnDemandHostBlob(const BlobDesc& blob_desc) {
    blob_desc_.reset(new BlobDesc(blob_desc));
    Init();
  }
  explicit OnDemandHostBlob(const Shape& shape, DataType data_type) {
    blob_desc_.reset(new BlobDesc(shape, data_type));
    Init();
  }
  ~OnDemandHostBlob() = default;

  Blob* blob() const { return blob_.get(); }

 private:
  void Init() {
    header.resize(blob_desc_->AlignedByteSizeOfBlobHeader());
    data.resize(blob_desc_->AlignedByteSizeOfBlobBody());
    MemoryCase host_mem_case;
    host_mem_case.mutable_host_mem();
    blob_.reset(new Blob(host_mem_case, blob_desc_.get(), header.data(), data.data()));
  }

  std::vector<char> header;
  std::vector<char> data;
  std::unique_ptr<Blob> blob_;
  std::unique_ptr<const BlobDesc> blob_desc_;
};

template<DeviceType device_type>
void SyncCopyToHost(ep::Stream* stream, const void* src, void* dst, size_t size);

template<>
void SyncCopyToHost<DeviceType::kCPU>(ep::Stream* stream, const void* src, void* dst, size_t size) {
  std::memcpy(dst, src, size);
}

#ifdef WITH_CUDA
template<>
void SyncCopyToHost<DeviceType::kCUDA>(ep::Stream* stream, const void* src, void* dst,
                                       size_t size) {
  OF_CUDA_CHECK(cudaStreamSynchronize(stream->As<ep::CudaStream>()->cuda_stream()));
  OF_CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault,
                                stream->As<ep::CudaStream>()->cuda_stream()));
  OF_CUDA_CHECK(cudaStreamSynchronize(stream->As<ep::CudaStream>()->cuda_stream()));
}
#endif

template<DeviceType device_type>
void SyncCopyToDevice(ep::Stream* stream, const void* src, void* dst, size_t size);

template<>
void SyncCopyToDevice<DeviceType::kCPU>(ep::Stream* stream, const void* src, void* dst,
                                        size_t size) {
  std::memcpy(dst, src, size);
}

#ifdef WITH_CUDA
template<>
void SyncCopyToDevice<DeviceType::kCUDA>(ep::Stream* stream, const void* src, void* dst,
                                         size_t size) {
  OF_CUDA_CHECK(cudaStreamSynchronize(stream->As<ep::CudaStream>()->cuda_stream()));
  OF_CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault,
                                stream->As<ep::CudaStream>()->cuda_stream()));
  OF_CUDA_CHECK(cudaStreamSynchronize(stream->As<ep::CudaStream>()->cuda_stream()));
}
#endif

template<DeviceType device_type>
std::string SyncReadStringFromBlob(ep::Stream* stream, const Blob* blob) {
  std::vector<char> content;
  const int64_t size = blob->shape().elem_cnt();
  content.resize(size);
  SyncCopyToHost<device_type>(stream, blob->dptr(), content.data(), size);
  return std::string(content.data(), content.size());
}

std::string GetTmpPartKey(const std::string& base, const int64_t parallel_id,
                          const int64_t parallel_num) {
  return "tmp-part-" + std::to_string(parallel_id) + "-" + std::to_string(parallel_num) + "-"
         + base;
}

std::string GetTmpPartKey(const std::string& base, const ParallelContext& parallel_ctx) {
  return GetTmpPartKey(base, parallel_ctx.parallel_id(), parallel_ctx.parallel_num());
}

void HostSliceCopy(Blob* dst, const TensorSliceView& dst_slice, const Blob* src,
                   const TensorSliceView& src_slice) {
  TensorSliceCopier copier(dst_slice, src_slice, dst->data_type(), DeviceType::kCPU);
  auto device = Global<ep::DeviceManagerRegistry>::Get()->GetDevice(DeviceType::kCPU, 0);
  CHECK(device);
  auto* stream = device->CreateStream();
  copier.Copy(stream, dst, src);
  device->DestroyStream(stream);
}

template<DeviceType device_type>
class AutoSyncBlobAccessor final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoSyncBlobAccessor);
  AutoSyncBlobAccessor(ep::Stream* stream, Blob* underlying, bool read_sync, bool write_sync)
      : stream_(stream),
        underlying_(underlying),
        read_sync_(read_sync),
        write_sync_(write_sync),
        host_blob_(underlying) {
    if (read_sync_) {
      SyncCopyToHost<device_type>(stream_, underlying_->dptr(), host_blob_.blob()->mut_dptr(),
                                  underlying_->ByteSizeOfBlobBody());
    }
  }
  ~AutoSyncBlobAccessor() {
    if (write_sync_) {
      SyncCopyToDevice<device_type>(stream_, host_blob_.blob()->dptr(), underlying_->mut_dptr(),
                                    underlying_->ByteSizeOfBlobBody());
    }
  }

  Blob* host_blob() { return host_blob_.blob(); }

 private:
  ep::Stream* stream_;
  Blob* underlying_;
  bool read_sync_;
  bool write_sync_;
  OnDemandHostBlob host_blob_;
};

template<>
class AutoSyncBlobAccessor<DeviceType::kCPU> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoSyncBlobAccessor);
  AutoSyncBlobAccessor(ep::Stream* stream, Blob* underlying, bool read_sync, bool write_sync)
      : underlying_(underlying) {}
  ~AutoSyncBlobAccessor() = default;

  Blob* host_blob() { return underlying_; }

 private:
  Blob* underlying_;
};

}  // namespace

template<DeviceType device_type>
class ModelInitV2Kernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelInitV2Kernel);
  ModelInitV2Kernel() = default;
  ~ModelInitV2Kernel() override = default;

 private:
  void VirtualKernelInit(KernelContext* ctx) override {
    const ParallelContext& parallel_ctx = this->kernel_conf().parallel_ctx();
    const auto& hierarchy =
        ParallelDesc(
            this->kernel_conf().op_attribute().parallel_conf_signature().op_parallel_conf())
            .hierarchy();
    NdIndexOffsetHelper<int64_t, SHAPE_MAX_AXIS_SIZE> hierarchy_index_helper(
        hierarchy->dim_vec().data(), hierarchy->NumAxes());
    std::vector<int64_t> parallel_rank(SHAPE_MAX_AXIS_SIZE);
    hierarchy_index_helper.OffsetToNdIndex(parallel_ctx.parallel_id(), parallel_rank.data());
    const auto& model_init_v2_conf = this->op_conf().model_init_v2_conf();
    const int64_t num_var = model_init_v2_conf.variable_op_name_size();
    CHECK_EQ(model_init_v2_conf.ref_size(), num_var);
    CHECK_EQ(model_init_v2_conf.original_variable_conf_size(), num_var);
    seeds_.reserve(num_var);
    tensor_slice_views_.reserve(num_var);
    FOR_RANGE(int64_t, i, 0, num_var) {
      int64_t seed_num = 1;
      int64_t seed_offset = 0;
      const auto& original_variable_conf = model_init_v2_conf.original_variable_conf(i);
      const NdSbp& nd_sbp = GetNdSbp(this->kernel_conf(), GenRepeatedBn("ref", i));
      FOR_RANGE(int64_t, j, 0, hierarchy->NumAxes()) {
        const SbpParallel& sbp_parallel = nd_sbp.sbp_parallel(j);
        CHECK(sbp_parallel.has_split_parallel() || sbp_parallel.has_broadcast_parallel());
        if (sbp_parallel.has_split_parallel()) {
          seed_num *= hierarchy->At(j);
          seed_offset = seed_offset * hierarchy->At(j) + parallel_rank.at(j);
        }
      }
      std::seed_seq seq{original_variable_conf.random_seed()};
      std::vector<int64_t> seeds(seed_num);
      seq.generate(seeds.begin(), seeds.end());
      seeds_.emplace_back(seeds.at(seed_offset));
      const Shape logical_blob_shape(original_variable_conf.shape());
      tensor_slice_views_.emplace_back(GetTensorSliceView4ParallelId(
          *hierarchy, nd_sbp, logical_blob_shape, parallel_ctx.parallel_id()));
    }
  }
  void Forward(KernelContext* ctx) const override { ForwardDataContent(ctx); }
  void ForwardDataContent(KernelContext* ctx) const override {
    const ModelInitV2OpConf& conf = this->op_conf().model_init_v2_conf();

    FOR_RANGE(int64_t, i, 0, conf.variable_op_name_size()) {
      Blob* ref = ctx->BnInOp2Blob(GenRepeatedBn("ref", i));
      const DataType data_type = ref->data_type();
      const VariableOpConf& original_variable_conf = conf.original_variable_conf(i);
      AutoSyncBlobAccessor<device_type> ref_accessor(ctx->stream(), ref, false, true);
      if (original_variable_conf.has_initializer()) {
        std::mt19937 random_seed_gen(seeds_.at(i));
        InitializeWithConfUtil::SwitchInitializeWithConf(
            SwitchCase(data_type), original_variable_conf.initializer(), random_seed_gen(),
            ref_accessor.host_blob());
      } else if (original_variable_conf.has_initialize_with_snapshot()) {
        const auto& snapshot_conf = original_variable_conf.initialize_with_snapshot();
        const std::string& var_lbn =
            GenLogicalBlobName(conf.variable_op_name(i), original_variable_conf.out());
        const std::string key = snapshot_conf.has_key() ? snapshot_conf.key() : var_lbn;
        const Shape logical_blob_shape(original_variable_conf.shape());
        const SnapshotReader reader(snapshot_conf.path());
        reader.Read(key, logical_blob_shape, tensor_slice_views_.at(i), ref_accessor.host_blob());
      } else {
        UNIMPLEMENTED();
      }
    }
  }

  std::vector<int64_t> seeds_;
  std::vector<TensorSliceView> tensor_slice_views_;
};

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kModelInitV2Conf, ModelInitV2Kernel);

template<DeviceType device_type>
class ModelLoadV2Kernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelLoadV2Kernel);
  ModelLoadV2Kernel() = default;
  ~ModelLoadV2Kernel() override = default;

 private:
  void VirtualKernelInit(KernelContext* ctx) override {
    const auto& hierarchy =
        ParallelDesc(
            this->kernel_conf().op_attribute().parallel_conf_signature().op_parallel_conf())
            .hierarchy();
    const auto& model_load_v2_conf = this->op_conf().model_load_v2_conf();
    const int64_t num_var = model_load_v2_conf.variable_op_name_size();
    CHECK_EQ(model_load_v2_conf.ref_size(), num_var);
    CHECK_EQ(model_load_v2_conf.original_variable_conf_size(), num_var);
    tensor_slice_views_.reserve(num_var);
    FOR_RANGE(int64_t, i, 0, num_var) {
      const NdSbp& nd_sbp = GetNdSbp(this->kernel_conf(), GenRepeatedBn("ref", i));
      const Shape logical_blob_shape(model_load_v2_conf.original_variable_conf(i).shape());
      tensor_slice_views_.emplace_back(
          GetTensorSliceView4ParallelId(*hierarchy, nd_sbp, logical_blob_shape,
                                        this->kernel_conf().parallel_ctx().parallel_id()));
    }
  }
  void Forward(KernelContext* ctx) const override { ForwardDataContent(ctx); }
  void ForwardDataContent(KernelContext* ctx) const override {
    const ModelLoadV2OpConf& conf = this->op_conf().model_load_v2_conf();
    const Blob* path = ctx->BnInOp2Blob("path");
    const std::string snapshot_path = SyncReadStringFromBlob<device_type>(ctx->stream(), path);
    SnapshotReader reader(snapshot_path);
    FOR_RANGE(int64_t, i, 0, conf.variable_op_name_size()) {
      Blob* ref = ctx->BnInOp2Blob(GenRepeatedBn("ref", i));
      const VariableOpConf& original_variable_conf = conf.original_variable_conf(i);
      const Shape logical_blob_shape(original_variable_conf.shape());
      const std::string& var_lbn =
          GenLogicalBlobName(conf.variable_op_name(i), original_variable_conf.out());
      AutoSyncBlobAccessor<device_type> ref_accessor(ctx->stream(), ref, false, true);
      reader.Read(var_lbn, logical_blob_shape, tensor_slice_views_.at(i), ref_accessor.host_blob());
    }
  }
  std::vector<TensorSliceView> tensor_slice_views_;
};

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kModelLoadV2Conf, ModelLoadV2Kernel);

template<DeviceType device_type>
class ModelSaveV2Kernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveV2Kernel);
  ModelSaveV2Kernel() = default;
  ~ModelSaveV2Kernel() override = default;

 private:
  void VirtualKernelInit(KernelContext* ctx) override {
    const auto& hierarchy =
        ParallelDesc(
            this->kernel_conf().op_attribute().parallel_conf_signature().op_parallel_conf())
            .hierarchy();
    const auto NeedDoSave = [&](const std::vector<int64_t>& parallel_rank,
                                const NdSbp& nd_sbp) -> bool {
      FOR_RANGE(int64_t, j, 0, hierarchy->NumAxes()) {
        const SbpParallel& sbp_parallel = nd_sbp.sbp_parallel(j);
        if (sbp_parallel.has_broadcast_parallel() && parallel_rank.at(j) != 0) { return false; }
      }
      return true;
    };
    NdIndexOffsetHelper<int64_t, SHAPE_MAX_AXIS_SIZE> hierarchy_index_helper(
        hierarchy->dim_vec().data(), hierarchy->NumAxes());
    std::vector<int64_t> parallel_rank(SHAPE_MAX_AXIS_SIZE);
    const auto& model_save_v2_conf = this->op_conf().model_save_v2_conf();
    const int64_t num_var = model_save_v2_conf.variable_op_name_size();
    CHECK_EQ(model_save_v2_conf.in_size(), num_var);
    CHECK_EQ(model_save_v2_conf.original_variable_conf_size(), num_var);
    counters_.reserve(num_var);
    part_id2slice_views_.reserve(num_var);
    need_do_saves_.reserve(num_var);
    part_ids_.reserve(num_var);
    FOR_RANGE(int64_t, i, 0, num_var) {
      counters_.emplace_back(new int64_t(0));
      const NdSbp& nd_sbp = GetNdSbp(this->kernel_conf(), GenRepeatedBn("in", i));
      const Shape logical_blob_shape(model_save_v2_conf.original_variable_conf(i).shape());
      bool variable_need_do_save = false;
      int64_t variable_part_id = 0;
      std::vector<TensorSliceView> variable_part_id2slice_views;
      variable_part_id2slice_views.reserve(hierarchy->elem_cnt());
      FOR_RANGE(int64_t, j, 0, hierarchy->elem_cnt()) {
        hierarchy_index_helper.OffsetToNdIndex(j, parallel_rank.data());
        bool cur_id_need_do_save = NeedDoSave(parallel_rank, nd_sbp);
        if (j == this->kernel_conf().parallel_ctx().parallel_id()) {
          variable_need_do_save = cur_id_need_do_save;
          variable_part_id = variable_part_id2slice_views.size();
        }
        if (cur_id_need_do_save) {
          variable_part_id2slice_views.emplace_back(GetTensorSliceView4ParallelRank(
              *hierarchy, nd_sbp, logical_blob_shape, parallel_rank));
        }
      }
      need_do_saves_.emplace_back(variable_need_do_save);
      part_ids_.emplace_back(variable_part_id);
      part_id2slice_views_.emplace_back(variable_part_id2slice_views);
    }
  }

  void Forward(KernelContext* ctx) const override { ForwardDataContent(ctx); }
  void ForwardDataContent(KernelContext* ctx) const override {
    const ModelSaveV2OpConf& conf = this->op_conf().model_save_v2_conf();
    const Blob* path_blob = ctx->BnInOp2Blob("path");
    const std::string snapshot_path = SyncReadStringFromBlob<device_type>(ctx->stream(), path_blob);
    SnapshotWriter writer(snapshot_path);
    SnapshotReader reader(snapshot_path);
    FOR_RANGE(int64_t, i, 0, conf.variable_op_name_size()) {
      if (!need_do_saves_.at(i)) { continue; }
      *(counters_.at(i)) += 1;
      const std::vector<TensorSliceView>& variable_part_id2slice_views = part_id2slice_views_.at(i);
      Blob* in_blob = ctx->BnInOp2Blob(GenRepeatedBn("in", i));
      const VariableOpConf& original_variable_conf = conf.original_variable_conf(i);
      const Shape logical_blob_shape(original_variable_conf.shape());
      const DataType data_type = original_variable_conf.data_type();
      AutoSyncBlobAccessor<device_type> in_accessor(ctx->stream(), in_blob, true, false);
      const std::string var_lbn =
          GenLogicalBlobName(conf.variable_op_name(i), original_variable_conf.out());
      const bool is_broadcast = ShapeView(logical_blob_shape) == in_blob->shape();
      if (is_broadcast) { CHECK_EQ(variable_part_id2slice_views.size(), 1); }
      const std::string key = is_broadcast ? var_lbn
                                           : GetTmpPartKey(var_lbn, part_ids_.at(i),
                                                           variable_part_id2slice_views.size());
      writer.Write(key, in_accessor.host_blob());
      if (!is_broadcast) {
        const std::string rpc_key =
            snapshot_path + "-" + var_lbn + "-Counter-" + std::to_string(*(counters_.at(i)));
        int32_t counter = Global<CtrlClient>::Get()->IncreaseCount(rpc_key);
        if (counter < variable_part_id2slice_views.size()) { continue; }
        TensorSliceView total_slice(logical_blob_shape);
        OnDemandHostBlob total_blob(logical_blob_shape, data_type);
        FOR_RANGE(int64_t, j, 0, variable_part_id2slice_views.size()) {
          const TensorSliceView part_slice = variable_part_id2slice_views.at(j);
          const std::string part_key =
              GetTmpPartKey(var_lbn, j, variable_part_id2slice_views.size());
          OnDemandHostBlob part_blob(part_slice.shape(), data_type);
          reader.Read(part_key, part_blob.blob());
          HostSliceCopy(total_blob.blob(), total_slice, part_blob.blob(), part_slice);
          SnapshotFS()->RecursivelyDeleteDir(Dirname(JoinPath(snapshot_path, part_key)));
        }
        writer.Write(var_lbn, total_blob.blob());
        Global<CtrlClient>::Get()->EraseCount(rpc_key);
      }
    }
  }
  std::vector<std::unique_ptr<int64_t>> counters_;
  std::vector<std::vector<TensorSliceView>> part_id2slice_views_;
  std::vector<bool> need_do_saves_;
  std::vector<int64_t> part_ids_;
};

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kModelSaveV2Conf, ModelSaveV2Kernel);

}  // namespace oneflow
