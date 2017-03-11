#include "memory/cpu_device_memory.h"
#include "memory/gpu_device_memory.h"
#include "memory/cuda_pinned_memory.h"
#include "memory/naive_manage_strategy.h"
#include "memory/memory_manager.h"
#include "gtest/gtest.h"

TEST(CPUDeviceMemoryTest, TestAllocAndFree) {
  size_t size = 1024 * 1024;
  void* ptr = caffe::CPUDeviceMemory::Alloc(size);
  ASSERT_NE(ptr, nullptr);
  caffe::CPUDeviceMemory::Free(ptr);
}

template <class T>
class DeviceMemoryTest : public ::testing::Test {
 public:
  static void* Alloc(size_t size) {
    return T::Alloc(size);
  }

  static void Free(void* ptr) {
    T::Free(ptr);
  }
};

// Memory types to be test
typedef ::testing::Types<caffe::CPUDeviceMemory,
  caffe::GPUDeviceMemory,
  caffe::CUDAPinnedMemory> DeviceMemoryTypes;
TYPED_TEST_CASE(DeviceMemoryTest, DeviceMemoryTypes);

TYPED_TEST(DeviceMemoryTest, TestAllocAndFree) {
  size_t size = 1024 * 1024;
  void* ptr = TestFixture::Alloc(size);
  TestFixture::Free(ptr);
}

template <class T>
class MemoryStrategyTest : public ::testing::Test {
 public:
  T* mem_strategy_; 

  virtual void SetUp() {
    mem_strategy_ = new T();
  }
};

// Strategy types to be test
typedef ::testing::Types<caffe::NaiveManageStrategy<caffe::CPUDeviceMemory>,
  caffe::NaiveManageStrategy<caffe::GPUDeviceMemory>,
  caffe::NaiveManageStrategy<caffe::CUDAPinnedMemory>> MemoryStrategyTypes;
TYPED_TEST_CASE(MemoryStrategyTest, MemoryStrategyTypes);

TYPED_TEST(MemoryStrategyTest, TestAllocAndFree) {
  size_t size = 1024 * 1024;
  TypeParam* mem_alloc = this->mem_strategy_;
  void* ptr = mem_alloc->Alloc(size);
  EXPECT_EQ(mem_alloc->Size(), size);
  mem_alloc->Free(ptr, size);
  EXPECT_EQ(mem_alloc->Size(), 0);
}

TEST(MemoryManagerTest, TestAllocAndFree) {
  size_t size = 1024 * 1024 * 1024ul;
  // Test CPU memory alloc
  caffe::MemoryManager::Context ctx;
  ctx.dev_type = caffe::MemoryManager::Context::kCPU;
  ctx.dev_id = 0;
  auto handle = caffe::MemoryManager::Get()->Alloc(size, ctx);
  EXPECT_EQ(handle.size, size);
  EXPECT_EQ(handle.ctx.dev_id, ctx.dev_id);
  EXPECT_EQ(handle.ctx.dev_type, ctx.dev_type);
  caffe::MemoryManager::Get()->Free(handle);

  // Test CUDA Pinned memory alloc
  ctx.dev_type = caffe::MemoryManager::Context::kCPUPinned;
  ctx.dev_id = 0;
  handle = caffe::MemoryManager::Get()->Alloc(size, ctx);
  EXPECT_EQ(handle.size, size);
  EXPECT_EQ(handle.ctx.dev_id, ctx.dev_id);
  EXPECT_EQ(handle.ctx.dev_type, ctx.dev_type);
  caffe::MemoryManager::Get()->Free(handle);

  // Test GPU memory alloc
  ctx.dev_type = caffe::MemoryManager::Context::kGPU;
  ctx.dev_id = 0;
  handle = caffe::MemoryManager::Get()->Alloc(size, ctx);
  EXPECT_EQ(handle.size, size);
  EXPECT_EQ(handle.ctx.dev_id, ctx.dev_id);
  EXPECT_EQ(handle.ctx.dev_type, ctx.dev_type);
  caffe::MemoryManager::Get()->Free(handle);
}
