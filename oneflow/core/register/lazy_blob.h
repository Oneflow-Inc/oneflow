#ifndef ONEFLOW_CORE_REGISTER_LAZY_BLOB_H_
#define ONEFLOW_CORE_REGISTER_LAZY_BLOB_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/core/graph/graph.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

#define LAZY_EVALUATE(T, var)                                                          \
  for (LazyBlobVarBuilder<T> var(std::make_unique<LazyBlobGraph>()); var.Touch() == 0; \
       var.graph().Evaluate())

class Slice final {
 public:
  Slice(int64_t x);
  ~Slice() = default;
};

class LazyBlobEdge;
class LazyBlobGraph;

class LazyBlobNode : public Node<LazyBlobNode, LazyBlobEdge> {
 public:
  virtual ~LazyBlobNode() = default;

  void Evaluate() const {
    if (backend_blob()) { Evaluate(nullptr); }
  }
  virtual void Evaluate(Blob* output_blob) const = 0;

  // Getters
  const LazyBlobGraph* graph() const { return graph_; }
  const Blob* backend_blob() const { return backend_blob_; }
  const Shape& shape() const { return shape_; }
  DataType data_type() const { return data_type_; }

  // Setters
  LazyBlobGraph* mut_graph() const { return graph_; }

 protected:
  LazyBlobNode(LazyBlobGraph* graph, const Shape& shape, DataType data_type);
  LazyBlobNode(LazyBlobGraph* graph, Blob* backend_blob);

  Blob* mut_backend_blob() const { return backend_blob_; }

 private:
  LazyBlobGraph* graph_;
  Blob* backend_blob_;
  Shape shape_;
  DataType data_type_;
};

class LazyBlobEdge final : public Edge<LazyBlobNode, LazyBlobEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyBlobEdge);
  LazyBlobEdge() = default;
  ~LazyBlobEdge() = default;
};

class LazyBlobGraph final : public Graph<LazyBlobNode, LazyBlobEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyBlobGraph);
  LazyBlobGraph() = default;
  ~LazyBlobGraph() = default;

  void Evaluate() const {
    TopoForEachNode([](const LazyBlobNode* blob_node) { blob_node->Evaluate(); });
  }

  bool IsBlobAssigned(const Blob* blob) const {
    return assigned_blob_.find(blob) != assigned_blob_.end();
  }
  void AddAssignedBlob(const Blob* blob) { CHECK(assigned_blob_.emplace(blob).second); }

 private:
  HashSet<const Blob*> assigned_blob_;
};

template<typename DerivedT>
class LazyBlobIf : public LazyBlobNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyBlobIf);
  virtual ~LazyBlobIf() = default;

  virtual void Evaluate(Blob* output_blob) const override {
    CHECK(this->backend_blob() == nullptr);
    switch (this->shape().NumAxes()) {
      case 1: return Evaluate1(output_blob);
      case 2: return Evaluate2(output_blob);
      case 3: return Evaluate3(output_blob);
      case 4: return Evaluate4(output_blob);
      case 5: return Evaluate5(output_blob);
      default: UNIMPLEMENTED();
    }
  }

  /*
  LazyBlobNode& operator()(const Slice& dim0) { UNIMPLEMENTED(); }
  LazyBlobNode& operator()(const Slice& dim0, const Slice& dim1) { UNIMPLEMENTED(); }
  LazyBlobNode& operator()(const Slice& dim0, const Slice& dim1, const Slice& dim2) {
    UNIMPLEMENTED();
  }
  LazyBlobNode& operator()(const Slice& dim0, const Slice& dim1, const Slice& dim2,
                           const Slice& dim3) {
    UNIMPLEMENTED();
  }
  LazyBlobNode& operator()(const Slice& dim0, const Slice& dim1, const Slice& dim2,
                           const Slice& dim3, const Slice& dim4) {
    UNIMPLEMENTED();
  }
  */

 protected:
  LazyBlobIf(LazyBlobGraph* graph, const Shape& shape, DataType data_type)
      : LazyBlobNode(graph, shape, data_type) {}
  LazyBlobIf(LazyBlobGraph* graph, Blob* backend_blob) : LazyBlobNode(graph, backend_blob) {}

 private:
  void Evaluate1(Blob* output_blob) const;
  void Evaluate2(Blob* output_blob) const;
  void Evaluate3(Blob* output_blob) const;
  void Evaluate4(Blob* output_blob) const;
  void Evaluate5(Blob* output_blob) const;
};

template<typename T>
class VarLazyBlob final : public LazyBlobIf<VarLazyBlob<T>> {
 public:
  typedef T dtype;

  OF_DISALLOW_COPY_AND_MOVE(VarLazyBlob);
  VarLazyBlob(LazyBlobGraph* graph, Blob* backend_blob)
      : LazyBlobIf<VarLazyBlob<T>>(graph, backend_blob),
        dptr_(backend_blob->mut_dptr<T>()),
        dim0_next_dim_count_(ShapeDefaultedNextDimCount(backend_blob->shape(), 0)),
        dim1_next_dim_count_(ShapeDefaultedNextDimCount(backend_blob->shape(), 1)),
        dim2_next_dim_count_(ShapeDefaultedNextDimCount(backend_blob->shape(), 2)),
        dim3_next_dim_count_(ShapeDefaultedNextDimCount(backend_blob->shape(), 3)),
        dim4_next_dim_count_(ShapeDefaultedNextDimCount(backend_blob->shape(), 4)),
        value_lazy_blob_node_(nullptr) {
    CHECK_EQ(GetDataType<T>::value, backend_blob->data_type());
  }

  VarLazyBlob<T>& operator=(const LazyBlobNode& value_lazy_blob_node) {
    CHECK(value_lazy_blob_node.backend_blob() == nullptr);
    CHECK(!this->graph()->IsBlobAssigned(this->backend_blob()))
        << "a blob should be only assigned once";
    CHECK(this->shape() == value_lazy_blob_node.shape());
    value_lazy_blob_node_ = &value_lazy_blob_node;
    this->mut_graph()->AddAssignedBlob(this->backend_blob());
    return *this;
  }

  void Evaluate(Blob* output_blob) const override {
    CHECK(output_blob == nullptr);
    if (value_lazy_blob_node_ == nullptr) { return; }
    value_lazy_blob_node_->Evaluate(this->mut_backend_blob());
  }

  inline dtype operator()(int64_t dim0) const { return dptr_[dim0]; }
  inline dtype operator()(int64_t dim0, int64_t dim1) const {
    return dptr_[dim0 * dim0_next_dim_count_ + dim1];
  }
  inline dtype operator()(int64_t dim0, int64_t dim1, int64_t dim2) const {
    return dptr_[dim0 * dim0_next_dim_count_ + dim1 * dim1_next_dim_count_ + dim2];
  }
  inline dtype operator()(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    return dptr_[dim0 * dim0_next_dim_count_ + dim1 * dim1_next_dim_count_
                 + dim2 * dim2_next_dim_count_ + dim3];
  }
  inline dtype operator()(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                          int64_t dim4) const {
    return dptr_[dim0 * dim0_next_dim_count_ + dim1 * dim1_next_dim_count_
                 + dim2 * dim2_next_dim_count_ + dim3 * dim3_next_dim_count_ + dim4];
  }

 private:
  int64_t ShapeDefaultedNextDimCount(const Shape& shape, int32_t index) const {
    CHECK_GE(index, 0);
    return (index + 1 < shape.NumAxes() ? shape.Count(index + 1) : MaxVal<int32_t>());
  };
  T* dptr_;
  const int64_t dim0_next_dim_count_;
  const int64_t dim1_next_dim_count_;
  const int64_t dim2_next_dim_count_;
  const int64_t dim3_next_dim_count_;
  const int64_t dim4_next_dim_count_;
  const LazyBlobNode* value_lazy_blob_node_;
};

template<template<typename> class CoreFunc, typename XT>
class UnaryExpresionLazyBlob final : public LazyBlobIf<UnaryExpresionLazyBlob<CoreFunc, XT>> {
 public:
  using T = typename XT::dtype;
  typedef decltype(CoreFunc<T>::Invoke(*(const T*)nullptr)) dtype;

  OF_DISALLOW_COPY_AND_MOVE(UnaryExpresionLazyBlob);
  explicit UnaryExpresionLazyBlob(const XT& x)
      : LazyBlobIf<UnaryExpresionLazyBlob<CoreFunc, XT>>(x.mut_graph(), x.shape(),
                                                         GetDataType<dtype>::value),
        x_(x) {}

  inline dtype operator()(int64_t dim0) const { return CoreFunc<T>::Invoke(x_(dim0)); }
  inline dtype operator()(int64_t dim0, int64_t dim1) const {
    return CoreFunc<T>::Invoke(x_(dim0, dim1));
  }
  inline dtype operator()(int64_t dim0, int64_t dim1, int64_t dim2) const {
    return CoreFunc<T>::Invoke(x_(dim0, dim1, dim2));
  }
  inline dtype operator()(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    return CoreFunc<T>::Invoke(x_(dim0, dim1, dim2, dim3));
  }
  inline dtype operator()(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                          int64_t dim4) const {
    return CoreFunc<T>::Invoke(x_(dim0, dim1, dim2, dim3, dim4));
  }

 private:
  const XT& x_;
};

template<template<typename> class CoreFunc, typename XT, typename YT = XT,
         typename = typename std::enable_if<
             std::is_same<typename XT::dtype, typename YT::dtype>::value>::type>
class BinaryExpresionLazyBlob final : public LazyBlobIf<BinaryExpresionLazyBlob<CoreFunc, XT, YT>> {
 public:
  using T = typename XT::dtype;
  typedef decltype(CoreFunc<T>::Invoke(*(const T*)nullptr, *(const T*)nullptr)) dtype;

  OF_DISALLOW_COPY_AND_MOVE(BinaryExpresionLazyBlob);
  BinaryExpresionLazyBlob(const XT& x, const YT& y)
      : LazyBlobIf<BinaryExpresionLazyBlob<CoreFunc, XT, YT>>(x.mut_graph(), x.shape(),
                                                              GetDataType<dtype>::value),
        x_(x),
        y_(y) {
    CHECK(x.shape() == y.shape());
  }

  inline dtype operator()(int64_t dim0) const { return CoreFunc<T>::Invoke(x_(dim0), y_(dim0)); }
  inline dtype operator()(int64_t dim0, int64_t dim1) const {
    return CoreFunc<T>::Invoke(x_(dim0, dim1), y_(dim0, dim1));
  }
  inline dtype operator()(int64_t dim0, int64_t dim1, int64_t dim2) const {
    return CoreFunc<T>::Invoke(x_(dim0, dim1, dim2), y_(dim0, dim1, dim2));
  }
  inline dtype operator()(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    return CoreFunc<T>::Invoke(x_(dim0, dim1, dim2, dim3), y_(dim0, dim1, dim2, dim3));
  }
  inline dtype operator()(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                          int64_t dim4) const {
    return CoreFunc<T>::Invoke(x_(dim0, dim1, dim2, dim3, dim4), y_(dim0, dim1, dim2, dim3, dim4));
  }

 private:
  const XT& x_;
  const YT& y_;
};

template<typename XT>
class BroadcastLazyBlob final : public LazyBlobIf<BroadcastLazyBlob<XT>> {
 public:
  typedef typename XT::dtype dtype;

  OF_DISALLOW_MOVE(BroadcastLazyBlob);
  BroadcastLazyBlob(const BroadcastLazyBlob<XT>&) = default;
  BroadcastLazyBlob(const XT& x, const Shape& shape)
      : LazyBlobIf<BroadcastLazyBlob<XT>>(x.mut_graph(), shape, GetDataType<dtype>::value),
        x_(x),
        dim0_size_(DefaultedShapeAt(0)),
        dim1_size_(DefaultedShapeAt(1)),
        dim2_size_(DefaultedShapeAt(2)),
        dim3_size_(DefaultedShapeAt(3)),
        dim4_size_(DefaultedShapeAt(4)) {
    CheckShape(x.shape(), shape);
  }

  inline dtype operator()(int64_t dim0) const { return x_(dim0 % dim0_size_); }
  inline dtype operator()(int64_t dim0, int64_t dim1) const {
    return x_(dim0 % dim0_size_, dim1 % dim1_size_);
  }
  inline dtype operator()(int64_t dim0, int64_t dim1, int64_t dim2) const {
    return x_(dim0 % dim0_size_, dim1 % dim1_size_, dim2 % dim2_size_);
  }
  inline dtype operator()(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) const {
    return x_(dim0 % dim0_size_, dim1 % dim1_size_, dim2 % dim2_size_, dim3 % dim3_size_);
  }
  inline dtype operator()(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
                          int64_t dim4) const {
    return x_(dim0 % dim0_size_, dim1 % dim1_size_, dim2 % dim2_size_, dim3 % dim3_size_,
              dim4 % dim4_size_);
  }

 private:
  void CheckShape(const Shape& small_shape, const Shape& big_shape) {
    CHECK_EQ(small_shape.NumAxes(), big_shape.NumAxes());
    FOR_RANGE(int, i, 0, small_shape.NumAxes()) {
      CHECK_EQ(big_shape.At(i) % small_shape.At(i), 0);
    }
  }
  int64_t DefaultedShapeAt(const Shape& shape, int32_t index) const {
    CHECK_GE(index, 0);
    return (index < shape.NumAxes() ? shape.At(index) : MaxVal<int32_t>());
  };
  const XT& x_;
  const int64_t dim0_size_;
  const int64_t dim1_size_;
  const int64_t dim2_size_;
  const int64_t dim3_size_;
  const int64_t dim4_size_;
};

template<typename T>
class LazyBlobVarBuilder final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyBlobVarBuilder);
  explicit LazyBlobVarBuilder(std::unique_ptr<LazyBlobGraph>&& graph)
      : graph_(std::move(graph)), touched_cnt_(0) {}

  // for define lazy blob var
  template<typename dtype = T>
  VarLazyBlob<dtype>& operator()(Blob* blob) {
    auto* lazy_blob = new VarLazyBlob<dtype>(graph_.get(), blob);
    graph_->AddAllocatedNode(lazy_blob);
    return *lazy_blob;
  }

  // Getter
  const LazyBlobGraph& graph() const { return *graph_; }

  //  never use Touch() in your code
  int32_t Touch() { return touched_cnt_++; }

 private:
  std::unique_ptr<LazyBlobGraph> graph_;
  int32_t touched_cnt_;
};

#define LAZY_BLOB_BINARY_CORE_OP_FUNC_SEQ    \
  OF_PP_MAKE_TUPLE_SEQ(Add, +, T)            \
  OF_PP_MAKE_TUPLE_SEQ(Sub, -, T)            \
  OF_PP_MAKE_TUPLE_SEQ(Mul, *, T)            \
  OF_PP_MAKE_TUPLE_SEQ(Div, /, T)            \
  OF_PP_MAKE_TUPLE_SEQ(Mod, %, T)            \
  OF_PP_MAKE_TUPLE_SEQ(Eq, ==, bool)         \
  OF_PP_MAKE_TUPLE_SEQ(Ne, !=, bool)         \
  OF_PP_MAKE_TUPLE_SEQ(Gt, >, bool)          \
  OF_PP_MAKE_TUPLE_SEQ(Ge, >=, bool)         \
  OF_PP_MAKE_TUPLE_SEQ(Lt, <, bool)          \
  OF_PP_MAKE_TUPLE_SEQ(Le, <=, bool)         \
  OF_PP_MAKE_TUPLE_SEQ(LogicalAnd, &&, bool) \
  OF_PP_MAKE_TUPLE_SEQ(LogicalOr, &&, bool)

#define DECLARE_LAZY_BLOB_BINARY_CORE(name, op, ret_type)                  \
  template<typename T>                                                     \
  struct LazyBlobCore##name final {                                        \
    static inline ret_type Invoke(const T x, const T y) { return x op y; } \
  };
OF_PP_FOR_EACH_TUPLE(DECLARE_LAZY_BLOB_BINARY_CORE, LAZY_BLOB_BINARY_CORE_OP_FUNC_SEQ);

#define LAZY_BLOB_UNARY_CORE_OP_FUNC_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(Negative, -, T)   \
  OF_PP_MAKE_TUPLE_SEQ(LogicalNot, !, bool)

#define DECLARE_LAZY_BLOB_UNARY_CORE(name, op, ret_type)      \
  template<typename T>                                        \
  struct LazyBlobCore##name final {                           \
    static inline ret_type Invoke(const T x) { return op x; } \
  };
OF_PP_FOR_EACH_TUPLE(DECLARE_LAZY_BLOB_UNARY_CORE, LAZY_BLOB_UNARY_CORE_OP_FUNC_SEQ);

template<template<typename> class LazyBlobCoreFunc, typename XT, typename YT = XT>
typename std::enable_if<std::is_base_of<LazyBlobNode, XT>::value
                            && std::is_base_of<LazyBlobNode, YT>::value,
                        BinaryExpresionLazyBlob<LazyBlobCoreFunc, XT, YT>>::type&
BuildBinaryLazyBlob(XT& x, YT& y) {
  auto* ret = new BinaryExpresionLazyBlob<LazyBlobCoreFunc, XT, YT>(x, y);
  CHECK(x.graph() == y.graph());
  CHECK(x.shape() == y.shape());
  LazyBlobGraph* graph = x.mut_graph();
  LazyBlobNode* base_ret = ret;
  graph->AddAllocatedNode(base_ret);
  Connect(dynamic_cast<LazyBlobNode*>(&x), graph->NewEdge(), base_ret);
  Connect(dynamic_cast<LazyBlobNode*>(&y), graph->NewEdge(), base_ret);
  if (x.backend_blob() == nullptr) { CHECK_EQ(x.out_edges().size(), 1); }
  if (x.backend_blob() == nullptr) { CHECK_EQ(y.out_edges().size(), 1); }
  return *ret;
}

#define OVERLOAD_BINARY_LAZY_BLOB_OP_FUNC(name, op, ret_type)                         \
  template<typename XType, typename YType = XType,                                    \
           typename XT = typename std::remove_reference<XType>::type,                 \
           typename YT = typename std::remove_reference<YType>::type>                 \
  typename std::enable_if<std::is_base_of<LazyBlobNode, XT>::value                    \
                              && std::is_base_of<LazyBlobNode, YT>::value,            \
                          BinaryExpresionLazyBlob<LazyBlobCore##name, XT, YT>>::type& \
  operator op(XType& x, YType& y) {                                                   \
    return BuildBinaryLazyBlob<LazyBlobCore##name, XT, YT>(x, y);                     \
  }
OF_PP_FOR_EACH_TUPLE(OVERLOAD_BINARY_LAZY_BLOB_OP_FUNC, LAZY_BLOB_BINARY_CORE_OP_FUNC_SEQ);

template<template<typename> class LazyBlobCoreFunc, typename XT>
typename std::enable_if<std::is_base_of<LazyBlobNode, XT>::value,
                        UnaryExpresionLazyBlob<LazyBlobCoreFunc, XT>>::type&
BuildUnaryLazyBlob(XT& x) {
  auto* ret = new UnaryExpresionLazyBlob<LazyBlobCoreFunc, XT>(x);
  LazyBlobGraph* graph = x.mut_graph();
  LazyBlobNode* base_ret = ret;
  graph->AddAllocatedNode(base_ret);
  Connect(&x, graph->NewEdge(), base_ret);
  if (x.backend_blob() == nullptr) { CHECK_EQ(x.out_edges().size(), 1); }
  return *ret;
}

#define OVERLOAD_UNARY_LAZY_BLOB_OP_FUNC(name, op, ret_type)                          \
  template<typename XType, typename XT = typename std::remove_reference<XType>::type> \
  typename std::enable_if<std::is_base_of<LazyBlobNode, XT>::value,                   \
                          UnaryExpresionLazyBlob<LazyBlobCore##name, XT>>::type&      \
  operator op(XType& x) {                                                             \
    return BuildUnaryLazyBlob<LazyBlobCore##name, XT>(x);                             \
  }
OF_PP_FOR_EACH_TUPLE(OVERLOAD_UNARY_LAZY_BLOB_OP_FUNC, LAZY_BLOB_UNARY_CORE_OP_FUNC_SEQ);

//  implementations

template<typename DerivedT>
void LazyBlobIf<DerivedT>::Evaluate1(Blob* output_blob) const {
  using DT = typename DerivedT::dtype;
  CHECK_EQ(shape().NumAxes(), 1);
  DT* dptr = output_blob->mut_dptr<DT>();
  const DerivedT* this_ptr = dynamic_cast<const DerivedT*>(this);
  int64_t dim0_size = shape().At(0);
  FOR_RANGE(int64_t, i, 0, dim0_size) { dptr[i] = (*this_ptr)(i); }
}
template<typename DerivedT>
void LazyBlobIf<DerivedT>::Evaluate2(Blob* output_blob) const {
  using DT = typename DerivedT::dtype;
  CHECK_EQ(shape().NumAxes(), 2);
  DT* dptr = output_blob->mut_dptr<DT>();
  const DerivedT* this_ptr = dynamic_cast<const DerivedT*>(this);
  int64_t dim0_size = shape().At(0);
  int64_t dim1_size = shape().At(1);
  int64_t dim0_next_dim_count = shape().Count(1);
  FOR_RANGE(int64_t, i, 0, dim0_size) {
    DT* dptr_i = dptr + i * dim0_next_dim_count;
    FOR_RANGE(int64_t, j, 0, dim1_size) { dptr_i[j] = (*this_ptr)(i, j); }
  }
}
template<typename DerivedT>
void LazyBlobIf<DerivedT>::Evaluate3(Blob* output_blob) const {
  using DT = typename DerivedT::dtype;
  CHECK_EQ(shape().NumAxes(), 3);
  DT* dptr = output_blob->mut_dptr<DT>();
  const DerivedT* this_ptr = dynamic_cast<const DerivedT*>(this);
  int64_t dim0_size = shape().At(0);
  int64_t dim1_size = shape().At(1);
  int64_t dim2_size = shape().At(2);
  int64_t dim0_next_dim_count = shape().Count(1);
  int64_t dim1_next_dim_count = shape().Count(2);
  FOR_RANGE(int64_t, i, 0, dim0_size) {
    DT* dptr_i = dptr + i * dim0_next_dim_count;
    FOR_RANGE(int64_t, j, 0, dim1_size) {
      DT* dptr_j = dptr_i + j * dim1_next_dim_count;
      FOR_RANGE(int64_t, k, 0, dim2_size) { dptr_j[k] = (*this_ptr)(i, j, k); }
    }
  }
}
template<typename DerivedT>
void LazyBlobIf<DerivedT>::Evaluate4(Blob* output_blob) const {
  using DT = typename DerivedT::dtype;
  CHECK_EQ(shape().NumAxes(), 4);
  DT* dptr = output_blob->mut_dptr<DT>();
  const DerivedT* this_ptr = dynamic_cast<const DerivedT*>(this);
  int64_t dim0_size = shape().At(0);
  int64_t dim1_size = shape().At(1);
  int64_t dim2_size = shape().At(2);
  int64_t dim3_size = shape().At(3);
  int64_t dim0_next_dim_count = shape().Count(1);
  int64_t dim1_next_dim_count = shape().Count(2);
  int64_t dim2_next_dim_count = shape().Count(3);
  FOR_RANGE(int64_t, i, 0, dim0_size) {
    DT* dptr_i = dptr + i * dim0_next_dim_count;
    FOR_RANGE(int64_t, j, 0, dim1_size) {
      DT* dptr_j = dptr_i + j * dim1_next_dim_count;
      FOR_RANGE(int64_t, k, 0, dim2_size) {
        DT* dptr_k = dptr_j + k * dim2_next_dim_count;
        FOR_RANGE(int64_t, s, 0, dim3_size) { dptr_k[s] = (*this_ptr)(i, j, k, s); }
      }
    }
  }
}
template<typename DerivedT>
void LazyBlobIf<DerivedT>::Evaluate5(Blob* output_blob) const {
  using DT = typename DerivedT::dtype;
  CHECK_EQ(shape().NumAxes(), 5);
  DT* dptr = output_blob->mut_dptr<DT>();
  const DerivedT* this_ptr = dynamic_cast<const DerivedT*>(this);
  int64_t dim0_size = shape().At(0);
  int64_t dim1_size = shape().At(1);
  int64_t dim2_size = shape().At(2);
  int64_t dim3_size = shape().At(3);
  int64_t dim4_size = shape().At(4);
  int64_t dim0_next_dim_count = shape().Count(1);
  int64_t dim1_next_dim_count = shape().Count(2);
  int64_t dim2_next_dim_count = shape().Count(3);
  int64_t dim3_next_dim_count = shape().Count(4);
  FOR_RANGE(int64_t, i, 0, dim0_size) {
    DT* dptr_i = dptr + i * dim0_next_dim_count;
    FOR_RANGE(int64_t, j, 0, dim1_size) {
      DT* dptr_j = dptr_i + j * dim1_next_dim_count;
      FOR_RANGE(int64_t, k, 0, dim2_size) {
        DT* dptr_k = dptr_j + k * dim2_next_dim_count;
        FOR_RANGE(int64_t, s, 0, dim3_size) {
          DT* dptr_s = dptr_k + s * dim3_next_dim_count;
          FOR_RANGE(int64_t, t, 0, dim4_size) { dptr_s[t] = (*this_ptr)(i, j, k, s, t); }
        }
      }
    }
  }
}

LazyBlobNode::LazyBlobNode(LazyBlobGraph* graph, const Shape& shape, DataType data_type)
    : graph_(graph), backend_blob_(nullptr), shape_(shape), data_type_(data_type) {}

LazyBlobNode::LazyBlobNode(LazyBlobGraph* graph, Blob* backend_blob)
    : graph_(graph),
      backend_blob_(backend_blob),
      shape_(backend_blob->shape()),
      data_type_(backend_blob->data_type()) {}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_LAZY_BLOB_H_
