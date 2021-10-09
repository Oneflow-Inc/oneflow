### 概念与数据结构
本子系统可以方便用户定义可侵入式类型。内建支持侵入式智能指针intrusive::shared_ptr和侵入式容器。
目前有主要有两类侵入式容器：
1. intrusive::List，双链表。基于此，还提供了intrusive::MutexedList和intrusive::Channel。
2. intrusive::SkipList，跳表，等同于map。

为了管理元素CURD所带来的生命周期，侵入式容器需要intrusive::shared_ptr来实现内存生命周期的管理，它与std::shared_ptr的不同在于其引用计数嵌入在目标结构体里。
### 接口
由于侵入式容器要求其元素类型T必须满足std::is_standard_layout<T>::value，为了减少麻烦的编译问题及字段访问权限相关的悖论，我们使用一组宏定义可侵入类型。
1. INTRUSIVE_BEGIN，开始定义可侵入类型。
2. INTRUSIVE_END，结束定义可侵入类型。
3. INTRUSIVE_DEFINE_FIELD，定义可侵入类型的字段。
4. INTRUSIVE_FIELD，描述可侵入类型的某个字段，属性包括容器类型，字段类型，字段偏移，常用于侵入式容器的构建。

由于侵入式容器支持比标准容器更为强大的迭代方式，同时为了性能起见，我们提供三类迭代宏：
1. INTRUSIVE_FOR_EACH，支持迭代过程中删除当前元素，同时使用intrusive::shared_ptr管理好当前元素生命周期
2. INTRUSIVE_FOR_EACH_PTR，支持迭代过程中删除当前元素，类型直接为裸指针，即不负责当前元素生命周期的管理
3. INTRUSIVE_UNSAFE_FOR_EACH_PTR，不支持迭代中删除元素，不负责当前元素生命周期的管理。

### 特点
本组件与boost::intrusive最大不同在于实现了完整的生命周期管理，另外提供了其他更能减少内存分配的容器定义方式（详见intrusive::HeadFreeList）。

