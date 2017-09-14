#ifndef ONEFLOW_CORE_SCHEDULE_FACTORY_UTIL_H_
#define ONEFLOW_CORE_SCHEDULE_FACTORY_UTIL_H_

#define CLONE_FACTORY(obj, getter)                                \
  do {                                                            \
    if (&obj->getter) { mut_##getter = obj->getter.Clone(this); } \
  } while (0)

#define DEFINE_FACTORY_METHOD_CLONE(class_name, base)                      \
  virtual std::unique_ptr<base> Clone(ScheduleFactoryProvider* ph) const { \
    return of_make_unique<class_name>(ph);                                 \
  }

#define DEFINE_FACTORY_PURE_VIRTUAL_CLONE(class_name)                    \
  virtual std::unique_ptr<class_name> Clone(ScheduleFactoryProvider* ph) \
      const = 0

#endif  // ONEFLOW_CORE_SCHEDULE_FACTORY_UTIL_H_
