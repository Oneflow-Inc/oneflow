#ifndef ONEFLOW_CORE_VECTORIZED_VEC_BINARY_MATH_H_
#define ONEFLOW_CORE_VECTORIZED_VEC_BINARY_MATH_H_
#include <iostream>
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/vectorized/vec.h"


namespace oneflow {

template<typename T>
void vec_binary_add(const T* x,const T* y, T* out, size_t len)
{
    vectorized_init();

    MultiThreadVecLoop(len, [=](size_t begin, size_t end){
        VecFunc<T>::add_func(begin, end, x, y, out);
    });
}

}
#endif
