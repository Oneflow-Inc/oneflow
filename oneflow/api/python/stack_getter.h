#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <frameobject.h>

void enable_eval_frame_shim_for_current_thread();

void push_frame(PyFrameObject* frame);
void pop_frame();

#ifdef __cplusplus
}
#endif
