#add_library(oneflow SHARED inference_session.cpp job_instance.cpp tensor/tensor.cpp)
add_executable(oneflow_infer_test inference_session_test.cpp inference_session.cpp job_instance.cpp tensor/tensor.cpp)
#target_link_libraries(oneflow_infer_test oneflow)