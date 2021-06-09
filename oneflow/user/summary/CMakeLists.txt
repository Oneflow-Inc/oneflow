oneflow_add_user_library(
    oneflow_user_summary

    ../kernels/summary_kernels.cpp
    events_writer.cpp
    event_writer_helper.cpp
    histogram.cpp
    plan_to_physical_graph.cpp
)
target_link_libraries(oneflow_user_summary ${OPENCV_STATIC_LIBRARIES})
