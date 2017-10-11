#ifndef ONEFLOW_CORE_KERNEL_BATCH_DATA.h
#define ONEFLOW_CORE_KERNEL_BATCH_DATA.h
namespace oneflow {

struct BatchHeaderDesc final {
	int32_t version = 0;
	char type[12]; // "feature" or "label"
	int32_t header_len;
	int32_t batch_size;
};

struct BatchFeatureHeader final {
	int32_t dim_array_size;
	int32_t dim_vec[0];
};

struct BatchLabelHeader final {
	int32_t label_array_size;
	char label_name[0][64];
};

struct BatchFeature final {
	double data[0];
};

struct BatchLabel final {
	int32_t feature_label_index[0];
};

#endif  // ONEFLOW_CORE_KERNEL_BATCH_DATA.h
