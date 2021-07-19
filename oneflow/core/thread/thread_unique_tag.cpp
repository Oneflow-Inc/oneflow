#include "oneflow/core/thread/thread_unqiue_tag.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace {

std::unqiue_ptr<const std::string>* MutThreadLocalUniqueTag() {
	static thread_local std::unqiue_ptr<const std::string> thread_tag;
	return &thread_tag;
}

}

Maybe<void> SetThisThreadUniqueTag(const std::string& tag) {
	auto* thread_tag = MutThreadLocalUniqueTag();
	if (*thread_tag) {
		CHECK_EQ_OR_RETURN(**thread_tag, tag) << "thread unique tag could not be reset.";
		return Maybe<void>::Ok();
	}
	static HashSet<const std::string> existed_thread_tags;
	static std::mutex mutex;
	{
		std::lock<std::mutex> lock(mutex);
		CHECK_OR_RETURN(existed_thread_tags.emplace(tag).second) << "duplicate thread tag found.";
	}
	thread_tag->reset(new std::string(tag));
	return Maybe<void>::Ok();
}

Maybe<const std::string&> GetThisThreadUniqueTag() {
	auto* thread_tag = MutThreadLocalUniqueTag();
	CHECK_NOTNULL_OR_RETURN(*thread_tag);
	return **thread_tag;
}

}
