#ifndef ONEFLOW_CORE_JOB_RANK_GROUP_SCOPE_H_
#define ONEFLOW_CORE_JOB_RANK_GROUP_SCOPE_H_

#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

class RankGroupScope final {
 public:
	~RankGroupScope();

	Symbol<RankGroup> rank_group() const { rank_group_; }
	const RankGroupScope& root() const { return *root_; }

	static Maybe<RankGroupScope> MakeNestedRankGroupScope(Symbol<RankGroup> rank_group);

	static Maybe<RankGroupScope> MakeInitialRankGroupScope(Symbol<RankGroup> rank_group);

	static Maybe<Symbol<RankGroup>> CurrentRankGroup();

	static Maybe<Symbol<RankGroup>> RootRankGroup();

 private:
	RankGroupScope(Symbol<RankGroup> rank_group, RankGroupScope* parent, RankGroupScope* root);

	Maybe<void> SetRootSelf();

	Symbol<RankGroup> rank_group_;
	RankGroupScope* parent_;
	RankGroupScope* root_;
};

}

#endif  // ONEFLOW_CORE_JOB_RANK_GROUP_SCOPE_H_
