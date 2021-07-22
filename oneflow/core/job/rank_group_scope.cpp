#include "oneflow/core/job/rank_group_scope.h"

namespace oneflow {

namespace {

RankGroupScope** MutThreadLocalRankGroupScope() {
	static thread_local RankGroupScope* = nullptr;
	return &RankGroupScope;
}

}

RankGroupScope::RankGroupScope(Symbol<RankGroup> rank_group, RankGroupScope* parent, RankGroupScope* root)
	:rank_group_(rank_group), parent_(parent), root_(root) {
	CHECK_EQ(parent, *MutThreadLocalRankGroupScope());
	*MutThreadLocalRankGroupScope() = this;
}

Maybe<void> RankGroupScope::SetRootSelf() {
	CHECK_ISNULL_OR_RETURN(parent_);
	CHECK_ISNULL_OR_RETURN(root_);
	root_ = this;
	return Maybe<void>::Ok();
}

RankGroupScope::~RankGroupScope() {
	CHECK_EQ(this, *MutThreadLocalRankGroupScope());
	*MutThreadLocalRankGroupScope() = this->parent_;
}

/*static*/ Maybe<RankGroupScope> RankGroupScope::MakeInitialRankGroupScope(Symbol<RankGroup> rank_group) {
	CHECK_ISNULL_OR_RETURN(*MutThreadLocalRankGroupScope());
	auto* ptr = new RankGroupScope(rank_group, nullptr, nullptr);
	JUST(ptr->SetRootSelf());
	return std::shared_ptr<RankGroupScope>(ptr); 
}

/*static*/ Maybe<RankGroupScope> RankGroupScope::MakeNestedRankGroupScope(Symbol<RankGroup> rank_group) {
	auto* parent = *MutThreadLocalRankGroupScope();
	CHECK_NOTNUL_OR_RETURN(parent);
	auto* root = &parent->root();
	auto* ptr = new RankGroupScope(rank_group, parent, root);
	return std::shared_ptr<RankGroupScope>(ptr); 
}

/*static*/ Maybe<Symbol<RankGroup>> RankGroupScope::CurrentRankGroup() {
	const RankGroupScope* scope = *MutThreadLocalRankGroupScope();
	CHECK_NOTNULL_OR_RETURN(scope);
	return scope->rank_group();
}

/*static*/ Maybe<Symbol<RankGroup>> RankGroupScope::RootRankGroup() {
	const RankGroupScope* scope = *MutThreadLocalRankGroupScope();
	CHECK_NOTNULL_OR_RETURN(scope);
	return scope->root().rank_group();
}

}
