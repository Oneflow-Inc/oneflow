// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Transform/TransformDialectExtension.h"

#include <deque>

#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/RecyclingAllocator.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// BEGIN copied from mlir/lib/Transforms/CSE.cpp
//===----------------------------------------------------------------------===//
namespace {
struct SimpleOperationInfo : public llvm::DenseMapInfo<Operation*> {
  static unsigned getHashValue(const Operation* opC) {
    return OperationEquivalence::computeHash(const_cast<Operation*>(opC),
                                             /*hashOperands=*/OperationEquivalence::directHashValue,
                                             /*hashResults=*/OperationEquivalence::ignoreHashValue,
                                             OperationEquivalence::IgnoreLocations);
  }
  static bool isEqual(const Operation* lhsC, const Operation* rhsC) {
    auto* lhs = const_cast<Operation*>(lhsC);
    auto* rhs = const_cast<Operation*>(rhsC);
    if (lhs == rhs) return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() || rhs == getTombstoneKey()
        || rhs == getEmptyKey())
      return false;

    // If op has no regions, operation equivalence w.r.t operands alone is
    // enough.
    if (lhs->getNumRegions() == 0 && rhs->getNumRegions() == 0) {
      return OperationEquivalence::isEquivalentTo(
          const_cast<Operation*>(lhsC), const_cast<Operation*>(rhsC),
          OperationEquivalence::exactValueMatch,
          /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations);
    }

    // If lhs or rhs does not have a single region with a single block, they
    // aren't CSEed for now.
    if (lhs->getNumRegions() != 1 || rhs->getNumRegions() != 1
        || !llvm::hasSingleElement(lhs->getRegion(0)) || !llvm::hasSingleElement(rhs->getRegion(0)))
      return false;

    // Compare the two blocks.
    Block& lhsBlock = lhs->getRegion(0).front();
    Block& rhsBlock = rhs->getRegion(0).front();

    // Don't CSE if number of arguments differ.
    if (lhsBlock.getNumArguments() != rhsBlock.getNumArguments()) return false;

    // Map to store `Value`s from `lhsBlock` that are equivalent to `Value`s
    // in `rhsBlock`. `Value`s from `lhsBlock` are the key.
    DenseMap<Value, Value> areEquivalentValues;
    for (auto bbArgs :
         llvm::zip(lhs->getRegion(0).getArguments(), rhs->getRegion(0).getArguments())) {
      areEquivalentValues[std::get<0>(bbArgs)] = std::get<1>(bbArgs);
    }

    // Helper function to get the parent operation.
    auto getParent = [](Value v) -> Operation* {
      if (auto blockArg = v.dyn_cast<BlockArgument>())
        return blockArg.getParentBlock()->getParentOp();
      return v.getDefiningOp()->getParentOp();
    };

    // Callback to compare if operands of ops in the region of `lhs` and `rhs`
    // are equivalent.
    auto checkEquivalent = [&](Value lhsValue, Value rhsValue) -> LogicalResult {
      if (lhsValue == rhsValue) return success();
      if (areEquivalentValues.lookup(lhsValue) == rhsValue) return success();
      return failure();
    };

    // Callback to compare if results of ops in the region of `lhs` and `rhs`
    // are equivalent.
    auto markEquivalent = [&](Value lhsResult, Value rhsResult) {
      if (getParent(lhsResult) == lhs && getParent(rhsResult) == rhs) {
        areEquivalentValues.insert({lhsResult, rhsResult});
      }
    };

    return OperationEquivalence::isEquivalentTo(
        const_cast<Operation*>(lhsC), const_cast<Operation*>(rhsC), checkEquivalent, markEquivalent,
        OperationEquivalence::IgnoreLocations);
  }
};
}  // namespace

namespace {
/// Simple common sub-expression elimination.
//===----------------------------------------------------------------------===//
// END copied from mlir/lib/Transforms/CSE.cpp
//===----------------------------------------------------------------------===//
/// Copy of CSE::runOnOperation, without the pass baggage.
// struct CSE : public impl::CSEBase<CSE> {
struct CSE {
  //===----------------------------------------------------------------------===//
  // BEGIN copied from mlir/lib/Transforms/CSE.cpp
  //===----------------------------------------------------------------------===//
  /// Shared implementation of operation elimination and scoped map
  /// definitions.
  using AllocatorTy = llvm::RecyclingAllocator<llvm::BumpPtrAllocator,
                                               llvm::ScopedHashTableVal<Operation*, Operation*>>;
  using ScopedMapTy =
      llvm::ScopedHashTable<Operation*, Operation*, SimpleOperationInfo, AllocatorTy>;

  /// Cache holding MemoryEffects information between two operations. The
  /// first operation is stored has the key. The second operation is stored
  /// inside a pair in the value. The pair also hold the MemoryEffects between
  /// those two operations. If the MemoryEffects is nullptr then we assume
  /// there is no operation with MemoryEffects::Write between the two
  /// operations.
  using MemEffectsCache = DenseMap<Operation*, std::pair<Operation*, MemoryEffects::Effect*>>;

  /// Represents a single entry in the depth first traversal of a CFG.
  struct CFGStackNode {
    CFGStackNode(ScopedMapTy& knownValues, DominanceInfoNode* node)
        : scope(knownValues), node(node), childIterator(node->begin()) {}

    /// Scope for the known values.
    ScopedMapTy::ScopeTy scope;

    DominanceInfoNode* node;
    DominanceInfoNode::const_iterator childIterator;

    /// If this node has been fully processed yet or not.
    bool processed = false;
  };

  /// Attempt to eliminate a redundant operation. Returns success if the
  /// operation was marked for removal, failure otherwise.
  LogicalResult simplifyOperation(ScopedMapTy& knownValues, Operation* op, bool hasSSADominance);
  void simplifyBlock(ScopedMapTy& knownValues, Block* bb, bool hasSSADominance);
  void simplifyRegion(ScopedMapTy& knownValues, Region& region);

  // void runOnOperation() override;
  void doItOnOperation(Operation* rootOp, DominanceInfo* domInfo, RewriterBase::Listener* listener);

 private:
  void replaceUsesAndDelete(ScopedMapTy& knownValues, Operation* op, Operation* existing,
                            bool hasSSADominance);

  /// Check if there is side-effecting operations other than the given effect
  /// between the two operations.
  bool hasOtherSideEffectingOpInBetween(Operation* fromOp, Operation* toOp);

  /// Operations marked as dead and to be erased.
  std::vector<Operation*> opsToErase;
  DominanceInfo* domInfo = nullptr;
  MemEffectsCache memEffectsCache;
  //===----------------------------------------------------------------------===//
  // END copied from mlir/lib/Transforms/CSE.cpp
  //===----------------------------------------------------------------------===//
  /// An optional listener to notify of replaced or erased operations.
  RewriterBase::Listener* listener;
  int64_t numDCE = 0, numCSE = 0;

  //===----------------------------------------------------------------------===//
  // BEGIN copied from mlir/lib/Transforms/CSE.cpp
  //===----------------------------------------------------------------------===//
};
}  // namespace

void CSE::replaceUsesAndDelete(ScopedMapTy& knownValues, Operation* op, Operation* existing,
                               bool hasSSADominance) {
  // If we find one then replace all uses of the current operation with the
  // existing one and mark it for deletion. We can only replace an operand in
  // an operation if it has not been visited yet.
  if (hasSSADominance) {
    // If the region has SSA dominance, then we are guaranteed to have not
    // visited any use of the current operation.
    //===----------------------------------------------------------------------===//
    // END copied from mlir/lib/Transforms/CSE.cpp
    //===----------------------------------------------------------------------===//
    if (listener) listener->notifyOperationReplaced(op, existing->getResults());
    //===----------------------------------------------------------------------===//
    // BEGIN copied from mlir/lib/Transforms/CSE.cpp
    //===----------------------------------------------------------------------===//
    op->replaceAllUsesWith(existing);
    opsToErase.push_back(op);
  } else {
    // When the region does not have SSA dominance, we need to check if we
    // have visited a use before replacing any use.
    for (auto it : llvm::zip(op->getResults(), existing->getResults())) {
      std::get<0>(it).replaceUsesWithIf(std::get<1>(it), [&](OpOperand& operand) {
        return !knownValues.count(operand.getOwner());
      });
    }

    // There may be some remaining uses of the operation.
    if (op->use_empty()) opsToErase.push_back(op);
  }

  // If the existing operation has an unknown location and the current
  // operation doesn't, then set the existing op's location to that of the
  // current op.
  if (existing->getLoc().isa<UnknownLoc>() && !op->getLoc().isa<UnknownLoc>())
    existing->setLoc(op->getLoc());

  ++numCSE;
}

bool CSE::hasOtherSideEffectingOpInBetween(Operation* fromOp, Operation* toOp) {
  assert(fromOp->getBlock() == toOp->getBlock());
  assert(isa<MemoryEffectOpInterface>(fromOp)
         && cast<MemoryEffectOpInterface>(fromOp).hasEffect<MemoryEffects::Read>()
         && isa<MemoryEffectOpInterface>(toOp)
         && cast<MemoryEffectOpInterface>(toOp).hasEffect<MemoryEffects::Read>());
  Operation* nextOp = fromOp->getNextNode();
  auto result = memEffectsCache.try_emplace(fromOp, std::make_pair(fromOp, nullptr));
  if (result.second) {
    auto memEffectsCachePair = result.first->second;
    if (memEffectsCachePair.second == nullptr) {
      // No MemoryEffects::Write has been detected until the cached operation.
      // Continue looking from the cached operation to toOp.
      nextOp = memEffectsCachePair.first;
    } else {
      // MemoryEffects::Write has been detected before so there is no need to
      // check further.
      return true;
    }
  }
  while (nextOp && nextOp != toOp) {
    auto nextOpMemEffects = dyn_cast<MemoryEffectOpInterface>(nextOp);
    // TODO: Do we need to handle other effects generically?
    // If the operation does not implement the MemoryEffectOpInterface we
    // conservatively assumes it writes.
    if ((nextOpMemEffects && nextOpMemEffects.hasEffect<MemoryEffects::Write>())
        || !nextOpMemEffects) {
      result.first->second = std::make_pair(nextOp, MemoryEffects::Write::get());
      return true;
    }
    nextOp = nextOp->getNextNode();
  }
  result.first->second = std::make_pair(toOp, nullptr);
  return false;
}

/// Attempt to eliminate a redundant operation.
LogicalResult CSE::simplifyOperation(ScopedMapTy& knownValues, Operation* op,
                                     bool hasSSADominance) {
  // Don't simplify terminator operations.
  if (op->hasTrait<OpTrait::IsTerminator>()) return failure();

  // If the operation is already trivially dead just add it to the erase list.
  if (isOpTriviallyDead(op)) {
    opsToErase.push_back(op);
    ++numDCE;
    return success();
  }

  // Don't simplify operations with nested blocks. We don't currently model
  // equality comparisons correctly among other things. It is also unclear
  // whether we would want to CSE such operations.
  if (!(op->getNumRegions() == 0
        || (op->getNumRegions() == 1 && llvm::hasSingleElement(op->getRegion(0)))))
    return failure();

  // Some simple use case of operation with memory side-effect are dealt with
  // here. Operations with no side-effect are done after.
  if (!isMemoryEffectFree(op)) {
    auto memEffects = dyn_cast<MemoryEffectOpInterface>(op);
    // TODO: Only basic use case for operations with MemoryEffects::Read can
    // be eleminated now. More work needs to be done for more complicated
    // patterns and other side-effects.
    if (!memEffects || !memEffects.onlyHasEffect<MemoryEffects::Read>()) return failure();

    // Look for an existing definition for the operation.
    if (auto* existing = knownValues.lookup(op)) {
      if (existing->getBlock() == op->getBlock()
          && !hasOtherSideEffectingOpInBetween(existing, op)) {
        // The operation that can be deleted has been reach with no
        // side-effecting operations in between the existing operation and
        // this one so we can remove the duplicate.
        replaceUsesAndDelete(knownValues, op, existing, hasSSADominance);
        return success();
      }
    }
    knownValues.insert(op, op);
    return failure();
  }

  // Look for an existing definition for the operation.
  if (auto* existing = knownValues.lookup(op)) {
    replaceUsesAndDelete(knownValues, op, existing, hasSSADominance);
    ++numCSE;
    return success();
  }

  // Otherwise, we add this operation to the known values map.
  knownValues.insert(op, op);
  return failure();
}

void CSE::simplifyBlock(ScopedMapTy& knownValues, Block* bb, bool hasSSADominance) {
  for (auto& op : *bb) {
    // If the operation is simplified, we don't process any held regions.
    if (succeeded(simplifyOperation(knownValues, &op, hasSSADominance))) continue;

    // Most operations don't have regions, so fast path that case.
    if (op.getNumRegions() == 0) continue;

    // If this operation is isolated above, we can't process nested regions
    // with the given 'knownValues' map. This would cause the insertion of
    // implicit captures in explicit capture only regions.
    if (op.mightHaveTrait<OpTrait::IsIsolatedFromAbove>()) {
      ScopedMapTy nestedKnownValues;
      for (auto& region : op.getRegions()) simplifyRegion(nestedKnownValues, region);
      continue;
    }

    // Otherwise, process nested regions normally.
    for (auto& region : op.getRegions()) simplifyRegion(knownValues, region);
  }
  // Clear the MemoryEffects cache since its usage is by block only.
  memEffectsCache.clear();
}

void CSE::simplifyRegion(ScopedMapTy& knownValues, Region& region) {
  // If the region is empty there is nothing to do.
  if (region.empty()) return;

  bool hasSSADominance = domInfo->hasSSADominance(&region);

  // If the region only contains one block, then simplify it directly.
  if (region.hasOneBlock()) {
    ScopedMapTy::ScopeTy scope(knownValues);
    simplifyBlock(knownValues, &region.front(), hasSSADominance);
    return;
  }

  // If the region does not have dominanceInfo, then skip it.
  // TODO: Regions without SSA dominance should define a different
  // traversal order which is appropriate and can be used here.
  if (!hasSSADominance) return;

  // Note, deque is being used here because there was significant performance
  // gains over vector when the container becomes very large due to the
  // specific access patterns. If/when these performance issues are no
  // longer a problem we can change this to vector. For more information see
  // the llvm mailing list discussion on this:
  // http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20120116/135228.html
  std::deque<std::unique_ptr<CFGStackNode>> stack;

  // Process the nodes of the dom tree for this region.
  stack.emplace_back(std::make_unique<CFGStackNode>(knownValues, domInfo->getRootNode(&region)));

  while (!stack.empty()) {
    auto& currentNode = stack.back();

    // Check to see if we need to process this node.
    if (!currentNode->processed) {
      currentNode->processed = true;
      simplifyBlock(knownValues, currentNode->node->getBlock(), hasSSADominance);
    }

    // Otherwise, check to see if we need to process a child node.
    if (currentNode->childIterator != currentNode->node->end()) {
      auto* childNode = *(currentNode->childIterator++);
      stack.emplace_back(std::make_unique<CFGStackNode>(knownValues, childNode));
    } else {
      // Finally, if the node and all of its children have been processed
      // then we delete the node.
      stack.pop_back();
    }
  }
}

//===----------------------------------------------------------------------===//
// END copied from mlir/lib/Transforms/CSE.cpp
//===----------------------------------------------------------------------===//

/// Copy of CSE::runOnOperation, without the pass baggage.
void CSE::doItOnOperation(Operation* rootOp, DominanceInfo* domInfo,
                          RewriterBase::Listener* listener) {
  /// A scoped hash table of defining operations within a region.
  ScopedMapTy knownValues;
  this->domInfo = domInfo;
  this->listener = listener;

  for (auto& region : rootOp->getRegions()) simplifyRegion(knownValues, region);

  /// Erase any operations that were marked as dead during simplification.
  for (auto* op : opsToErase) {
    if (listener) listener->notifyOperationRemoved(op);
    op->erase();
  }
  opsToErase.clear();
}

/// Run CSE on the provided operation
LogicalResult
mlir::eliminateCommonSubexpressions(Operation *op, DominanceInfo *domInfo,
                                    RewriterBase::Listener *listener) {
  assert(op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         "can only do CSE on isolated-from-above ops");
  std::optional<DominanceInfo> defaultDomInfo;
  if (domInfo == nullptr) {
    defaultDomInfo.emplace(op);
    domInfo = &*defaultDomInfo;
  }
  CSE().doItOnOperation(op, domInfo, listener);
  return success();
}

