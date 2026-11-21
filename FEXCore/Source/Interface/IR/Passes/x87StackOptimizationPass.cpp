// SPDX-License-Identifier: MIT
#include "FEXCore/Utils/LogManager.h"
#include "Interface/Core/Interpreter/Fallbacks/FallbackOpHandler.h"
#include "Interface/IR/IR.h"
#include "Interface/IR/IREmitter.h"
#include "Interface/IR/Passes.h"
#include <fexc_hook.h>
#include <fexc_precision.h>
#include "Interface/IR/PassManager.h"
#include "FEXCore/IR/IR.h"
#include "FEXCore/Utils/Profiler.h"
#include "FEXCore/Core/HostFeatures.h"
#include "Interface/Core/Addressing.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdint.h>

// This file adds a pass to process X87 stack instructions.
// These instructions are marked in IR.json with `X87: true` and are generated
// by X87 guest instructions.
// The way is works is that there's a virtual stack `StackData`, where we load and store
// and apply the operations in a block of code. Once the block finishes, we emit the necessary operations
// that we recorded onto the virtual stack. This allows us to save a lot of code movement
// to and from stack registers, top management and valid flags. It also allows us to
// perform memcpy optimizations like the one performed in STORESTACKMEM.
//
// By default we run on the fast path - i.e. we assume all values are in the stack and we have a complete
// stack overview. However, if we encounter a value that's not in the virtual stack - maybe it was added
// to the stack in a previous block, we move onto the slow path which loads and stores values to the stack
// registers.
// Once in a slow path, we won't return to the fast pass until the beginning of the following block.

namespace FEXCore::IR {

// FIXME(pmatos): copy from OpcodeDispatcher.h
static uint32_t MMBaseOffset() {
  return static_cast<uint32_t>(offsetof(Core::CPUState, mm[0][0]));
}

// Similar helper to the one in OpcodeDispatcher.h except we do not
// need to handle flags, etc.
template<typename T>
static void DeriveOp(Ref& RefV, IROps NewOp, IREmitter::IRPair<T> Expr) {
  Expr.first->Header.Op = NewOp;
  RefV = Expr;
}

enum class StackSlot { UNUSED, INVALID, VALID };
// FixedSizeStack is a model of the x87 Stack where each element in this
// fixed size stack lives at an offset from top. The top of the stack is at
// index 0.
template<typename T>
class FixedSizeStack {
public:
  struct StackSlotEntry final {
    StackSlot Type = StackSlot::UNUSED;
    T Value = T::Invalid;
  };

  static constexpr uint8_t size = 8;

  // Real top as an offset from stored top value (or the one at the beginning of the block)
  // For example, if we start and push a value to our simulated stack, because we don't
  // update top straight away the TopOffset is 1.
  // If SlowPath is true, then TopOffset is always zero.
  int8_t TopOffset = 0;

  FixedSizeStack() = default;

  void push(const T& Value) {
    rotate();
    buffer.front() = {StackSlot::VALID, Value};
  }

  // Rotate the elements with the direction controlled by Right
  void rotate(bool Right = true) {
    if (Right) {
      std::rotate(buffer.begin(), buffer.end() - 1, buffer.end());
      TopOffset++;
    } else {
      std::rotate(buffer.begin(), buffer.begin() + 1, buffer.end());
      TopOffset--;
    }
  }

  void pop() {
    buffer.front() = {StackSlot::INVALID, T::Invalid};
    rotate(false);
  }

  const StackSlotEntry& top(size_t Offset = 0) const {
    return buffer[Offset];
  }

  void setTop(const T& Value, size_t Offset = 0) {
    buffer[Offset] = {StackSlot::VALID, Value};
  }

  bool isValid(size_t Offset) const {
    return buffer[Offset].Type == StackSlot::VALID;
  }

  void clear() {
    buffer.fill({StackSlot::UNUSED, T::Invalid});
    TopOffset = 0;
  }

  void dump() const {
    LogMan::Msg::DFmt("-- Stack");

    for (size_t i = 0; i < buffer.size(); i++) {
      const auto& [Valid, Element] = buffer[i];
      if (Valid == StackSlot::VALID) {
        LogMan::Msg::DFmt("| ST{}: 0x{:x}", i, (uintptr_t)(Element.StackDataNode));
      } else if (Valid == StackSlot::INVALID) {
        LogMan::Msg::DFmt("| ST{}: INVALID", i);
      }
    }
    LogMan::Msg::DFmt("--");
  }

  void setTagInvalid(size_t Index) {
    buffer[Index].Type = StackSlot::INVALID;
  }

  // Returns a mask to set in AbridgedTagWord
  uint8_t getValidMask() const {
    uint8_t Mask = 0;
    for (size_t i = 0; i < buffer.size(); i++) {
      if (buffer[i].Type == StackSlot::VALID) {
        Mask |= 1U << i;
      }
    }
    return Mask;
  }

  // Returns a mask to set in AbridgedTagWord
  uint8_t getInvalidMask() const {
    uint8_t Mask = 0;
    for (size_t i = 0; i < buffer.size(); i++) {
      if (buffer[i].Type == StackSlot::INVALID) {
        Mask |= 1U << i;
      }
    }
    return Mask;
  }

private:
  std::array<StackSlotEntry, size> buffer {};
};

class X87StackOptimization final : public Pass {
public:
  X87StackOptimization(const FEXCore::HostFeatures& Features, OpSize GPROpSize)
    : Features(Features)
    , GPROpSize(GPROpSize) {
    FEX_CONFIG_OPT(ReducedPrecision, X87REDUCEDPRECISION);
    ReducedPrecisionMode = ReducedPrecision;
  }
  void Run(IREmitter* Emit) override;

private:
  const FEXCore::HostFeatures& Features;
  const OpSize GPROpSize;
  bool ReducedPrecisionMode;
  FEX_CONFIG_OPT(DisableVixlIndirectCalls, DISABLE_VIXL_INDIRECT_RUNTIME_CALLS);

  // Helpers
  Ref RotateRight8(uint32_t V, Ref Amount);

  void F80SplitStore_Helper(const IROp_StoreStackMem* Op, Ref StackNode, Ref AddrNode, Ref Offset, OpSize Align, MemOffsetType OffsetType,
                            uint8_t OffsetScale) {
    IREmit->_StoreMemFPR(OpSize::i64Bit, StackNode, AddrNode, Offset, Align, OffsetType, OffsetScale);
    auto Upper = IREmit->_VExtractToGPR(OpSize::i128Bit, OpSize::i64Bit, StackNode, 1);

    // Store the Upper part of the register (the remaining 2 bytes) into memory.
    AddressMode A {.Base = AddrNode,
                   .Index = Op->Offset.IsInvalid() ? nullptr : Offset,
                   .Offset = 8,
                   .IndexType = MemOffsetType::SXTX,
                   .IndexScale = OffsetScale,
                   .AddrSize = OpSize::i64Bit};
    A = SelectAddressMode(IREmit, A, GPROpSize, Features.SupportsTSOImm9, false, false, OpSize::i16Bit);
    IREmit->_StoreMemGPR(OpSize::i16Bit, Upper, A.Base, A.Index, OpSize::i64Bit, MemOffsetType::SXTX, A.IndexScale);
  }

  void Store80BitToMem(const IROp_StoreStackMem* Op, Ref StackNode, Ref AddrNode, Ref Offset, OpSize Align, MemOffsetType OffsetType,
                       uint8_t OffsetScale) {
    if (Features.SupportsSVE()) {
      AddressMode A {.Base = AddrNode,
                     .Index = Op->Offset.IsInvalid() ? nullptr : Offset,
                     .IndexType = MemOffsetType::SXTX,
                     .IndexScale = OffsetScale,
                     .AddrSize = OpSize::i64Bit};
      AddrNode = LoadEffectiveAddress(IREmit, A, GPROpSize, false);
      IREmit->_StoreMemX87SVEOptPredicate(OpSize::i128Bit, OpSize::i16Bit, StackNode, AddrNode);
    } else {
      F80SplitStore_Helper(Op, StackNode, AddrNode, Offset, Align, OffsetType, OffsetScale);
    }
  }

  void StoreStackMem_Helper(const IRListView& IR, const IROp_StoreStackMem* Op, Ref StackNode) {
    LOGMAN_THROW_A_FMT(!ReducedPrecisionMode, "Full precision mode expected.");

    Ref AddrNode = IR.GetNode(Op->Addr);
    Ref Offset = IR.GetNode(Op->Offset);
    OpSize Align = Op->Align;
    MemOffsetType OffsetType = Op->OffsetType;
    uint8_t OffsetScale = Op->OffsetScale;

    // Normal Precision Mode
    switch (Op->StoreSize) {
    case OpSize::i32Bit:
    case OpSize::i64Bit: {
      StackNode = IREmit->_F80CVT(Op->StoreSize, StackNode);
      IREmit->_StoreMemFPR(Op->StoreSize, StackNode, AddrNode, Offset, Align, OffsetType, OffsetScale);
      break;
    }

    case OpSize::f80Bit: {
      Store80BitToMem(Op, StackNode, AddrNode, Offset, Align, OffsetType, OffsetScale);
      break;
    }
    default: ERROR_AND_DIE_FMT("Unsupported x87 size");
    }
  }

  // Performs a store to memory from a value the stack passed in as StackNode.
  // This is the version dealing with the reduced precision case.
  void StoreStackMem_Reduced_Helper(const IRListView& IR, const IROp_StoreStackMem* Op, Ref StackNode) {
    LOGMAN_THROW_A_FMT(ReducedPrecisionMode, "Reduced precision mode expected.");

    Ref AddrNode = IR.GetNode(Op->Addr);
    Ref Offset = IR.GetNode(Op->Offset);
    OpSize Align = Op->Align;
    MemOffsetType OffsetType = Op->OffsetType;
    uint8_t OffsetScale = Op->OffsetScale;

    switch (Op->StoreSize) {
    case OpSize::i32Bit: {
      StackNode = IREmit->_Float_FToF(OpSize::i32Bit, OpSize::i64Bit, StackNode);
      [[fallthrough]];
    }
    case OpSize::i64Bit: {
      // natemu slice1 (I6); M2 task 4: but a value this block computed, which is no signalling NaN
      if (fexc_slice1 && Op->StoreSize == OpSize::i64Bit && !(FexcCache() && TopQuiet[0])) StackNode = QuietSNaN(StackNode);
      if (fexc_slice1 && Op->StoreSize == OpSize::i64Bit && (fexc_x87_unmasked & 16)) {
        // natemu slice1 (I5c): a nonzero double denormal stored to m64 is tiny (exact): x86's unmasked UE (SDM Vol.1 8.5.5).
        // natemu slice1 (I5c review): tiny = the magnitude H = Bits << 1 is nonzero and its exponent field H >> 53 is zero (DBL_MIN
        // is not tiny), flag-free: ((H >> 53) - 1) >> 63 and (H | -H) >> 63.
        Ref Bits = IREmit->_VExtractToGPR(OpSize::i64Bit, OpSize::i64Bit, StackNode, 0);
        Ref H = IREmit->_Lshl(OpSize::i64Bit, Bits, GetConstant(1));
        Ref ExpZero = IREmit->_Lshr(OpSize::i64Bit, IREmit->_Sub(OpSize::i64Bit, IREmit->_Lshr(OpSize::i64Bit, H, GetConstant(53)), GetConstant(1)), GetConstant(63));
        Ref NonZero = IREmit->_Lshr(OpSize::i64Bit, IREmit->_Or(OpSize::i64Bit, H, IREmit->_Neg(OpSize::i64Bit, H)), GetConstant(63));
        Tiny = IREmit->_And(OpSize::i64Bit, ExpZero, NonZero);
      }
      IREmit->_StoreMemFPR(Op->StoreSize, StackNode, AddrNode, Offset, Align, OffsetType, OffsetScale);
      break;
    }

    case OpSize::f80Bit: {
      StackNode = IREmit->_F80CVTTo(StackNode, OpSize::i64Bit);
      Store80BitToMem(Op, StackNode, AddrNode, Offset, Align, OffsetType, OffsetScale);
      break;
    }
    default: ERROR_AND_DIE_FMT("Unsupported x87 size");
    }
  }

  // Handles a Unary operation.
  // Takes the op we are handling, the Node for the reduced precision case and the node for the normal case.
  // Depending on the type of Op64, we might need to pass a couple of extra constant arguments, this happens
  // when VFOp64 is true.
  void HandleUnop(IROps Op64, bool VFOp64, IROps Op80);
  void HandleBinopValue(IROps Op64, bool VFOp64, IROps Op80, uint8_t DestStackOffset, bool MarkDestValid, uint8_t StackOffset,
                        Ref ValueNode, bool Reverse = false);
  void HandleBinopStack(IROps Op64, bool VFOp64, IROps Op80, uint8_t DestStackOffset, uint8_t StackOffset1, uint8_t StackOffset2,
                        bool Reverse = false);

  // Top Management Helpers
  /// Set the valid tag for Value as valid (if Valid is true), or invalid (if Valid is false).
  void SetX87ValidTag(uint8_t Offset, bool Valid);
  // Generates slow code to load/store a value from an offset from the top of the stack
  Ref LoadStackValueAtOffset_Slow(uint8_t Offset = 0);
  // natemu slice1 (I6): x87 stack faults with their masked response (SDM Vol.1 8.5.1.1; every exception is masked): a read of
  // an empty register gives the QNaN floating-point indefinite (stack underflow), a push onto a non-empty one loads it (stack
  // overflow); the instruction's IE and SF are set and C1 = 1 for an overflow, 0 for an underflow, at the next instruction's
  // start (after any store of this one: a fault there leaves FSW as it was). Fault/Overflow accumulate over one instruction.
  Ref StackChecked(Ref Value, uint8_t Offset, bool Push);
  void FlushStackFault();
  Ref Fault {}, Overflow {};
  bool CheckReads = true;
  // natemu slice1 (I6): the current op's own stack fault (its arithmetic result is then the indefinite, whatever the other
  // operand), an SNaN an FST m64 quieted (IE), and whether the instruction defines C1 itself: a push (C1 = overflow) or
  // FCHS/FABS (C1 = 0).
  Ref OpFault {}, SNaN {};
  bool Pushed = false, C1Zero = false;
  // natemu slice1 (M2 task 4): the register, TOP and tag caches live across the x87 instructions of an IR block while every
  // x87 exception is masked (FexcCache), not per op. What is pending is written to the guest-state page before every
  // instruction that is not a register-only x87 form (fexc_x87_quiet: it has no fault site, stub or helper), before a helper
  // call and before any other op that reads or writes the x87 stack state there, so each of them finds the page complete.
  // TopBad: the check bit of a cached register whose tag was read at run time by the inline check (a later read of the
  // cached value in the same instruction faults alike; the next instruction reads and checks again);
  // TagDirty: the tags to write (the others are known, not changed). A tag known in the block needs no check at all.
  // The caches hold scratch registers (a value each, a general register per TOP + i and one for the tag byte), and slice1
  // admits no spill: past a budget (64-bit code has half the scratch registers of 32-bit code) everything is written at
  // the instruction's start and the registers are given up; what the block knows of the tags stays.
  bool FexcCache() const { return fexc_slice1 && !fexc_step_unit && uint64_t(fexc_x87_unmasked) == 0; }
  std::array<Ref, 8> TopBad {};
  std::array<bool, 8> TagDirty {};
  void NoteFault(Ref Bad, bool Push);
  // M2 task 4, lazy flags (FexcX87Lazy, fexc_hook.h): an instruction with a stack check gets FexcX87Hold at its start (its
  // GuestOpcode node) and FexcX87Release at its end, which drops its own FPSR flags after a stack fault.
  Ref FexcOpcode {};
  bool Held = false;
  // M2 task 4, the stack check as a stub (FexcX87Tag): where the caches live across instructions (FexcCache) a tag that is not
  // known is tested at the instruction's start, before anything the instruction does, and a stack fault leaves to the
  // resident, which steps the instruction alone (the one-instruction unit has the masked response below, as before). The
  // test passed, the tag is known for the rest of the block. Mutated: the instruction changed the x87 stack state already
  // (no check follows a change in any form; if one did it would be the inline one). Boundary*: TOP's registers and the tag
  // byte as they were at the instruction's start, which the test and a late flush before the instruction may use.
  bool Mutated = false, BoundaryFlushed = false;
  std::array<Ref, 8> BoundaryTop {};
  Ref BoundaryFTW {}, StubCursor {};   // StubCursor: the instruction's last test (the next one follows it)
  bool TagStub(uint8_t Offset, bool Push);
  void FexcBoundary(bool Flush);
  // M2 task 4: what the block knows without a run-time test. C1Read: the instruction read a stack register (C1 = 0 at its
  // end); C1Known0: the page's C1 is 0 already (nothing wrote it since this pass stored 0). TopQuiet: the cached register
  // holds the result of a floating-point op of this block, which is never a signalling NaN (FST m64 need not quiet it).
  bool C1Read = false, C1Known0 = false;
  std::array<bool, 8> TopQuiet {};
  bool FexcQuietValue(const IRListView& IR, Ref Node) const;
  bool FexcX87State(const IROp_Header* IROp) const;
  Ref FaultResult(Ref Value);
  Ref QuietSNaN(Ref Value);
  // natemu slice1 (I5c): the guest RIP of the instruction being lowered, an FST m64's exact-tiny bit (an unmasked UE stores
  // nothing), and the end of an instruction: its check operands recorded, its flags folded (FexcEndInstruction).
  uint64_t FexcRip = ~0ull;
  Ref Tiny {};
  void FexcEndInstruction();
  void StoreStackValueAtOffset_Slow(Ref Value, uint8_t Offset = 0, bool SetValid = true);
  // Update Top value in slow path for a pop
  void UpdateTopForPop_Slow(bool InvalidateTag = true);
  void UpdateTopForPush_Slow();
  // Synchronizes the current simulated stack with the actual values.
  // Returns a new value for Top, that's synchronized between the simulated stack
  // and the actual FPU stack.
  Ref SynchronizeStackValues();
  // Moves us from the fast to the slow path if ShouldMigrate is true.
  void MigrateToSlowPathIf(bool ShouldMigrate);
  // Top Cache Management
  Ref GetTopWithCache_Slow();
  Ref GetOffsetTopWithCache_Slow(uint8_t Offset, bool Reverse = false);
  Ref GetOffsetTopAddressWithCache_Slow(uint8_t Offset);
  void SetTopWithCache_Slow(Ref Value);
  Ref GetX87ValidTag_Slow(uint8_t Offset);
  // Resets fields to initial values
  void Reset();

  struct StackMemberInfo {
    constexpr StackMemberInfo() = default;
    constexpr StackMemberInfo(Ref Data)
      : StackDataNode(Data) {}
    constexpr StackMemberInfo(Ref Data, Ref Source, OpSize Size)
      : StackDataNode(Data)
      , Source({Size, Source}) {}
    Ref StackDataNode {}; // Reference to the data in the Stack.
                          // This is the source data node in the stack format, possibly converted to 64/80 bits.
    struct StackMemberData final {
      OpSize Size;
      Ref Node;
    };

    static const StackMemberInfo Invalid;

    // Tuple is only valid if we have information about the Source of the Stack Data Node.
    // In it's valid then OpSize is the original source size and Ref is the original source node.
    std::optional<StackMemberData> Source {};
  };

  // StackData, TopCache need to be always properly set to ensure
  // they reflect the current state of the FPU. This sync only makes sense while
  // taking the fast path. Once in the slow path, these don't make sense anymore
  // and we are syncing everything.

  // Index on vector is offset to top value at start of block
  // If slow path is true, then StackData is always empty.
  FixedSizeStack<StackMemberInfo> StackData;

  void InvalidateCaches();
  void InvalidateCachedRegs();

  // Path Migration helper management
  std::optional<StackMemberInfo> MigrateToSlowPath_IfInvalid(uint8_t Offset = 0);
  Ref LoadStackValue(uint8_t Offset = 0);
  void StoreStackValue(Ref Value, uint8_t Offset = 0, bool SetValid = false);
  void StackPop();

  // Cache for Constants
  // ConstantPoll[i] has IREmit->_Constant(i);
  std::array<Ref, 8> ConstantPool {};
  Ref GetConstant(ssize_t Offset);

  // Cached value for Top
  // If slowpath is false, then TopCache is nullptr.
  bool FlushTopPending = false;
  std::array<bool, 8> FlushValuesPending {};
  bool FlushValidPending = false;
  void FlushCachedRegs();

  Ref GetFTW();

  Ref FTWCached {};
  std::array<Ref, 8> TopOffsetCache {};
  std::array<Ref, 8> TopOffsetAddressCache {};
  std::array<Ref, 8> TopValueCache {};
  std::array<StackSlot, 8> TopValidCache {};

  // Are we on the slow path?
  // Once we enter the slow path, we never come out.
  // This just simplifies the code atm. If there's a need to return to the fast path in the future
  // we can implement that but I would expect that there would be very few cases where that's necessary.
  // On the slow path TopCache is always the last obtained version of top.
  // TopOffset is ignored
  bool SlowPath = false;

  // Keeping IREmitter not to pass arguments around
  IREmitter* IREmit = nullptr;
};

inline const X87StackOptimization::StackMemberInfo X87StackOptimization::StackMemberInfo::Invalid {nullptr};

inline void X87StackOptimization::InvalidateCaches() {
  InvalidateCachedRegs();
  ConstantPool.fill(nullptr);
}

inline void X87StackOptimization::InvalidateCachedRegs() {
  FlushCachedRegs();
  FTWCached = {};
  TopOffsetCache.fill(nullptr);
  TopOffsetAddressCache.fill(nullptr);
  TopValueCache.fill(nullptr);
  TopValidCache.fill(StackSlot::UNUSED);
  TopBad.fill(nullptr);
  TagDirty.fill(false);
  TopQuiet.fill(false);
}

inline void X87StackOptimization::NoteFault(Ref Bad, bool Push) {
  if (FexcX87Lazy() && !Held && FexcOpcode) {
    Ref Cursor = IREmit->GetWriteCursor();
    IREmit->SetWriteCursor(FexcOpcode);
    IREmit->_FexcX87Hold();
    IREmit->SetWriteCursor(Cursor);
    Held = true;
  }
  Fault = Fault ? IREmit->_Or(OpSize::i32Bit, Fault, Bad) : Bad;
  OpFault = OpFault ? IREmit->_Or(OpSize::i32Bit, OpFault, Bad) : Bad;
  if (Push) {
    Overflow = Overflow ? IREmit->_Or(OpSize::i32Bit, Overflow, Bad) : Bad;
  }
}

inline void X87StackOptimization::FexcBoundary(bool Flush) {
  // The page is complete before an instruction that can fault, stop or call out (FexcCache's comment).
  const bool Wide = GPROpSize == OpSize::i64Bit;
  const auto Count = [](const auto& Cache) { return size_t(std::count_if(Cache.begin(), Cache.end(), [](Ref R) { return R != nullptr; })); };
  // The register budget (FexcCache's comment): general registers for TOP + i and the tag byte, vector registers for values.
  if (FexcCache() && (Count(TopOffsetCache) + (FTWCached ? 1 : 0) > (Wide ? 3 : 9) || Count(TopValueCache) > (Wide ? 5 : 8))) {
    FlushCachedRegs();
    const Ref Top = TopOffsetCache[0];
    TopOffsetCache.fill(nullptr);
    TopOffsetCache[0] = Top;
    TopValueCache.fill(nullptr);
    TopBad.fill(nullptr);
    Flush = true;
  }
  if (Flush) {
    FlushCachedRegs();
  } else if (FexcCache()) {
    // Left pending: what a flush before this instruction would use exists already (TagStub may ask for one later).
    bool Tags = false;
    for (size_t i = 0; i < TagDirty.size(); i++) {
      if (FlushValuesPending[i] || TagDirty[i]) GetOffsetTopWithCache_Slow(i);
      Tags |= TagDirty[i];
    }
    if (Tags) {
      GetFTW();
      GetTopWithCache_Slow();
    }
  }
  BoundaryFlushed = Flush;
  StubCursor = {};
  Mutated = false;
  BoundaryTop = TopOffsetCache;
  BoundaryFTW = FTWCached;
  ConstantPool.fill(nullptr);   // (a constant is not worth a register across instructions)
}

inline bool X87StackOptimization::TagStub(uint8_t Offset, bool Push) {
  if (!FexcCache() || !FexcOpcode || Mutated) {
    return false;
  }
  Ref Cursor = IREmit->GetWriteCursor();
  if (!BoundaryFlushed) {
    ConstantPool.fill(nullptr);
    IREmit->SetWriteCursorBefore(FexcOpcode);
    FlushCachedRegs();
    BoundaryFlushed = true;
    BoundaryFTW = FTWCached;
    ConstantPool.fill(nullptr);
  }
  IREmit->SetWriteCursor(StubCursor ? StubCursor : FexcOpcode);
  if (!BoundaryTop[0]) {
    BoundaryTop[0] = IREmit->_And(OpSize::i32Bit, IREmit->_LoadContextGPR(OpSize::i8Bit, offsetof(FEXCore::Core::CPUState, flags) + FEXCore::X86State::X87FLAG_TOP_LOC), IREmit->_Constant(7));
    if (!TopOffsetCache[0]) TopOffsetCache[0] = BoundaryTop[0];
  }
  if (!BoundaryTop[Offset]) {
    BoundaryTop[Offset] = IREmit->_And(OpSize::i32Bit, IREmit->Add(OpSize::i32Bit, BoundaryTop[0], Offset), IREmit->_Constant(7));
    if (!TopOffsetCache[Offset]) TopOffsetCache[Offset] = BoundaryTop[Offset];
  }
  if (!BoundaryFTW) {
    BoundaryFTW = IREmit->_LoadContextGPR(OpSize::i8Bit, offsetof(FEXCore::Core::CPUState, AbridgedFTW));
    if (!FTWCached) FTWCached = BoundaryFTW;
  }
  StubCursor = IREmit->_FexcX87Tag(BoundaryFTW, BoundaryTop[Offset], Push);
  IREmit->SetWriteCursor(Cursor);
  return true;
}

inline bool X87StackOptimization::FexcQuietValue(const IRListView& IR, Ref Node) const {
  const auto* Op = IR.GetOp<IROp_Header>(Node);
  switch (Op->Op) {
  case OP_VFADD: case OP_VFSUB: case OP_VFMUL: case OP_VFDIV: case OP_VFSQRT: case OP_FLOAT_FTOF: case OP_FLOAT_FROMGPR_S: return true;
  case OP_VCASTFROMGPR: {
    const auto* C = IR.GetOp<IROp_Header>(Op->Args[0]);
    if (C->Op != OP_CONSTANT) return false;
    const uint64_t V = C->C<IROp_Constant>()->Constant, M = V & 0x7fff'ffff'ffff'ffffULL;
    return !(M > 0x7ff0'0000'0000'0000ULL && !(V >> 51 & 1));
  }
  default: return false;
  }
}

// An op of the dispatcher (not this pass's) that reads or writes the x87 stack state in the guest-state page: TOP, the tags, mm[].
inline bool X87StackOptimization::FexcX87State(const IROp_Header* IROp) const {
  uint32_t At, Size;
  switch (IROp->Op) {
  case OP_LOADCONTEXT: At = IROp->C<IROp_LoadContext>()->Offset; Size = IR::OpSizeToSize(IROp->Size); break;
  case OP_STORECONTEXT: At = IROp->C<IROp_StoreContext>()->Offset; Size = IR::OpSizeToSize(IROp->Size); break;
  case OP_LOADCONTEXTINDEXED: At = IROp->C<IROp_LoadContextIndexed>()->BaseOffset; Size = 8 * IROp->C<IROp_LoadContextIndexed>()->Stride; break;
  case OP_STORECONTEXTINDEXED: At = IROp->C<IROp_StoreContextIndexed>()->BaseOffset; Size = 8 * IROp->C<IROp_StoreContextIndexed>()->Stride; break;
  case OP_LOADCONTEXTPAIR: At = IROp->C<IROp_LoadContextPair>()->Offset; Size = 32; break;
  case OP_STORECONTEXTPAIR: At = IROp->C<IROp_StoreContextPair>()->Offset; Size = 32; break;
  case OP_FEXCX87SVC: return true;
  default: return false;
  }
  const auto Hits = [&](uint32_t F, uint32_t N) { return At < F + N && F < At + Size; };
  return Hits(offsetof(FEXCore::Core::CPUState, flags) + FEXCore::X86State::X87FLAG_TOP_LOC, 1) ||
         Hits(offsetof(FEXCore::Core::CPUState, AbridgedFTW), 1) || Hits(MMBaseOffset(), 8 * 16);
}

inline Ref X87StackOptimization::StackChecked(Ref Value, uint8_t Offset, bool Push) {
  // Push: register TOP - 1 (offset 7) must be empty; read: register TOP + Offset must not be.
  Pushed |= Push;
  if (FexcCache() && TopValidCache[Offset] == (Push ? StackSlot::INVALID : StackSlot::VALID)) {
    return Value;   // the tag is known in this block: no fault
  }
  if (TagStub(Offset, Push)) {
    // Past the test the tag is as the instruction needs it: known from here on (not a change: nothing to write).
    if (TopValidCache[Offset] == StackSlot::UNUSED) TopValidCache[Offset] = Push ? StackSlot::INVALID : StackSlot::VALID;
    return Value;
  }
  Ref Valid = GetX87ValidTag_Slow(Offset);
  Ref Bad = Push ? Valid : IREmit->_Xor(OpSize::i32Bit, Valid, GetConstant(1));
  NoteFault(Bad, Push);
  if (!Push && TopValidCache[Offset] == StackSlot::UNUSED) {
    TopBad[Offset] = Bad;
  }
  Ref Keep = IREmit->_VCastFromGPR(OpSize::i64Bit, OpSize::i64Bit, IREmit->_Sub(OpSize::i64Bit, Bad, GetConstant(1)));   // all ones: no fault
  Ref Indefinite = IREmit->_VCastFromGPR(OpSize::i64Bit, OpSize::i64Bit, IREmit->_Constant(0xfff8'0000'0000'0000ULL));
  return IREmit->_VBSL(OpSize::i64Bit, Keep, Value, Indefinite);
}

inline Ref X87StackOptimization::FaultResult(Ref Value) {
  // An arithmetic result after a stack underflow of this op is the indefinite (the masked response), even when the other
  // operand is a NaN that the operation would propagate.
  if (!fexc_slice1 || !OpFault) {
    return Value;
  }
  Ref Keep = IREmit->_VCastFromGPR(OpSize::i64Bit, OpSize::i64Bit, IREmit->_Sub(OpSize::i64Bit, OpFault, GetConstant(1)));
  Ref Indefinite = IREmit->_VCastFromGPR(OpSize::i64Bit, OpSize::i64Bit, IREmit->_Constant(0xfff8'0000'0000'0000ULL));
  return IREmit->_VBSL(OpSize::i64Bit, Keep, Value, Indefinite);
}

inline Ref X87StackOptimization::QuietSNaN(Ref Value) {
  // FST/FSTP m64 of an SNaN stores it quieted, with IE (SDM FST #IA); flag-free, every other value unchanged.
  Ref Bits = IREmit->_VExtractToGPR(OpSize::i64Bit, OpSize::i64Bit, Value, 0);
  Ref Abs = IREmit->_And(OpSize::i64Bit, Bits, IREmit->_Constant(0x7fff'ffff'ffff'ffffULL));
  Ref IsNaN = IREmit->_Lshr(OpSize::i64Bit, IREmit->Add(OpSize::i64Bit, Abs, 0x000f'ffff'ffff'ffffULL), IREmit->_Constant(63));
  Ref Quiet = IREmit->_And(OpSize::i64Bit, IREmit->_Lshr(OpSize::i64Bit, Bits, IREmit->_Constant(51)), IREmit->_Constant(1));
  Ref IsSNaN = IREmit->_Andn(OpSize::i64Bit, IsNaN, Quiet);
  SNaN = SNaN ? IREmit->_Or(OpSize::i32Bit, SNaN, IsSNaN) : IsSNaN;
  return IREmit->_VCastFromGPR(OpSize::i64Bit, OpSize::i64Bit, IREmit->_Orlshl(OpSize::i64Bit, Bits, IsSNaN, 51));
}

inline void X87StackOptimization::FlushStackFault() {
  const auto Flag = [&](uint32_t Loc) { return offsetof(FEXCore::Core::CPUState, flags) + Loc; };
  const auto OrFlag = [&](uint32_t Loc, Ref V) {
    IREmit->_StoreContextGPR(OpSize::i8Bit, IREmit->_Or(OpSize::i32Bit, IREmit->_LoadContextGPR(OpSize::i8Bit, Flag(Loc)), V), Flag(Loc));
  };
  if (Fault) {
    OrFlag(FEXCore::X86State::X87FLAG_IE_LOC, Fault);
    OrFlag(FEXCore::X86State::X87FLAG_SF_LOC, Fault);
  }
  if (SNaN) {
    OrFlag(FEXCore::X86State::X87FLAG_IE_LOC, SNaN);
  }
  if ((Pushed || C1Zero) && Overflow) {
    // FLD/FILD/the constants: C1 = 1 for a stack overflow, else 0; FCHS/FABS: 0 (SDM "FPU Flags Affected").
    IREmit->_StoreContextGPR(OpSize::i8Bit, Overflow, Flag(FEXCore::X86State::X87FLAG_C1_LOC));
    C1Known0 = false;
  } else if ((Pushed || C1Zero || Fault || C1Read) && !C1Known0) {
    // Any instruction that read a stack register: C1 = 0. Exact for
    // an underflow and for the forms that define C1 = 0; where x87 would report a round-up in C1 (arithmetic, FSQRT, FRNDINT,
    // FST/FSTP, FIST/FISTTP, FCMOVcc), C1 reads 0 (deviation 5).
    IREmit->_StoreContextGPR(OpSize::i8Bit, GetConstant(0), Flag(FEXCore::X86State::X87FLAG_C1_LOC));
    C1Known0 = FexcCache();
  }
  Fault = Overflow = SNaN = {};
  Pushed = C1Zero = C1Read = false;
}

inline void X87StackOptimization::FexcEndInstruction() {
  // natemu slice1: the instruction's check operands for FexcPrecision::PlaceFPChecks, and its
  // flags from the pending byte into the exception word, after everything it committed.
  if (FexcRip != ~0ull && (Fault || Overflow || SNaN || Tiny)) {
    auto& E = fexc_x87_extra[FexcRip];   // (FIST's out-of-range bit, the dispatcher's, may be there already)
    E.fault = Fault;
    E.overflow = Overflow;
    if (SNaN) E.invalid = SNaN;
    E.tiny = Tiny;
  }
  if (FexcX87Lazy()) {
    // M2 task 4: no per-instruction fold (FPSR keeps the flags until something reads the word); a stack fault drops them.
    if (Held) IREmit->_FexcX87Release(Fault);
    Held = false;
  } else if (FexcRip != ~0ull && fexc_x87.count(FexcRip)) {
    Ref Pending = IREmit->_LoadContextGPR(OpSize::i8Bit, FEXC_X87_PENDING);
    // A stack fault has precedence over the operation's own exceptions (SDM Vol.1 4.9.2): their flags are dropped.
    if (Fault) Pending = IREmit->_And(OpSize::i32Bit, Pending, IREmit->_Sub(OpSize::i32Bit, Fault, GetConstant(1)));
    Ref Word = IREmit->_LoadContextGPR(OpSize::i32Bit, FEXC_X87_EXCEPTIONS);
    IREmit->_StoreContextGPR(OpSize::i32Bit, IREmit->_Or(OpSize::i32Bit, Word, Pending), FEXC_X87_EXCEPTIONS);
    IREmit->_StoreContextGPR(OpSize::i8Bit, GetConstant(0), FEXC_X87_PENDING);
  }
  Tiny = {};
  if (Fault || SNaN || Pushed || C1Zero || C1Read) {
    FlushStackFault();
  }
}

inline void X87StackOptimization::Reset() {
  FexcRip = ~0ull;
  FexcOpcode = {};
  Held = false;
  C1Read = C1Known0 = false;
  Mutated = BoundaryFlushed = false;
  BoundaryTop.fill(nullptr);
  BoundaryFTW = StubCursor = {};
  Tiny = {};
  Fault = Overflow = OpFault = SNaN = {};
  Pushed = C1Zero = false;
  // natemu slice1 (I6): always FEX's slow path (its register, TOP and tag caches over the guest-state page, not the
  // simulated stack, which is written only where FEX's own code reads the page). M2 task 4: the caches are written to the
  // page before every instruction that can fault, stop or call out (FexcCache); a step unit and a unit under unmasked
  // x87 exceptions write after each lowered op, as before.
  SlowPath = fexc_slice1;
  StackData.clear();
  InvalidateCaches();
}

inline Ref X87StackOptimization::GetConstant(ssize_t Offset) {
  if (Offset < 0 || Offset >= X87StackOptimization::ConstantPool.size()) {
    // not dealt by pool
    return IREmit->_Constant(Offset);
  }
  if (ConstantPool[Offset] == nullptr) {

    ConstantPool[Offset] = IREmit->_Constant(Offset);
  }
  return ConstantPool[Offset];
}

inline void X87StackOptimization::MigrateToSlowPathIf(bool ShouldMigrate) {
  if (ShouldMigrate && !SlowPath) {
    SynchronizeStackValues();
    StackData.clear();
    SlowPath = true;
  }
}

inline Ref X87StackOptimization::GetTopWithCache_Slow() {
  if (!TopOffsetCache[0]) {
    TopOffsetCache[0] = IREmit->_LoadContextGPR(OpSize::i8Bit, offsetof(FEXCore::Core::CPUState, flags) + FEXCore::X86State::X87FLAG_TOP_LOC);
    // natemu slice1 (I6): TOP is a guest-state byte the guest can reach (the page deviation): keep every index inside mm[].
    if (fexc_slice1) TopOffsetCache[0] = IREmit->_And(OpSize::i32Bit, TopOffsetCache[0], GetConstant(7));
  }
  return TopOffsetCache[0];
}

inline Ref X87StackOptimization::GetOffsetTopWithCache_Slow(uint8_t Offset, bool Reverse) {
  if (Reverse) {
    Offset = 8 - Offset;
  }

  Offset &= 7;

  if (TopOffsetCache[Offset]) {
    return TopOffsetCache[Offset];
  }

  auto* OffsetTop = GetTopWithCache_Slow();
  if (Offset != 0) {
    OffsetTop = IREmit->_And(OpSize::i32Bit, IREmit->Add(OpSize::i32Bit, OffsetTop, Offset), GetConstant(7));
    // GetTopWithCache_Slow already sets the cache so we don't need to set it here for offset == 0
    TopOffsetCache[Offset] = OffsetTop;
  }

  return OffsetTop;
}

inline Ref X87StackOptimization::GetOffsetTopAddressWithCache_Slow(uint8_t Offset) {
  if (TopOffsetAddressCache[Offset]) {
    return TopOffsetAddressCache[Offset];
  }

  Ref OffsetRef = GetOffsetTopWithCache_Slow(Offset);
  TopOffsetAddressCache[Offset] = IREmit->_FormContextAddress(OpSize::i64Bit, OffsetRef, 16);

  return TopOffsetAddressCache[Offset];
}

inline void X87StackOptimization::SetTopWithCache_Slow(Ref Value) {
  InvalidateCachedRegs();
  TopOffsetCache[0] = Value;
  FlushTopPending = true;
}

inline Ref X87StackOptimization::GetFTW() {
  if (!FTWCached) {
    FTWCached = IREmit->_LoadContextGPR(OpSize::i8Bit, offsetof(FEXCore::Core::CPUState, AbridgedFTW));
  }
  return FTWCached;
}

inline void X87StackOptimization::SetX87ValidTag(uint8_t Offset, bool Valid) {
  if (FexcCache() && TopValidCache[Offset] == (Valid ? StackSlot::VALID : StackSlot::INVALID)) {
    return;   // known so already (written, or pending)
  }
  Mutated = true;
  TopValidCache[Offset] = Valid ? StackSlot::VALID : StackSlot::INVALID;
  TagDirty[Offset] = true;
  FlushValidPending = true;
}

inline Ref X87StackOptimization::GetX87ValidTag_Slow(uint8_t Offset) {
  switch (TopValidCache[Offset]) {
  case StackSlot::UNUSED:
    return IREmit->_And(OpSize::i32Bit, IREmit->_Lshr(OpSize::i32Bit, GetFTW(), GetOffsetTopWithCache_Slow(Offset)), GetConstant(1));
  case StackSlot::INVALID: return GetConstant(0);
  case StackSlot::VALID: return GetConstant(1);
  }
}

inline Ref X87StackOptimization::LoadStackValueAtOffset_Slow(uint8_t Offset) {
  OrderedNode* TopOffsetAddress = fexc_slice1 ? nullptr : GetOffsetTopAddressWithCache_Slow(Offset);
  auto Size = ReducedPrecisionMode ? OpSize::i64Bit : OpSize::i128Bit;
  C1Read |= CheckReads;
  if (fexc_slice1 && FexcCache() && TopValidCache[Offset] == StackSlot::INVALID) {
    // Known empty in this block (FFREE, a pop): a stack underflow; the register keeps what it holds (the cache is not touched).
    Ref Indefinite = IREmit->_VCastFromGPR(OpSize::i64Bit, OpSize::i64Bit, IREmit->_Constant(0xfff8'0000'0000'0000ULL));
    if (CheckReads && !TagStub(Offset, false)) NoteFault(GetConstant(1), false);
    return Indefinite;
  }
  if (TopValueCache[Offset] && fexc_slice1 && TopBad[Offset] && CheckReads) {
    NoteFault(TopBad[Offset], false);   // the cached value of a register read with a run-time tag: the same fault again
  }
  if (!TopValueCache[Offset] && fexc_slice1) {
    // natemu slice1 (I6): `add x0, x28, xI, lsl #4; ldr dN, [x0, #mm]`, a guest-state access (not a guest memory op: no site).
    // (M2 task 4: the tag's test first, where it is a stub at the instruction's start: TagStub.)
    const bool Stub = CheckReads && FexcCache() && TopValidCache[Offset] == StackSlot::UNUSED && TagStub(Offset, false);
    if (Stub) TopValidCache[Offset] = StackSlot::VALID;
    Ref Raw = IREmit->_LoadContextFPRIndexed(GetOffsetTopWithCache_Slow(Offset), Size, MMBaseOffset(), 16);
    TopValueCache[Offset] = CheckReads ? StackChecked(Raw, Offset, false) : Raw;
  }
  if (!TopValueCache[Offset]) {
    TopValueCache[Offset] = IREmit->_LoadMemFPR(Size, TopOffsetAddress, IREmit->_InlineConstant(MMBaseOffset()), Size, MemOffsetType::SXTX, 1);
  }
  return TopValueCache[Offset];
}

inline void X87StackOptimization::StoreStackValueAtOffset_Slow(Ref Value, uint8_t Offset, bool SetValid) {
  Mutated = true;
  TopValueCache[Offset] = Value;
  TopBad[Offset] = {};
  TopQuiet[Offset] = false;   // (the caller sets it after)
  FlushValuesPending[Offset] = true;
  // mark it valid
  // In some cases we might already know it has been previously set as valid so we don't need to do it again
  // natemu slice1 (I6): always: the register written may have been empty (a stack underflow's masked response fills it).
  if (SetValid || fexc_slice1) {
    SetX87ValidTag(Offset, true);
  }
}

inline Ref X87StackOptimization::RotateRight8(uint32_t V, Ref Amount) {
  return IREmit->_Lshr(OpSize::i32Bit, GetConstant(V | (V << 8)), Amount);
}

inline std::optional<X87StackOptimization::StackMemberInfo> X87StackOptimization::MigrateToSlowPath_IfInvalid(uint8_t Offset) {
  const auto& [Valid, StackMember] = StackData.top(Offset);
  MigrateToSlowPathIf(Valid != StackSlot::VALID);
  if (Valid == StackSlot::VALID) {
    return StackMember;
  }
  return {};
}

inline Ref X87StackOptimization::LoadStackValue(uint8_t Offset) {
  const auto& StackValue = MigrateToSlowPath_IfInvalid(Offset);
  return SlowPath ? LoadStackValueAtOffset_Slow(Offset) : StackValue->StackDataNode;
}

inline void X87StackOptimization::StoreStackValue(Ref Value, uint8_t Offset, bool SetValid) {
  if (SlowPath) {
    StoreStackValueAtOffset_Slow(Value, Offset, SetValid);
  } else {
    StackData.setTop(StackMemberInfo {Value}, Offset);
  }
}

inline void X87StackOptimization::StackPop() {
  if (SlowPath) {
    UpdateTopForPop_Slow();
  } else {
    StackData.pop();
  }
}


void X87StackOptimization::HandleUnop(IROps Op64, bool VFOp64, IROps Op80) {
  Ref St0 = LoadStackValue();
  Ref Value {};

  if (ReducedPrecisionMode) {
    if (VFOp64) {
      DeriveOp(Value, Op64, IREmit->_VFSqrt(OpSize::i64Bit, OpSize::i64Bit, St0));
    } else {
      DeriveOp(Value, Op64, IREmit->_F64SIN(St0));
    }
  } else {
    DeriveOp(Value, Op80, IREmit->_F80SQRT(St0));
  }

  StoreStackValue(FaultResult(Value));
  TopQuiet[0] = VFOp64;
}


void X87StackOptimization::HandleBinopValue(IROps Op64, bool VFOp64, IROps Op80, uint8_t DestStackOffset, bool MarkDestValid,
                                            uint8_t StackOffset, Ref ValueNode, bool Reverse) {
  LOGMAN_THROW_A_FMT(!Reverse || VFOp64, "There are no reverse operations using non VFOp64 ops");
  auto StackNode = LoadStackValue(StackOffset);

  Ref Node = {};
  if (ReducedPrecisionMode) {
    if (Reverse) {
      DeriveOp(Node, Op64, IREmit->_VFAdd(OpSize::i64Bit, OpSize::i64Bit, ValueNode, StackNode));
    } else {
      if (VFOp64) {
        DeriveOp(Node, Op64, IREmit->_VFAdd(OpSize::i64Bit, OpSize::i64Bit, StackNode, ValueNode));
      } else {
        DeriveOp(Node, Op64, IREmit->_F64FPREM(StackNode, ValueNode));
      }
    }
  } else {
    if (Reverse) {
      DeriveOp(Node, Op80, IREmit->_F80Add(ValueNode, StackNode));
    } else {
      DeriveOp(Node, Op80, IREmit->_F80Add(StackNode, ValueNode));
    }
  }

  StoreStackValue(FaultResult(Node), DestStackOffset, MarkDestValid && StackOffset != DestStackOffset);
  TopQuiet[DestStackOffset] = VFOp64;
}

void X87StackOptimization::HandleBinopStack(IROps Op64, bool VFOp64, IROps Op80, uint8_t DestStackOffset, uint8_t StackOffset1,
                                            uint8_t StackOffset2, bool Reverse) {
  auto StackNode = LoadStackValue(StackOffset2);
  HandleBinopValue(Op64, VFOp64, Op80, DestStackOffset, StackOffset2 != DestStackOffset, StackOffset1, StackNode, Reverse);
}

inline void X87StackOptimization::UpdateTopForPop_Slow(bool InvalidateTag) {
  const auto PopContainer = [](auto& container) {
    const auto begin = std::begin(container);
    std::rotate(begin, std::next(begin), std::end(container));
  };

  if (InvalidateTag) {
    SetX87ValidTag(0, false);
  }

  // Pop the top of the x87 stack
  GetOffsetTopWithCache_Slow(1);
  Mutated = true;
  PopContainer(TopOffsetCache);
  PopContainer(TopOffsetAddressCache);
  PopContainer(TopValueCache);
  PopContainer(FlushValuesPending);
  PopContainer(TopValidCache);
  PopContainer(TopBad);
  PopContainer(TopQuiet);
  PopContainer(TagDirty);
  FlushTopPending = true;
}

inline void X87StackOptimization::UpdateTopForPush_Slow() {
  const auto PushContainer = [](auto& container) {
    const auto end = std::end(container);
    std::rotate(std::begin(container), std::prev(end), end);
  };

  // Push the top of the x87 stack
  GetOffsetTopWithCache_Slow(1, true);
  Mutated = true;
  PushContainer(TopOffsetCache);
  PushContainer(TopOffsetAddressCache);
  PushContainer(TopValueCache);
  PushContainer(FlushValuesPending);
  PushContainer(TopValidCache);
  PushContainer(TopBad);
  PushContainer(TopQuiet);
  PushContainer(TagDirty);
  FlushTopPending = true;
}

void X87StackOptimization::FlushCachedRegs() {
  if (FlushTopPending) {
    IREmit->_StoreContextGPR(OpSize::i8Bit, TopOffsetCache[0], offsetof(FEXCore::Core::CPUState, flags) + FEXCore::X86State::X87FLAG_TOP_LOC);
    FlushTopPending = false;
  }

  auto Size = ReducedPrecisionMode ? OpSize::i64Bit : OpSize::i128Bit;
  for (size_t i = 0; i < FlushValuesPending.size(); i++) {
    if (FlushValuesPending[i] && fexc_slice1) {
      IREmit->_StoreContextFPRIndexed(TopValueCache[i], GetOffsetTopWithCache_Slow(i), Size, MMBaseOffset(), 16);   // natemu slice1 (I6)
      FlushValuesPending[i] = false;
    }
    if (FlushValuesPending[i]) {
      OrderedNode* TopOffsetAddress = GetOffsetTopAddressWithCache_Slow(i);
      IREmit->_StoreMemFPR(Size, TopValueCache[i], TopOffsetAddress, IREmit->_InlineConstant(MMBaseOffset()), Size, MemOffsetType::SXTX, 1);
      // store
      FlushValuesPending[i] = false;
    }
  }

  if (FlushValidPending) {
    uint8_t ValidMask = 0;
    uint8_t InvalidMask = 0;
    for (auto It = TopValidCache.rbegin(); It != TopValidCache.rend(); It++) {
      ValidMask <<= 1;
      InvalidMask <<= 1;
      if (*It == StackSlot::VALID) {
        ValidMask |= 1;
      } else if (*It == StackSlot::INVALID) {
        InvalidMask |= 1;
      }
    }
    if (FexcCache() && ValidMask != 0xff && InvalidMask != 0xff) {
      // The cache keeps what is known across flushes: write the tags that changed since the last one.
      uint8_t Dirty = 0;
      for (size_t i = 0; i < TagDirty.size(); i++) Dirty |= TagDirty[i] << i;
      ValidMask &= Dirty;
      InvalidMask &= Dirty;
    }
    TagDirty.fill(false);

    if (ValidMask || InvalidMask) {
      Ref NewFTW = [&]() {
        if (ValidMask == 0xff || InvalidMask == 0xff) {
          // If InvalidMask == 0xff then ValidMask = 0
          return GetConstant(ValidMask);
        } else {
          Ref NewFTW = GetFTW();
          Ref RotAmount {};
          if (std::popcount(ValidMask) == 1) {
            uint8_t BitIdx = std::countr_zero(ValidMask);
            Ref RegMask = IREmit->_Lshl(OpSize::i32Bit, GetConstant(1), GetOffsetTopWithCache_Slow(BitIdx));
            NewFTW = IREmit->_Or(OpSize::i32Bit, NewFTW, RegMask);
          } else if (ValidMask) {
            RotAmount = IREmit->_Sub(OpSize::i32Bit, GetConstant(8), GetTopWithCache_Slow());
            // perform a rotate right on mask by top
            NewFTW = IREmit->_Or(OpSize::i32Bit, NewFTW, RotateRight8(ValidMask, RotAmount));
          }

          if (std::popcount(InvalidMask) == 1) {
            uint8_t BitIdx = std::countr_zero(InvalidMask);
            Ref RegMask = IREmit->_Lshl(OpSize::i32Bit, GetConstant(1), GetOffsetTopWithCache_Slow(BitIdx));
            NewFTW = IREmit->_Andn(OpSize::i32Bit, NewFTW, RegMask);
          } else if (InvalidMask) {
            if (!RotAmount) {
              RotAmount = IREmit->_Sub(OpSize::i32Bit, GetConstant(8), GetTopWithCache_Slow());
            }
            NewFTW = IREmit->_Andn(OpSize::i32Bit, NewFTW, RotateRight8(InvalidMask, RotAmount));
          }
          return NewFTW;
        }
      }();

      IREmit->_StoreContextGPR(OpSize::i8Bit, NewFTW, offsetof(FEXCore::Core::CPUState, AbridgedFTW));
      FTWCached = NewFTW;
    }

    FlushValidPending = false;
  }
}

// We synchronize stack values in a few occasions but one of the most important of those,
// is when we move from fast to a slow path and need to make sure that the context is properly
// written.
Ref X87StackOptimization::SynchronizeStackValues() {
  if (SlowPath) {
    return GetTopWithCache_Slow();
  }

  // Store new top which is now the original top minus recorded top offset
  // Careful with underflow wraparound.
  const auto TopOffset = StackData.TopOffset;

  if (TopOffset != 0) {
    Ref NewTop = GetOffsetTopWithCache_Slow(TopOffset, true);
    SetTopWithCache_Slow(NewTop);
  }
  StackData.TopOffset = 0;

  // Before leaving we need to write the current values in the stack to
  // context so that the values are correct. Copy SourceDataNode in the
  // stack to the respective mmX register.
  Ref TopValue = GetTopWithCache_Slow();
  for (size_t i = 0; i < StackData.size; ++i) {
    const auto& [Valid, StackMember] = StackData.top(i);

    if (Valid == StackSlot::VALID) {
      StoreStackValueAtOffset_Slow(StackMember.StackDataNode, i, false);
    }
  }
  { // Set valid tags
    uint8_t ValidMask = StackData.getValidMask();
    uint8_t InvalidMask = StackData.getInvalidMask();
    for (auto& Elem : TopValidCache) {
      Elem = (ValidMask & 1) ? StackSlot::VALID : ((InvalidMask & 1) ? StackSlot::INVALID : StackSlot::UNUSED);

      ValidMask >>= 1;
      InvalidMask >>= 1;
    }
    FlushValidPending = true;
  }

  return TopValue;
}

void X87StackOptimization::Run(IREmitter* Emit) {
  FEXCORE_PROFILE_SCOPED("PassManager::x87StackOpt");

  auto CurrentIR = Emit->ViewIR();
  auto* HeaderOp = CurrentIR.GetHeader();
  LOGMAN_THROW_A_FMT(HeaderOp->Header.Op == OP_IRHEADER, "First op wasn't IRHeader");

  if (!HeaderOp->HasX87) {
    // If there is no x87 in this, just early exit.
    return;
  }

  // Initialize IREmit member
  IREmit = Emit;
  const uint32_t FexcFirstNew = CurrentIR.GetSSACount();   // natemu: this pass's own nodes are numbered from here

  // Run optimization proper
  for (auto [BlockNode, BlockHeader] : CurrentIR.GetBlocks()) {
    auto BlockIROp = BlockHeader->CW<FEXCore::IR::IROp_CodeBlock>();
    // Each time we deal with a new block we need to start over.
    // The optimization should run per-block
    Reset();

    IREmit->SetCurrentCodeBlock(BlockNode);
    for (auto [CodeNode, IROp] : CurrentIR.GetCode(BlockNode)) {
      if (fexc_slice1 && IROp->Op == OP_GUESTOPCODE) {
        IREmit->SetWriteCursorBefore(CodeNode);   // natemu slice1 (I6, I5c): the previous instruction's stack fault flags, its checks
        FexcEndInstruction();
        FexcRip = HeaderOp->OriginalRIP + IROp->C<IROp_GuestOpcode>()->GuestEntryOffset;
        FexcOpcode = CodeNode;
        FexcBoundary(!fexc_x87_quiet.count(FexcRip));
        for (size_t i = 0; i < TopBad.size(); i++) {
          if (TopBad[i]) {   // a register read with a run-time tag check: read and checked again (its check bit is not kept in a register)
            TopBad[i] = {};
            TopValueCache[i] = {};
          }
        }
        continue;
      }
      if (fexc_slice1 && CurrentIR.GetID(CodeNode).Value < FexcFirstNew &&
          (IROp->Op == OP_FEXCX87SVC || (IROp->Op == OP_STORECONTEXT && IROp->C<IROp_StoreContext>()->Offset ==
                                           offsetof(FEXCore::Core::CPUState, flags) + FEXCore::X86State::X87FLAG_C1_LOC))) {
        C1Known0 = false;   // the dispatcher's own C1 (FCOMI, FXAM), the resident's
      }
      if (fexc_slice1 && CurrentIR.GetID(CodeNode).Value < FexcFirstNew && FexcX87State(IROp)) {
        IREmit->SetWriteCursorBefore(CodeNode);   // M2 task 4: a dispatcher op on the page's x87 state: written first, read again after
        InvalidateCachedRegs();
        Mutated = true;
        continue;
      }
      if (!LoweredX87(IROp->Op)) {
        continue;
      }
      IREmit->SetWriteCursor(CodeNode);
      OpFault = {};
      switch (IROp->Op) {
      case OP_F80ADDSTACK: {
        const auto* Op = IROp->C<IROp_F80AddStack>();
        HandleBinopStack(OP_VFADD, true, OP_F80ADD, Op->SrcStack1, Op->SrcStack1, Op->SrcStack2);
        break;
      }

      case OP_F80SUBSTACK: {
        const auto* Op = IROp->C<IROp_F80SubStack>();
        HandleBinopStack(OP_VFSUB, true, OP_F80SUB, Op->DstStack, Op->SrcStack1, Op->SrcStack2);
        break;
      }

      case OP_F80MULSTACK: {
        const auto* Op = IROp->C<IROp_F80MulStack>();
        HandleBinopStack(OP_VFMUL, true, OP_F80MUL, Op->SrcStack1, Op->SrcStack1, Op->SrcStack2);
        break;
      }

      case OP_F80DIVSTACK: {
        const auto* Op = IROp->C<IROp_F80DivStack>();
        HandleBinopStack(OP_VFDIV, true, OP_F80DIV, Op->DstStack, Op->SrcStack1, Op->SrcStack2);
        break;
      }

      case OP_F80FPREMSTACK: {
        HandleBinopStack(OP_F64FPREM, false, OP_F80FPREM, 0, 0, 1);
        break;
      }

      case OP_F80FPREM1STACK: {
        HandleBinopStack(OP_F64FPREM1, false, OP_F80FPREM1, 0, 0, 1);
        break;
      }

      case OP_F80SCALESTACK: {
        HandleBinopStack(OP_F64SCALE, false, OP_F80SCALE, 0, 0, 1);
        break;
      }

      case OP_F80FYL2XSTACK: {
        HandleBinopStack(OP_F64FYL2X, false, OP_F80FYL2X, 1, 0, 1);
        StackPop();
        break;
      }

      case OP_F80FYL2XP1STACK: {
        HandleBinopStack(OP_F64FYL2XP1, false, OP_F80FYL2XP1, 1, 0, 1);
        StackPop();
        break;
      }

      case OP_F80ATANSTACK: {
        HandleBinopStack(OP_F64ATAN, false, OP_F80ATAN, 1, 1, 0);
        StackPop();
        break;
      }

      case OP_F80ADDVALUE: {
        const auto* Op = IROp->C<IROp_F80AddValue>();
        HandleBinopValue(OP_VFADD, true, OP_F80ADD, 0, true, Op->SrcStack, CurrentIR.GetNode(Op->X80Src));
        break;
      }

      case OP_F80SUBRVALUE:
      case OP_F80SUBVALUE: {
        const auto* Op = IROp->C<IROp_F80SubValue>();
        HandleBinopValue(OP_VFSUB, true, OP_F80SUB, 0, true, Op->SrcStack, CurrentIR.GetNode(Op->X80Src), IROp->Op == OP_F80SUBRVALUE);
        break;
      }

      case OP_F80DIVRVALUE:
      case OP_F80DIVVALUE: {
        const auto* Op = IROp->C<IROp_F80DivValue>();
        HandleBinopValue(OP_VFDIV, true, OP_F80DIV, 0, true, Op->SrcStack, CurrentIR.GetNode(Op->X80Src), IROp->Op == OP_F80DIVRVALUE);
        break;
      }

      case OP_F80MULVALUE: {
        const auto* Op = IROp->C<IROp_F80MulValue>();
        HandleBinopValue(OP_VFMUL, true, OP_F80MUL, 0, true, Op->SrcStack, CurrentIR.GetNode(Op->X80Src));
        break;
      }

      case OP_F80SQRTSTACK: {
        HandleUnop(OP_VFSQRT, true, OP_F80SQRT);
        break;
      }

      case OP_F80SINSTACK: {
        HandleUnop(OP_F64SIN, false, OP_F80SIN);
        break;
      }

      case OP_F80COSSTACK: {
        HandleUnop(OP_F64COS, false, OP_F80COS);
        break;
      }

      case OP_F80F2XM1STACK: {
        HandleUnop(OP_F64F2XM1, false, OP_F80F2XM1);
        break;
      }


      case OP_F80PTANSTACK: {
        HandleUnop(OP_F64TAN, false, OP_F80TAN);
        Ref OneConst {};
        if (ReducedPrecisionMode) {
          OneConst = IREmit->_VCastFromGPR(OpSize::i64Bit, OpSize::i64Bit, GetConstant(0x3FF0000000000000));
        } else {
          OneConst = IREmit->_LoadNamedVectorConstant(OpSize::i128Bit, NamedVectorConstant::NAMED_VECTOR_X87_ONE);
        }

        if (SlowPath) {
          UpdateTopForPush_Slow();
          StoreStackValueAtOffset_Slow(OneConst);
        } else {
          StackData.push(StackMemberInfo {OneConst});
        }
        break;
      }

      case OP_F80SINCOSSTACK: {
        Ref St0 = LoadStackValue();

        Ref SinValue {};
        Ref CosValue {};

        if (ReducedPrecisionMode) {
          SinValue = IREmit->_F64SIN(St0);
          CosValue = IREmit->_F64COS(St0);
        }
#ifdef VIXL_SIMULATOR
        else if (DisableVixlIndirectCalls() == 0) {
          SinValue = IREmit->_F80SIN(St0);
          CosValue = IREmit->_F80COS(St0);
        }
#endif
        else {
          SinValue = IREmit->_AllocateFPR(OpSize::i128Bit, OpSize::i128Bit);
          CosValue = IREmit->_AllocateFPR(OpSize::i128Bit, OpSize::i128Bit);
          IREmit->_F80SINCOS(St0, SinValue, CosValue);
        }

        // Push values
        if (SlowPath) {
          StoreStackValueAtOffset_Slow(SinValue, 0, false);
          UpdateTopForPush_Slow();
          StoreStackValueAtOffset_Slow(CosValue, 0, true);
        } else {
          StackData.setTop(StackMemberInfo {SinValue});
          StackData.push(StackMemberInfo {CosValue});
        }
        break;
      }

      case OP_INITSTACK: {
        StackData.clear();
        InvalidateCachedRegs();
        break;
      }

      case OP_INVALIDATESTACK: {
        const auto* Op = IROp->C<IROp_ReadStackValue>();
        auto Offset = Op->StackLocation;

        if (Offset != 0xff) { // invalidate single offset
          if (SlowPath) {
            SetX87ValidTag(Offset, false);
          } else {
            StackData.setTagInvalid(Offset);
          }
        } else { // invalidate all
          if (SlowPath) {
            Mutated = true;
            TopValidCache.fill(StackSlot::INVALID);
            TagDirty.fill(true);
            FlushValidPending = true;
          } else {
            for (size_t i = 0; i < StackData.size; i++) {
              StackData.setTagInvalid(i);
            }
          }
        }
        break;
      }

      case OP_PUSHSTACK: {
        const auto* Op = IROp->C<IROp_PushStack>();
        auto* SourceNode = CurrentIR.GetNode(Op->X80Src);

        if (SlowPath) {
          const bool Quiet = fexc_slice1 && FexcQuietValue(CurrentIR, SourceNode);
          if (fexc_slice1) SourceNode = StackChecked(SourceNode, 7, true);   // natemu slice1 (I6): stack overflow
          UpdateTopForPush_Slow();
          StoreStackValueAtOffset_Slow(SourceNode);
          TopQuiet[0] = Quiet;
        } else {
          if (Op->OriginalValue.IsInvalid()) {
            // No original value to track - just push the converted data
            StackData.push(StackMemberInfo {SourceNode});
          } else {
            auto* OriginalNode = CurrentIR.GetNode(Op->OriginalValue);
            StackData.push(StackMemberInfo {SourceNode, OriginalNode, Op->LoadSize});
          }
        }
        break;
      }

      case OP_COPYPUSHSTACK: {
        const auto* Op = IROp->C<IROp_CopyPushStack>();
        auto Offset = Op->StackLocation;
        auto Value = MigrateToSlowPath_IfInvalid(Offset);

        if (SlowPath) {
          Ref St0 = LoadStackValueAtOffset_Slow(Offset);
          const bool Quiet = TopQuiet[Offset];
          if (fexc_slice1) St0 = StackChecked(St0, 7, true);   // natemu slice1 (I6): stack overflow
          UpdateTopForPush_Slow();
          StoreStackValueAtOffset_Slow(St0);
          TopQuiet[0] = Quiet;
        } else {
          StackData.push(*Value);
        }
        break;
      }

      case OP_READSTACKVALUE: {
        const auto* Op = IROp->C<IROp_ReadStackValue>();
        auto Offset = Op->StackLocation;
        Ref NewValue = LoadStackValue(Offset);

        IREmit->ReplaceUsesWithAfter(CodeNode, NewValue, CodeNode);
        break;
      }

      case OP_STACKVALIDTAG: {
        // Returns 0 if value is valid and 1 otherwise.
        const auto* Op = IROp->C<IROp_StackValidTag>();
        auto Offset = Op->StackLocation;
        auto Value = MigrateToSlowPath_IfInvalid(Offset);

        Ref Tag {};
        if (SlowPath) {
          Tag = GetX87ValidTag_Slow(Offset);
        } else {
          Tag = Value ? GetConstant(1) : GetConstant(0);
        }

        IREmit->ReplaceUsesWithAfter(CodeNode, Tag, CodeNode);
        break;
      }

      case OP_STORESTACKMEM: {
        const auto* Op = IROp->C<IROp_StoreStackMem>();
        const auto& Value = MigrateToSlowPath_IfInvalid();
        Ref StackNode = SlowPath ? LoadStackValueAtOffset_Slow() : Value->StackDataNode;
        Ref AddrNode = CurrentIR.GetNode(Op->Addr);
        Ref Offset = CurrentIR.GetNode(Op->Offset);
        OpSize Align = Op->Align;
        MemOffsetType OffsetType = Op->OffsetType;
        uint8_t OffsetScale = Op->OffsetScale;

        // On the fast path we can optimize memory copies.
        // If we are doing:
        // fld dword [rax]
        // fst dword [rbx]
        // We can optimize this to:
        // ldr w2, [x0]
        // str w2, [x1]
        // or similar. As long as the source size and dest size are one and the same.
        // This will avoid any conversions between source and stack element size and conversion back.
        OpSize StoreSize = Op->StoreSize;
        LOGMAN_THROW_A_FMT(Op->StoreSize == OpSize::i32Bit || Op->StoreSize == OpSize::i64Bit || Op->StoreSize == OpSize::f80Bit,
                           "Invalid store size in x87 store stack mem");
        if (!SlowPath && Value->Source && Value->Source->Size == StoreSize) {
          Ref SourceValue = Value->Source->Node;
          if (Op->StoreSize == OpSize::f80Bit) {
            Store80BitToMem(Op, SourceValue, AddrNode, Offset, Align, OffsetType, OffsetScale);
          } else {
            IREmit->_StoreMemFPR(StoreSize, SourceValue, AddrNode, Offset, Align, OffsetType, OffsetScale);
          }
          break;
        }

        if (ReducedPrecisionMode) {
          StoreStackMem_Reduced_Helper(CurrentIR, Op, StackNode);
          break;
        }

        StoreStackMem_Helper(CurrentIR, Op, StackNode);
        break;
      }

      case OP_STORESTACKTOSTACK: { // stores top of stack in another place in stack.
        const auto* Op = IROp->C<IROp_StoreStackToStack>();
        auto Offset = Op->StackLocation;

        auto Value = MigrateToSlowPath_IfInvalid();

        // Need to store st0 to stack location - basically a copy.
        if (SlowPath) {
          const bool Quiet = TopQuiet[0];
          StoreStackValueAtOffset_Slow(LoadStackValueAtOffset_Slow(), Offset);
          TopQuiet[Offset] = Quiet;
        } else {
          StackData.setTop(*Value, Offset);
        }
        break;
      }
      case OP_POPSTACKDESTROY: {
        StackPop();
        break;
      }

      case OP_F80STACKXCHANGE: {
        const auto* Op = IROp->C<IROp_F80StackXchange>();
        auto Offset = Op->SrcStack;

        if (Offset == 0) {
          // No-op
          break;
        }

        const auto [ValidTop, StackMemberTop] = StackData.top(0);
        const auto [ValidOffset, StackMemberOffset] = StackData.top(Offset);

        if (ValidTop != StackSlot::VALID || ValidOffset != StackSlot::VALID) {
          // Slow path: do actual memory operations
          Ref ValueTop = LoadStackValue();
          Ref ValueOffset = LoadStackValue(Offset);
          const bool QuietTop = TopQuiet[0], QuietOffset = TopQuiet[Offset];
          StoreStackValue(ValueOffset, 0, true);
          StoreStackValue(ValueTop, Offset, true);
          TopQuiet[0] = QuietOffset;
          TopQuiet[Offset] = QuietTop;
        } else {
          // Fast path: swap complete StackMemberInfo preserving Source metadata
          StackData.setTop(StackMemberOffset, 0);
          StackData.setTop(StackMemberTop, Offset);
        }
        break;
      }

      case OP_F80STACKCHANGESIGN: {
        // natemu slice1 (I6): an empty ST(0) gives the indefinite itself, not its negation (checked after the operation).
        CheckReads = !fexc_slice1;
        Ref Value = LoadStackValue();
        CheckReads = true;

        // We need a couple of intermediate instructions to change the sign
        // of a value
        Ref ResultNode {};
        if (ReducedPrecisionMode && fexc_slice1) {
          // natemu slice1 (I6): FCHS flips the sign bit of any value, a NaN too (Arm FNEG leaves a NaN under FPCR.AH = 1).
          ResultNode = IREmit->_VXor(OpSize::i64Bit, Value, IREmit->_VCastFromGPR(OpSize::i64Bit, OpSize::i64Bit, IREmit->_Constant(0x8000'0000'0000'0000ULL)));
          ResultNode = StackChecked(ResultNode, 0, false);
          C1Zero = true;
        } else if (ReducedPrecisionMode) {
          ResultNode = IREmit->_VFNeg(OpSize::i64Bit, OpSize::i64Bit, Value);
        } else {
          Ref HelperNode = IREmit->_LoadNamedVectorConstant(OpSize::i128Bit, IR::NamedVectorConstant::NAMED_VECTOR_F80_SIGN_MASK);
          ResultNode = IREmit->_VXor(OpSize::i128Bit, Value, HelperNode);
        }
        StoreStackValue(ResultNode);
        break;
      }

      case OP_F80STACKABS: {
        CheckReads = !fexc_slice1;   // natemu slice1 (I6): as FCHS
        Ref Value = LoadStackValue();
        CheckReads = true;

        Ref ResultNode {};
        if (ReducedPrecisionMode && fexc_slice1) {
          // natemu slice1 (I6): FABS clears the sign bit of any value, a NaN too (Arm FABS leaves a NaN under FPCR.AH = 1).
          ResultNode = IREmit->_VAndn(OpSize::i64Bit, Value, IREmit->_VCastFromGPR(OpSize::i64Bit, OpSize::i64Bit, IREmit->_Constant(0x8000'0000'0000'0000ULL)));
          ResultNode = StackChecked(ResultNode, 0, false);
          C1Zero = true;
        } else if (ReducedPrecisionMode) {
          ResultNode = IREmit->_VFAbs(OpSize::i64Bit, OpSize::i64Bit, Value);
        } else {
          // Intermediate insts
          Ref HelperNode = IREmit->_LoadNamedVectorConstant(OpSize::i128Bit, IR::NamedVectorConstant::NAMED_VECTOR_F80_SIGN_MASK);
          ResultNode = IREmit->_VAndn(OpSize::i128Bit, Value, HelperNode);
        }
        StoreStackValue(ResultNode);
        break;
      }

      case OP_F80CMPSTACK: {
        const auto* Op = IROp->C<IROp_F80CmpStack>();
        auto Offset = Op->SrcStack;
        Ref StackValue1 = LoadStackValue();
        Ref StackValue2 = LoadStackValue(Offset);

        Ref CmpNode {};
        if (ReducedPrecisionMode) {
          CmpNode = IREmit->_FCmp(OpSize::i64Bit, StackValue1, StackValue2);
        } else {
          CmpNode = IREmit->_F80Cmp(StackValue1, StackValue2);
        }

        IREmit->ReplaceUsesWithAfter(CodeNode, CmpNode, CodeNode);
        break;
      }
      case OP_F80STACKTEST: {
        const auto* Op = IROp->C<IROp_F80StackTest>();
        auto Offset = Op->SrcStack;
        auto StackNode = LoadStackValue(Offset);
        Ref ZeroConst = IREmit->_VCastFromGPR(ReducedPrecisionMode ? OpSize::i64Bit : OpSize::i128Bit, OpSize::i64Bit, GetConstant(0));

        Ref CmpNode {};
        if (ReducedPrecisionMode) {
          CmpNode = IREmit->_FCmp(OpSize::i64Bit, StackNode, ZeroConst);
        } else {
          CmpNode = IREmit->_F80Cmp(StackNode, ZeroConst);
        }
        IREmit->ReplaceUsesWithAfter(CodeNode, CmpNode, CodeNode);
        break;
      }


      case OP_F80CMPVALUE: {
        const auto* Op = IROp->C<IROp_F80CmpValue>();
        const auto& Value = CurrentIR.GetNode(Op->X80Src);
        auto StackNode = LoadStackValue();

        Ref CmpNode {};
        if (ReducedPrecisionMode) {
          CmpNode = IREmit->_FCmp(OpSize::i64Bit, StackNode, Value);
        } else {
          CmpNode = IREmit->_F80Cmp(StackNode, Value);
        }
        IREmit->ReplaceUsesWithAfter(CodeNode, CmpNode, CodeNode);
        break;
      }

      case OP_SYNCSTACKTOSLOW: {
        // This synchronizes stack values but doesn't necessarily moves us off the FastPath!
        Ref NewTop = SynchronizeStackValues();
        FlushCachedRegs();
        IREmit->ReplaceUsesWithAfter(CodeNode, NewTop, CodeNode);
        break;
      }

      case OP_STACKFORCESLOW: {
        MigrateToSlowPathIf(true);
        InvalidateCachedRegs();
        break;
      }

      case OP_INCSTACKTOP: {
        if (SlowPath) {
          UpdateTopForPop_Slow(false);
        } else {
          StackData.rotate(false);
        }
        break;
      }

      case OP_DECSTACKTOP: {
        if (SlowPath) {
          UpdateTopForPush_Slow();
        } else {
          StackData.rotate(true);
        }
        break;
      }

      case OP_F80ROUNDSTACK: {
        Ref St0 = LoadStackValue();

        Ref Value {};
        if (ReducedPrecisionMode) {
          Value = IREmit->_Vector_FToI(OpSize::i64Bit, OpSize::i64Bit, St0, RoundMode::Host);
        } else {
          Value = IREmit->_F80Round(St0);
        }
        StoreStackValue(FaultResult(Value));
        TopQuiet[0] = ReducedPrecisionMode;
        break;
      }

      case OP_F80VBSLSTACK: {
        const auto* Op = IROp->C<IROp_F80VBSLStack>();

        auto StackOffset1 = Op->SrcStack1;
        auto StackOffset2 = Op->SrcStack2;
        Ref Value1 = LoadStackValue(StackOffset1);
        Ref Value2 = LoadStackValue(StackOffset2);

        Ref StackNode = IREmit->_VBSL(OpSize::i128Bit, CurrentIR.GetNode(Op->VectorMask), Value1, Value2);
        StoreStackValue(StackNode, 0, StackOffset1 && StackOffset2);
        break;
      }

      default: LOGMAN_THROW_A_FMT(false, "IROp was expected to be lowered");
      }
      // natemu slice1 (I6): without FexcCache the page holds the x87 state after every op.
      if (fexc_slice1 && !FexcCache()) InvalidateCaches();
      IREmit->Remove(CodeNode);
    }

    auto Last = CurrentIR.at(BlockIROp->Last);
    --Last;
    auto [LastCodeNode, LastIROp] = Last();
    LOGMAN_THROW_A_FMT(IsBlockExit(LastIROp->Op), "must be exit");
    IREmit->SetWriteCursorBefore(LastCodeNode);
    SynchronizeStackValues();
    FlushCachedRegs();
    if (fexc_slice1) FexcEndInstruction();
  }
}

fextl::unique_ptr<Pass> CreateX87StackOptimizationPass(const HostFeatures& Features, OpSize GPROpSize) {
  return fextl::make_unique<X87StackOptimization>(Features, GPROpSize);
}
} // namespace FEXCore::IR
