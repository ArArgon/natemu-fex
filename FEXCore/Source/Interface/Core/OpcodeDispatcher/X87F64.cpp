// SPDX-License-Identifier: MIT
/*
$info$
tags: frontend|x86-to-ir, opcodes|dispatcher-implementations
desc: Handles x86/64 x87 to IR
$end_info$
*/

#include "Interface/Core/OpcodeDispatcher.h"
#include "Interface/Core/X86Tables/X86Tables.h"
#include "Interface/IR/IR.h"

#include <FEXCore/Core/CoreState.h>
#include <FEXCore/Core/X86Enums.h>
#include <FEXCore/Utils/EnumUtils.h>
#include <FEXCore/Utils/LogManager.h>

#include <fexc_hook.h>

#include <stddef.h>
#include <stdint.h>

namespace FEXCore::IR {
class OrderedNode;

#define OpcodeArgs [[maybe_unused]] FEXCore::X86Tables::DecodedOp Op

void OpDispatchBuilder::X87LDENVF64(OpcodeArgs) {
  _StackForceSlow();

  const auto Size = OpSizeFromSrc(Op);
  Ref Mem = MakeSegmentAddress(Op, Op->Src[0]);

  auto NewFCW = _LoadMemGPR(OpSize::i16Bit, Mem, OpSize::i16Bit);
  // ignore the rounding precision, we're always 64-bit in F64.
  // extract rounding mode
  Ref roundingMode = _Bfe(OpSize::i32Bit, 3, 10, NewFCW);
  _SetRoundingMode(roundingMode, false, roundingMode);
  _StoreContextGPR(OpSize::i16Bit, NewFCW, offsetof(FEXCore::Core::CPUState, FCW));

  auto NewFSW = _LoadMemGPR(Size, Mem, Constant(IR::OpSizeToSize(Size)), Size, MemOffsetType::SXTX, 1);
  ReconstructX87StateFromFSW_Helper(NewFSW);

  {
    // FTW
    SetX87FTW(_LoadMemGPR(Size, Mem, Constant(IR::OpSizeToSize(Size) * 2), Size, MemOffsetType::SXTX, 1));
  }
}

void OpDispatchBuilder::X87FLDCWF64(OpcodeArgs) {
  _StackForceSlow();

  Ref NewFCW = LoadSourceGPR(Op, Op->Src[0], Op->Flags);
  // ignore the rounding precision, we're always 64-bit in F64.
  // extract rounding mode
  Ref roundingMode = _Bfe(OpSize::i32Bit, 3, 10, NewFCW);
  _SetRoundingMode(roundingMode, false, roundingMode);
  _StoreContextGPR(OpSize::i16Bit, NewFCW, offsetof(FEXCore::Core::CPUState, FCW));
}

// F64 ops
// Float load op with memory operand
void OpDispatchBuilder::FLDF64(OpcodeArgs, IR::OpSize Width) {
  Ref Data = LoadSourceFPR_WithOpSize(Op, Op->Src[0], Width, Op->Flags);
  // Convert to 64bit float
  Ref ConvertedData = Data;
  if (Width == OpSize::i32Bit) {
    ConvertedData = _Float_FToF(OpSize::i64Bit, OpSize::i32Bit, Data);
  } else if (Width == OpSize::f80Bit) {
    ConvertedData = _F80CVT(OpSize::i64Bit, Data);
  } else if (fexc_slice1 && Width == OpSize::i64Bit) {
    // natemu slice1 (I6): FLD m64 quiets an SNaN with IE and raises DE for a denormal (SDM FLD #IA, #D): a multiply by 1.0,
    // exact for every other value under the x87 FPCR (FZ = 0), inside the x87 FPSR wrapper.
    ConvertedData = _VFMul(OpSize::i64Bit, OpSize::i64Bit, Data, _VCastFromGPR(OpSize::i64Bit, OpSize::i64Bit, Constant(0x3ff0'0000'0000'0000ULL)));
  }
  _PushStack(ConvertedData, fexc_slice1 ? ConvertedData : Data, Width);
}

void OpDispatchBuilder::FBLDF64(OpcodeArgs) {
  // Read from memory
  Ref Data = LoadSourceFPR_WithOpSize(Op, Op->Src[0], OpSize::f80Bit, Op->Flags);
  Ref ConvertedData = _F80BCDLoad(Data);
  ConvertedData = _F80CVT(OpSize::i64Bit, ConvertedData);
  _PushStack(ConvertedData, Invalid(), OpSize::iInvalid);
}

void OpDispatchBuilder::FBSTPF64(OpcodeArgs) {
  Ref converted = _F80CVTTo(_ReadStackValue(0), OpSize::i64Bit);
  converted = _F80BCDStore(converted);
  StoreResultFPR_WithOpSize(Op, Op->Dest, converted, OpSize::f80Bit, OpSize::i8Bit);
  _PopStackDestroy();
}

void OpDispatchBuilder::FLDF64_Const(OpcodeArgs, uint64_t Num) {
  auto Data = _VCastFromGPR(OpSize::i64Bit, OpSize::i64Bit, Constant(Num));
  _PushStack(Data, Data, OpSize::i64Bit);
}

void OpDispatchBuilder::FILDF64(OpcodeArgs) {
  const auto ReadWidth = OpSizeFromSrc(Op);

  // Read from memory
  Ref Data = LoadSourceGPR_WithOpSize(Op, Op->Src[0], ReadWidth, Op->Flags);
  if (ReadWidth == OpSize::i16Bit) {
    Data = _Sbfe(OpSize::i64Bit, IR::OpSizeAsBits(ReadWidth), 0, Data);
  }
  auto ConvertedData = _Float_FromGPR_S(OpSize::i64Bit, ReadWidth == OpSize::i32Bit ? OpSize::i32Bit : OpSize::i64Bit, Data);
  _PushStack(ConvertedData, Invalid(), OpSize::iInvalid);
}

void OpDispatchBuilder::FISTF64(OpcodeArgs, bool Truncate) {
  const auto Size = OpSizeFromSrc(Op);

  if (fexc_slice1) {
    // natemu slice1 (I6): x86's FIST/FISTP/FISTTP (SDM Vol.2): round by FCW.RC (`frintx`: PE when inexact; the x87 FPCR) or
    // truncate (`fcvtzs`: PE when inexact), then store the integer, or the integer indefinite (the operand size's smallest
    // value) with IE for a NaN or a value outside the operand size's range. Arm saturates instead, and FEX converts the 16-bit
    // form at 64 bits and stores its low half. IE comes from FPSR.IOC (NaN, 64-bit overflow) or, for 16/32 bits, the range check.
    // Flag-free (no NZCV to save, few live registers: slice1 admits no spill area): NaN from the double's bits, the range
    // check by comparing the 64-bit result with its sign-extended low part (64 bits: the saturated INT64_MAX), a mask select.
    Ref Data = _ReadStackValue(0);
    // m64: frintx (FIST) or fcvtzs (FISTTP) raise PE; m16/m32 round without a flag (frinti/frintz) and compute PE below.
    Ref Integral = Truncate && Size == OpSize::i64Bit ? Data :
                   _Vector_FToI(OpSize::i64Bit, OpSize::i64Bit, Data, Truncate ? RoundMode::TowardsZero : RoundMode::Host);
    Ref Value = _Float_ToGPR_ZS(OpSize::i64Bit, OpSize::i64Bit, Integral);
    Ref Bits = _VExtractToGPR(OpSize::i64Bit, OpSize::i64Bit, Data, 0);
    Ref IsNaN = _Lshr(OpSize::i64Bit, Add(OpSize::i64Bit, _And(OpSize::i64Bit, Bits, Constant(0x7fff'ffff'ffff'ffffULL)), 0x000f'ffff'ffff'ffffULL),
                      Constant(63));
    Ref Diff = Size == OpSize::i64Bit ? _Xor(OpSize::i64Bit, Value, Constant(0x7fff'ffff'ffff'ffffULL))
                                      : _Xor(OpSize::i64Bit, Value, _Sbfe(OpSize::i64Bit, IR::OpSizeAsBits(Size), 0, Value));
    Ref NonZero = _Lshr(OpSize::i64Bit, _Or(OpSize::i64Bit, Diff, _Neg(OpSize::i64Bit, Diff)), Constant(63));
    Ref OutOfRange = Size == OpSize::i64Bit ? _Xor(OpSize::i64Bit, NonZero, Constant(1)) : NonZero;
    Ref Mask = _Neg(OpSize::i64Bit, _Or(OpSize::i64Bit, IsNaN, OutOfRange));
    const uint64_t Indefinite = Size == OpSize::i16Bit ? 0x8000 : Size == OpSize::i32Bit ? 0x8000'0000ULL : 0x8000'0000'0000'0000ULL;
    // m16/m32: PE when the rounding was inexact and the integer is stored (an out-of-range value raises IE alone). m64:
    // out of range is integral, and Arm raises no IXC beside IOC.
    Ref Inexact {};
    if (Size != OpSize::i64Bit) {
      Ref Diff2 = _Xor(OpSize::i64Bit, Bits, _VExtractToGPR(OpSize::i64Bit, OpSize::i64Bit, Integral, 0));
      Inexact = _Andn(OpSize::i64Bit, _Lshr(OpSize::i64Bit, _Or(OpSize::i64Bit, Diff2, _Neg(OpSize::i64Bit, Diff2)), Constant(63)), _Neg(OpSize::i64Bit, Mask));
    }
    Value = _Or(OpSize::i64Bit, _Andn(OpSize::i64Bit, Value, Mask), _And(OpSize::i64Bit, Mask, Constant(Indefinite)));
    StoreResultGPR_WithOpSize(Op, Op->Dest, Value, Size, OpSize::i8Bit, MemoryAccessType::NONTSO);
    if (Size != OpSize::i64Bit) {
      // After the store (a #PF there leaves FSW as it was, except the FPSR flags already accumulated; plan I6 deviations).
      SetRFLAG<FEXCore::X86State::X87FLAG_IE_LOC>(_Or(OpSize::i32Bit, GetRFLAG(FEXCore::X86State::X87FLAG_IE_LOC), OutOfRange));
    }
    if (Inexact) {
      SetRFLAG<FEXCore::X86State::X87FLAG_PE_LOC>(_Or(OpSize::i32Bit, GetRFLAG(FEXCore::X86State::X87FLAG_PE_LOC), Inexact));
    }
    if ((Op->TableInfo->Flags & X86Tables::InstFlags::FLAGS_POP) != 0) {
      _PopStackDestroy();
    }
    return;
  }

  Ref data = _ReadStackValue(0);
  bool CanUseFloatReg = Size == OpSize::i64Bit;
  if (CanUseFloatReg) {
    // If possible, it's faster to keep the data in an FPR than doing a GPR transfer.
    if (Truncate) {
      data = _Vector_FToZS(OpSize::i128Bit, OpSize::i64Bit, data);
    } else {
      data = _Vector_FToS(OpSize::i128Bit, OpSize::i64Bit, data);
    }
    StoreResultFPR_WithOpSize(Op, Op->Dest, data, OpSize::i64Bit, OpSize::i8Bit);
  } else {
    if (Truncate) {
      data = _Float_ToGPR_ZS(Size == OpSize::i32Bit ? OpSize::i32Bit : OpSize::i64Bit, OpSize::i64Bit, data);
    } else {
      data = _Float_ToGPR_S(Size == OpSize::i32Bit ? OpSize::i32Bit : OpSize::i64Bit, OpSize::i64Bit, data);
    }
    StoreResultGPR_WithOpSize(Op, Op->Dest, data, Size, OpSize::i8Bit,
                              CTX->IsVectorAtomicTSOEnabled() ? MemoryAccessType::DEFAULT : MemoryAccessType::NONTSO);
  }

  if ((Op->TableInfo->Flags & X86Tables::InstFlags::FLAGS_POP) != 0) {
    _PopStackDestroy();
  }
}

void OpDispatchBuilder::FADDF64(OpcodeArgs, IR::OpSize Width, bool Integer, OpDispatchBuilder::OpResult ResInST0) {
  if (Op->Src[0].IsNone()) { // Implicit argument case
    auto Offset = Op->OP & 7;
    auto St0 = 0;
    if (ResInST0 == OpResult::RES_STI) {
      _F80AddStack(Offset, St0);
    } else {
      _F80AddStack(St0, Offset);
    }
    if (Op->TableInfo->Flags & X86Tables::InstFlags::FLAGS_POP) {
      _PopStackDestroy();
    }
    return;
  }

  // We have one memory argument
  Ref arg {};

  if (Integer) {
    arg = LoadSourceGPR(Op, Op->Src[0], Op->Flags);
    if (Width == OpSize::i16Bit) {
      arg = _Sbfe(OpSize::i64Bit, 16, 0, arg);
    }
    arg = _Float_FromGPR_S(OpSize::i64Bit, Width == OpSize::i64Bit ? OpSize::i64Bit : OpSize::i32Bit, arg);
  } else if (Width == OpSize::i32Bit) {
    arg = LoadSourceFPR(Op, Op->Src[0], Op->Flags);
    arg = _Float_FToF(OpSize::i64Bit, OpSize::i32Bit, arg);
  } else if (Width == OpSize::i64Bit) {
    arg = LoadSourceFPR(Op, Op->Src[0], Op->Flags);
  } else {
    FEX_UNREACHABLE;
  }

  // top of stack is at offset zero
  _F80AddValue(0, arg);
}

// FIXME: following is very similar to FADDF64
void OpDispatchBuilder::FMULF64(OpcodeArgs, IR::OpSize Width, bool Integer, OpDispatchBuilder::OpResult ResInST0) {
  if (Op->Src[0].IsNone()) { // Implicit argument case
    auto offset = Op->OP & 7;
    auto st0 = 0;
    if (ResInST0 == OpResult::RES_STI) {
      _F80MulStack(offset, st0);
    } else {
      _F80MulStack(st0, offset);
    }
    if (Op->TableInfo->Flags & X86Tables::InstFlags::FLAGS_POP) {
      _PopStackDestroy();
    }
    return;
  }

  // We have one memory argument
  Ref arg {};

  if (Integer) {
    arg = LoadSourceGPR(Op, Op->Src[0], Op->Flags);
    if (Width == OpSize::i16Bit) {
      arg = _Sbfe(OpSize::i64Bit, 16, 0, arg);
    }
    arg = _Float_FromGPR_S(OpSize::i64Bit, Width == OpSize::i64Bit ? OpSize::i64Bit : OpSize::i32Bit, arg);
  } else if (Width == OpSize::i32Bit) {
    arg = LoadSourceFPR(Op, Op->Src[0], Op->Flags);
    arg = _Float_FToF(OpSize::i64Bit, OpSize::i32Bit, arg);
  } else if (Width == OpSize::i64Bit) {
    arg = LoadSourceFPR(Op, Op->Src[0], Op->Flags);
  } else {
    FEX_UNREACHABLE;
  }

  // top of stack is at offset zero
  _F80MulValue(0, arg);

  if (Op->TableInfo->Flags & X86Tables::InstFlags::FLAGS_POP) {
    _PopStackDestroy();
  }
}

void OpDispatchBuilder::FDIVF64(OpcodeArgs, IR::OpSize Width, bool Integer, bool Reverse, OpDispatchBuilder::OpResult ResInST0) {
  if (Op->Src[0].IsNone()) {
    const auto offset = Op->OP & 7;
    const auto st0 = 0;

    if (Reverse) {
      if (ResInST0 == OpResult::RES_STI) {
        _F80DivStack(offset, st0, offset);
      } else {
        _F80DivStack(st0, offset, st0);
      }
    } else {
      if (ResInST0 == OpResult::RES_STI) {
        _F80DivStack(offset, offset, st0);
      } else {
        _F80DivStack(st0, st0, offset);
      }
    }

    if (Op->TableInfo->Flags & X86Tables::InstFlags::FLAGS_POP) {
      _PopStackDestroy();
    }
    return;
  }

  // We have one memory argument
  Ref Arg {};

  if (Width == OpSize::i16Bit || Width == OpSize::i32Bit || Width == OpSize::i64Bit) {
    if (Integer) {
      Arg = LoadSourceGPR(Op, Op->Src[0], Op->Flags);
      if (Width == OpSize::i16Bit) {
        Arg = _Sbfe(OpSize::i64Bit, 16, 0, Arg);
      }
      Arg = _Float_FromGPR_S(OpSize::i64Bit, Width == OpSize::i64Bit ? OpSize::i64Bit : OpSize::i32Bit, Arg);
    } else if (Width == OpSize::i32Bit) {
      Arg = LoadSourceFPR(Op, Op->Src[0], Op->Flags);
      Arg = _Float_FToF(OpSize::i64Bit, OpSize::i32Bit, Arg);
    } else if (Width == OpSize::i64Bit) {
      Arg = LoadSourceFPR(Op, Op->Src[0], Op->Flags);
    }
  } else {
    FEX_UNREACHABLE;
  }

  // top of stack is at offset zero
  if (Reverse) {
    _F80DivRValue(Arg, 0);
  } else {
    _F80DivValue(0, Arg);
  }

  if (Op->TableInfo->Flags & X86Tables::InstFlags::FLAGS_POP) {
    _PopStackDestroy();
  }
}

void OpDispatchBuilder::FSUBF64(OpcodeArgs, IR::OpSize Width, bool Integer, bool Reverse, OpDispatchBuilder::OpResult ResInST0) {
  if (Op->Src[0].IsNone()) {
    const auto Offset = Op->OP & 7;
    const auto St0 = 0;

    if (Reverse) {
      if (ResInST0 == OpResult::RES_STI) {
        _F80SubStack(Offset, St0, Offset);
      } else {
        _F80SubStack(St0, Offset, St0);
      }
    } else {
      if (ResInST0 == OpResult::RES_STI) {
        _F80SubStack(Offset, Offset, St0);
      } else {
        _F80SubStack(St0, St0, Offset);
      }
    }

    if (Op->TableInfo->Flags & X86Tables::InstFlags::FLAGS_POP) {
      _PopStackDestroy();
    }
    return;
  }

  // We have one memory argument
  Ref arg {};

  if (Width == OpSize::i16Bit || Width == OpSize::i32Bit || Width == OpSize::i64Bit) {
    if (Integer) {
      arg = LoadSourceGPR(Op, Op->Src[0], Op->Flags);
      if (Width == OpSize::i16Bit) {
        arg = _Sbfe(OpSize::i64Bit, 16, 0, arg);
      }
      arg = _Float_FromGPR_S(OpSize::i64Bit, Width == OpSize::i64Bit ? OpSize::i64Bit : OpSize::i32Bit, arg);
    } else if (Width == OpSize::i32Bit) {
      arg = LoadSourceFPR(Op, Op->Src[0], Op->Flags);
      arg = _Float_FToF(OpSize::i64Bit, OpSize::i32Bit, arg);
    } else if (Width == OpSize::i64Bit) {
      arg = LoadSourceFPR(Op, Op->Src[0], Op->Flags);
    }
  } else {
    FEX_UNREACHABLE;
  }

  // top of stack is at offset zero
  if (Reverse) {
    _F80SubRValue(arg, 0);
  } else {
    _F80SubValue(0, arg);
  }

  if (Op->TableInfo->Flags & X86Tables::InstFlags::FLAGS_POP) {
    _PopStackDestroy();
  }
}

void OpDispatchBuilder::FTSTF64(OpcodeArgs) {
  // We are going to clobber NZCV, make sure it's in a GPR first.
  SaveNZCV();

  // Now we do our comparison.
  _F80StackTest(0);
  ConvertNZCVToX87();
}

void OpDispatchBuilder::FCOMIF64(OpcodeArgs, IR::OpSize Width, bool Integer, OpDispatchBuilder::FCOMIFlags WhichFlags, bool PopTwice) {
  Ref arg {};
  Ref b {};

  if (Op->Src[0].IsNone()) {
    // Implicit arg
    uint8_t offset = Op->OP & 7;
    b = _ReadStackValue(offset);
  } else if (Width == OpSize::i16Bit || Width == OpSize::i32Bit || Width == OpSize::i64Bit) {
    // Memory arg
    if (Integer) {
      arg = LoadSourceGPR(Op, Op->Src[0], Op->Flags);
      if (Width == OpSize::i16Bit) {
        arg = _Sbfe(OpSize::i64Bit, 16, 0, arg);
      }
      b = _Float_FromGPR_S(OpSize::i64Bit, Width == OpSize::i64Bit ? OpSize::i64Bit : OpSize::i32Bit, arg);
    } else if (Width == OpSize::i32Bit) {
      arg = LoadSourceFPR(Op, Op->Src[0], Op->Flags);
      b = _Float_FToF(OpSize::i64Bit, OpSize::i32Bit, arg);
    } else if (Width == OpSize::i64Bit) {
      b = LoadSourceFPR(Op, Op->Src[0], Op->Flags);
    }
  } else {
    FEX_UNREACHABLE;
  }

  if (WhichFlags == FCOMIFlags::FLAGS_X87) {
    // We are going to clobber NZCV, make sure it's in a GPR first.
    SaveNZCV();

    _F80CmpValue(b);
    ConvertNZCVToX87();
  } else {
    HandleNZCVWrite();
    _F80CmpValue(b);
    ComissFlags();
    if (fexc_slice1) {
      SetRFLAG<FEXCore::X86State::X87FLAG_C1_LOC>(Constant(0));   // natemu slice1 (I6): FCOMI/FUCOMI(P): C1 = 0 (SDM)
    }
  }

  if (PopTwice) {
    _PopStackDestroy();
    _PopStackDestroy();
  } else if ((Op->TableInfo->Flags & X86Tables::InstFlags::FLAGS_POP) != 0) {
    _PopStackDestroy();
  }
}

void OpDispatchBuilder::X87FXTRACTF64(OpcodeArgs) {
  // Split node into SIG and EXP while handling the special zero case.
  // i.e. if val == 0.0, then sig = 0.0, exp = -inf
  // if val == -0.0, then sig = -0.0, exp = -inf
  // if val is +/-Inf, then sig = val, exp = +inf
  // if val is NaN, then sig = val, exp = val
  // otherwise we just extract the 64-bit sig and exp as normal.
  Ref Node = _ReadStackValue(0);

  Ref Gpr = _VExtractToGPR(OpSize::i64Bit, OpSize::i64Bit, Node, 0);

  // zero case
  Ref ExpZV = _VCastFromGPR(OpSize::i64Bit, OpSize::i64Bit, Constant(0xfff0'0000'0000'0000UL));
  Ref SigZV = Node;

  // Inf/NaN case
  Ref ExpInfOnlyV = _VCastFromGPR(OpSize::i64Bit, OpSize::i64Bit, Constant(0x7ff0'0000'0000'0000UL));
  Ref ExpNanV = Node;
  Ref SigInfV = Node;

  // non zero case
  Ref ExpNZ = _Bfe(OpSize::i64Bit, 11, 52, Gpr);
  ExpNZ = Sub(OpSize::i64Bit, ExpNZ, Constant(1023));
  Ref ExpNZV = _Float_FromGPR_S(OpSize::i64Bit, OpSize::i64Bit, ExpNZ);

  Ref SigNZ = _And(OpSize::i64Bit, Gpr, Constant(0x800f'ffff'ffff'ffffLL));
  SigNZ = _Or(OpSize::i64Bit, SigNZ, Constant(0x3ff0'0000'0000'0000LL));
  Ref SigNZV = _VCastFromGPR(OpSize::i64Bit, OpSize::i64Bit, SigNZ);

  SaveNZCV();

  // Mantissa non-zero => NaN (exp result = input); else Inf (exp result = +Inf)
  Ref Mantissa = _And(OpSize::i64Bit, Gpr, Constant(0x000f'ffff'ffff'ffffULL));
  _TestNZ(OpSize::i64Bit, Mantissa, Constant(~0ULL));
  Ref ExpInfV = _NZCVSelectV(OpSize::i64Bit, CondClass::EQ, ExpInfOnlyV, ExpNanV);

  // Biased exponent == 0x7ff => Inf/NaN path, else non-zero-case.
  Ref BiasedExp = _Bfe(OpSize::i64Bit, 11, 52, Gpr);
  SubWithFlags(OpSize::i64Bit, BiasedExp, 0x7ff);
  Ref ExpNZOrInf = _NZCVSelectV(OpSize::i64Bit, CondClass::EQ, ExpInfV, ExpNZV);
  Ref SigNZOrInf = _NZCVSelectV(OpSize::i64Bit, CondClass::EQ, SigInfV, SigNZV);

  // Zero folds on top.
  _TestNZ(OpSize::i64Bit, Gpr, Constant(0x7fff'ffff'ffff'ffffUL));

  Ref Sig = _NZCVSelectV(OpSize::i64Bit, CondClass::EQ, SigZV, SigNZOrInf);
  Ref Exp = _NZCVSelectV(OpSize::i64Bit, CondClass::EQ, ExpZV, ExpNZOrInf);

  _PopStackDestroy();
  _PushStack(Exp, Invalid(), OpSize::iInvalid);
  _PushStack(Sig, Invalid(), OpSize::iInvalid);
}
} // namespace FEXCore::IR
