# MiniX64Vm

A small, dependency-free x86-64 interpreter in C++17.

MiniX64Vm decodes and executes x86-64 machine code one instruction at a time
against a guest register file. All memory access goes through callbacks you
provide, so the emulated code never touches host memory directly. Hooks on
`call`, `jmp`, conditional branches and `syscall` let you stub out external
functions or stop execution at interesting points.

It is not a full system emulator. It implements a practical subset of
integer, SSE and AVX instructions, which is often enough to run self-contained
functions (hashing, decoding, key schedules, obfuscated helper routines) in
isolation.

## Features

- Three source files, no third-party dependencies
- Guest memory through `read` / `write` / optional `fetch` callbacks
- Hooks for calls, branches and syscalls, plus a per-instruction trace callback
- Deterministic stop conditions: return from the entry function, stop RIP,
  instruction limit, call depth limit, memory faults
- Detailed stop result: reason, RIP, fault address, raw instruction bytes and mnemonic
- Built-in self-tests (`MiniX64Vm::RunSelfTests`)

## Supported instructions

| Group | Instructions |
|---|---|
| Data movement | `mov`, `movzx`, `movsx`, `movsxd`, `lea`, `xchg`, `push`, `pop`, `leave`, `cmovcc`, `setcc` |
| Arithmetic | `add`, `adc`, `sub`, `sbb`, `inc`, `dec`, `neg`, `cmp`, `mul`, `imul`, `div`, `idiv` |
| Logic and bits | `and`, `or`, `xor`, `not`, `test`, `shl`, `shr`, `sar`, `rol`, `ror`, `bt`, `bts`, `btr`, `btc` |
| Atomics | `xadd`, `cmpxchg` (the `lock` prefix is accepted and treated as a no-op) |
| Control flow | `call`, `ret`, `jmp`, `jcc`, `syscall` (via hook), `nop` |
| SSE | `movd`, `movq`, `movss`, `movsd`, `movaps`, `movups`, `movdqa`, `movlps`, `movhlps`, `movmskps`, `andps`, `orps`, `xorps`, `shufps`, `pand`, `por`, `pxor`, `paddq`, `pcmpeqd`, `pshufd`, `punpcklqdq`, `punpckhqdq` |
| AES-NI | `aesenc`, `aesenclast`, `aesdec`, `aesdeclast`, `aesimc` |
| AVX (VEX) | `vmovaps`, `vmovdqa`, `vmovd`, `vmovq`, `vandps`, `vorps`, `vxorps`, `vpxor`, `vblendps`, `vpaddd`, `vpmulld`, `vpshufd`, `vpsrld`, `vpsrlw`, `vpcmpeqb`, `vpcmpeqd`, `vpmovmskb`, `vptest`, `vtestps`, `vpextrq`, `vinsertf128`, `vextractf128` |
| VAES | `vaesenc`, `vaesenclast`, `vaesdec`, `vaesdeclast`, `vaesimc` |

Not all operand forms of every instruction are implemented. Anything outside
the subset stops with `StopReason::UnsupportedInstruction` and reports the raw
bytes, so it is easy to see what is missing. Not supported at all: x87 FPU,
string instructions (`rep movs` and friends), AVX-512, and privileged or
system instructions other than `syscall`.

## Requirements

- C++17 compiler
- x86-64 host with AES-NI (AES instructions are executed with host intrinsics)

Tested with MSVC 2022 on Windows. The code uses no OS APIs, and the CMake file
passes `-maes -msse4.1` to GCC/Clang, but those toolchains have not been tested yet.

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

To use it in your own CMake project:

```cmake
add_subdirectory(minix64vm)
target_link_libraries(your_target PRIVATE MiniX64Vm::minix64vm)
```

## Usage

```cpp
#include "MiniX64Vm.h"

using Vm = MiniVm::MiniX64Vm;

// lea rax, [rcx + rcx*2] ; add rax, rdx ; ret
const uint8_t code[] = { 0x48, 0x8D, 0x04, 0x49, 0x48, 0x01, 0xD0, 0xC3 };

bool Read(void* ctx, uint64_t address, void* dst, size_t size);        // your guest memory
bool Write(void* ctx, uint64_t address, const void* src, size_t size);

Vm::State state{};
state.rip = 0x10000000;               // where `code` is mapped in guest memory
state.gpr[Vm::RSP] = 0x20000F00;      // a writable guest stack
state.gpr[Vm::RCX] = 10;
state.gpr[Vm::RDX] = 7;

Vm::Options options{};
options.read = Read;
options.write = Write;

Vm::Result result = Vm::Run(&state, &options);
// result.reason == Vm::StopReason::Completed, state.gpr[Vm::RAX] == 37
```

A complete program with a memory map and an instruction trace is in
[`examples/basic.cpp`](examples/basic.cpp).

### Options

| Field | Purpose |
|---|---|
| `read`, `write`, `memoryContext` | Guest memory access. Return `false` to raise a read/write fault. |
| `fetch` | Optional separate callback for instruction bytes (e.g. to restrict execution to known code ranges). |
| `callHook` | Called on `call` and `jmp`. `Handled` on a `call` skips it (set the return value in `state` yourself); `Handled` on a `jmp` is treated as a stubbed tail call and returns to the address on top of the stack. `Stop` / `Fault` end emulation. |
| `branchHook` | Called before `call`, `jmp` and taken `jcc`. Return `true` if the hook has already written `state->rip` (for example the fallthrough) to suppress the branch. |
| `syscallHook` | Called on `syscall`. |
| `trace` | Called after every executed instruction with RIP, bytes, mnemonic and the current state. |
| `stopRip` | Stop when RIP reaches this address. |
| `instructionLimit`, `callDepthLimit` | Safety limits (defaults: 4096 instructions, depth 64). |
| `strictReturn` | If set, `ret` at depth 0 pops and jumps instead of finishing. |
| `useSegmentBases` | Add `State::fsBase` / `State::gsBase` to `fs:` / `gs:` memory operands (e.g. to emulate TEB access on Windows). |
| `rejectSegmentOverrides` | Treat any `fs:` / `gs:` prefix as an invalid instruction. |

### Stop reasons

`Completed`, `StopRip`, `InstructionLimit`, `ReadFault`, `WriteFault`,
`UnsupportedInstruction`, `InvalidInstruction`, `CallDepthLimit`, `HookStop`,
`HookFault`. Use `Vm::StopReasonName()` to print them.

## Project layout

```
include/MiniX64Vm.h      public API
src/MiniX64Vm.cpp        decoder, integer/SSE execution, self-tests
src/MiniX64VmVex.inl     VEX/AVX/VAES execution (included by MiniX64Vm.cpp)
tests/main.cpp           runs the built-in self-tests
examples/basic.cpp       minimal end-to-end example
```

## License

[MIT](LICENSE)
