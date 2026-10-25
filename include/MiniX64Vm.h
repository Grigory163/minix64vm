#pragma once

#include <cstddef>
#include <cstdint>

namespace MiniVm
{
	class MiniX64Vm
	{
	public:
		enum Gpr : uint8_t
		{
			RAX = 0,
			RCX = 1,
			RDX = 2,
			RBX = 3,
			RSP = 4,
			RBP = 5,
			RSI = 6,
			RDI = 7,
			R8 = 8,
			R9 = 9,
			R10 = 10,
			R11 = 11,
			R12 = 12,
			R13 = 13,
			R14 = 14,
			R15 = 15
		};

		enum Rflag : uint64_t
		{
			FlagCF = 1ull << 0,
			FlagPF = 1ull << 2,
			FlagAF = 1ull << 4,
			FlagZF = 1ull << 6,
			FlagSF = 1ull << 7,
			FlagOF = 1ull << 11
		};

		struct State
		{
			uint64_t gpr[16]{};
			uint64_t rip{};
			uint64_t rflags{ 2 };
			alignas(32) uint8_t ymm[16][32]{};
			uint64_t fsBase{};
			uint64_t gsBase{};
		};

		enum class StopReason : uint8_t
		{
			Completed,
			StopRip,
			InstructionLimit,
			ReadFault,
			WriteFault,
			UnsupportedInstruction,
			InvalidInstruction,
			CallDepthLimit,
			HookStop,
			HookFault
		};

		enum class CallHookResult : uint8_t
		{
			Unhandled,
			Handled,
			Stop,
			Fault
		};

		struct TraceRecord
		{
			uint64_t rip{};
			uint64_t nextRip{};
			uint8_t bytes[15]{};
			uint8_t length{};
			const char* mnemonic{};
			const State* state{};
		};

		using ReadFn = bool (*)(void* context, uint64_t address, void* dst, size_t size);
		using WriteFn = bool (*)(void* context, uint64_t address, const void* src, size_t size);
		using CallHookFn = CallHookResult (*)(void* context, uint64_t target, State* state);
		using SyscallHookFn = CallHookResult (*)(void* context, uint64_t instructionRip,
			State* state);
		/* true = hook already wrote state->rip (typically fallthrough); skip taken branch. */
		using BranchHookFn = bool (*)(void* context, uint64_t target, uint64_t fallthrough,
			State* state);
		using TraceFn = void (*)(void* context, const TraceRecord* record);

		struct Options
		{
			ReadFn read{};
			WriteFn write{};
			void* memoryContext{};
			CallHookFn callHook{};
			void* callHookContext{};
			SyscallHookFn syscallHook{};
			void* syscallHookContext{};
			BranchHookFn branchHook{};
			void* branchHookContext{};
			TraceFn trace{};
			void* traceContext{};
			uint64_t stopRip{};
			uint32_t instructionLimit{ 4096 };
			uint32_t callDepthLimit{ 64 };
			ReadFn fetch{};
			bool strictReturn{};
			bool rejectSegmentOverrides{};
			bool useSegmentBases{};
		};

		struct Result
		{
			StopReason reason{ StopReason::InvalidInstruction };
			uint64_t rip{};
			uint64_t faultAddress{};
			uint64_t branchTarget{};
			uint32_t instructionCount{};
			uint32_t callDepth{};
			uint8_t bytes[15]{};
			uint8_t length{};
			const char* mnemonic{};
		};

		static Result Run(State* state, const Options* options);
		static const char* StopReasonName(StopReason reason);
		static bool RunSelfTests(char* detail, size_t detailCapacity);
	};
}
