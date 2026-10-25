// Emulates a tiny function: rax = rcx * 3 + rdx, then prints a trace.

#include "MiniX64Vm.h"

#include <cstdio>
#include <cstring>

using Vm = MiniVm::MiniX64Vm;

namespace
{
	constexpr uint64_t kCodeBase = 0x10000000;
	constexpr uint64_t kStackBase = 0x20000000;

	const uint8_t kCode[] = {
		0x48, 0x8D, 0x04, 0x49, // lea rax, [rcx + rcx*2]
		0x48, 0x01, 0xD0,       // add rax, rdx
		0xC3                    // ret
	};

	uint8_t g_stack[0x1000];

	// Flat guest memory: one read-only code region and one writable stack region.
	bool Read(void*, uint64_t address, void* dst, size_t size)
	{
		if (address >= kCodeBase && address - kCodeBase + size <= sizeof(kCode))
		{
			std::memcpy(dst, kCode + (address - kCodeBase), size);
			return true;
		}
		if (address >= kStackBase && address - kStackBase + size <= sizeof(g_stack))
		{
			std::memcpy(dst, g_stack + (address - kStackBase), size);
			return true;
		}
		return false;
	}

	bool Write(void*, uint64_t address, const void* src, size_t size)
	{
		if (address >= kStackBase && address - kStackBase + size <= sizeof(g_stack))
		{
			std::memcpy(g_stack + (address - kStackBase), src, size);
			return true;
		}
		return false;
	}

	void Trace(void*, const Vm::TraceRecord* record)
	{
		std::printf("  %016llX  %-6s rax=%llX\n",
			static_cast<unsigned long long>(record->rip), record->mnemonic,
			static_cast<unsigned long long>(record->state->gpr[Vm::RAX]));
	}
}

int main()
{
	Vm::State state{};
	state.rip = kCodeBase;
	state.gpr[Vm::RSP] = kStackBase + sizeof(g_stack) - 0x20;
	state.gpr[Vm::RCX] = 10;
	state.gpr[Vm::RDX] = 7;

	Vm::Options options{};
	options.read = Read;
	options.write = Write;
	options.trace = Trace;

	const Vm::Result result = Vm::Run(&state, &options);
	std::printf("stop: %s after %u instructions, rax = %llu\n",
		Vm::StopReasonName(result.reason), result.instructionCount,
		static_cast<unsigned long long>(state.gpr[Vm::RAX]));
	return result.reason == Vm::StopReason::Completed && state.gpr[Vm::RAX] == 37 ? 0 : 1;
}
