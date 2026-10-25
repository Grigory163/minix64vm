#include "MiniX64Vm.h"

#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <wmmintrin.h>

namespace MiniVm
{
	namespace
	{
		using Vm = MiniX64Vm;

		struct Prefix
		{
			bool operand16{};
			bool address32{};
			bool rep{};
			bool repne{};
			bool rexPresent{};
			uint8_t rex{};
			uint8_t segment{};

			bool W() const { return (rex & 8u) != 0; }
			bool R() const { return (rex & 4u) != 0; }
			bool X() const { return (rex & 2u) != 0; }
			bool B() const { return (rex & 1u) != 0; }
		};

		struct Cursor
		{
			const uint8_t* bytes{};
			size_t available{};
			size_t at{};
			bool (*ensure)(void*, size_t){};
			void* context{};

			bool Ensure(size_t size)
			{
				const size_t required = at + size;
				if (required <= available) return true;
				if (!ensure || !ensure(context, required)) return false;
				available = required;
				return true;
			}

			bool Read8(uint8_t& value)
			{
				if (!Ensure(sizeof(value)))
					return false;
				value = bytes[at++];
				return true;
			}

			bool Read16(uint16_t& value)
			{
				if (!Ensure(sizeof(value)))
					return false;
				memcpy(&value, bytes + at, sizeof(value));
				at += sizeof(value);
				return true;
			}

			bool Read32(uint32_t& value)
			{
				if (!Ensure(sizeof(value)))
					return false;
				memcpy(&value, bytes + at, sizeof(value));
				at += sizeof(value);
				return true;
			}

			bool Read64(uint64_t& value)
			{
				if (!Ensure(sizeof(value)))
					return false;
				memcpy(&value, bytes + at, sizeof(value));
				at += sizeof(value);
				return true;
			}
		};

		struct RmOperand
		{
			bool isRegister{};
			uint8_t rawRegister{};
			uint8_t registerIndex{};
			bool ripRelative{};
			bool hasBase{};
			uint8_t base{};
			bool hasIndex{};
			uint8_t index{};
			uint8_t scale{};
			int64_t displacement{};
			uint8_t segment{};
			bool address32{};
		};

		struct ModRm
		{
			uint8_t raw{};
			uint8_t mod{};
			uint8_t regRaw{};
			uint8_t regIndex{};
			RmOperand rm{};
		};

		static uint64_t MaskFor(unsigned bits)
		{
			return bits == 64u ? ~0ull : ((1ull << bits) - 1ull);
		}

		static uint64_t SignFor(unsigned bits)
		{
			return 1ull << (bits - 1u);
		}

		static uint64_t SignExtend(uint64_t value, unsigned bits)
		{
			if (bits >= 64u)
				return value;
			const uint64_t sign = SignFor(bits);
			value &= MaskFor(bits);
			return (value ^ sign) - sign;
		}

		static void Mul64x64(uint64_t a, uint64_t b, uint64_t* high, uint64_t* low)
		{
			const uint64_t a0 = uint32_t(a);
			const uint64_t a1 = a >> 32u;
			const uint64_t b0 = uint32_t(b);
			const uint64_t b1 = b >> 32u;
			const uint64_t p0 = a0 * b0;
			const uint64_t p1 = a0 * b1;
			const uint64_t p2 = a1 * b0;
			const uint64_t p3 = a1 * b1;
			const uint64_t mid = (p0 >> 32u) + uint32_t(p1) + uint32_t(p2);
			*low = (p0 & 0xFFFFFFFFull) | (mid << 32u);
			*high = p3 + (p1 >> 32u) + (p2 >> 32u) + (mid >> 32u);
		}

		static bool Div128By64(uint64_t high, uint64_t low, uint64_t divisor,
			uint64_t* quotient, uint64_t* remainder)
		{
			if (!divisor || !quotient || !remainder || high >= divisor)
				return false;
			uint64_t rem = high;
			uint64_t quot = 0;
			for (int i = 0; i < 64; ++i)
			{
				const uint64_t carry = rem >> 63u;
				rem = (rem << 1u) | (low >> 63u);
				low <<= 1u;
				quot <<= 1u;
				if (carry || rem >= divisor)
				{
					rem -= divisor;
					quot |= 1u;
				}
			}
			*quotient = quot;
			*remainder = rem;
			return true;
		}

		static bool EvenParity(uint8_t value)
		{
			value ^= value >> 4u;
			value &= 0x0Fu;
			return ((0x6996u >> value) & 1u) == 0u;
		}

		class Runner
		{
		public:
			Runner(Vm::State& state, const Vm::Options& options)
				: state_(state), options_(options)
			{
			}

			Vm::Result Run();

		private:
			bool Step();
			bool Fetch(uint64_t address, uint8_t bytes[15], size_t& available);
			bool EnsureFetch(size_t required);
			bool ParsePrefixes(Cursor& cursor, Prefix& prefix);
			bool DecodeModRm(Cursor& cursor, const Prefix& prefix, ModRm& modrm);
			uint64_t EffectiveAddress(const RmOperand& operand, uint64_t nextRip,
				bool linear = true) const;
			uint64_t ReadRegister(uint8_t raw, uint8_t index, unsigned bits,
				bool rexPresent) const;
			void WriteRegister(uint8_t raw, uint8_t index, unsigned bits,
				bool rexPresent, uint64_t value);
			bool ReadRm(const ModRm& modrm, const Prefix& prefix, unsigned bits,
				uint64_t nextRip, uint64_t& value);
			bool WriteRm(const ModRm& modrm, const Prefix& prefix, unsigned bits,
				uint64_t nextRip, uint64_t value);
			bool ReadVectorRm(const ModRm& modrm, uint64_t nextRip, uint8_t value[16]);
			bool WriteVectorRm(const ModRm& modrm, uint64_t nextRip, const uint8_t value[16]);
			bool ReadYmmRm(const ModRm& modrm, uint64_t nextRip, uint8_t* value, size_t bytes);
			bool WriteYmmRm(const ModRm& modrm, uint64_t nextRip, const uint8_t* value,
				size_t bytes);
			void WriteYmmDest(uint8_t index, const uint8_t* value, size_t bytes);
			bool ExecuteVex(uint8_t lead, Cursor& cursor, uint64_t instructionRip,
				const char*& mnemonic);
			bool ReadMemory(uint64_t address, void* dst, size_t size);
			bool WriteMemory(uint64_t address, const void* src, size_t size);
			bool Push(uint64_t value);
			bool Pop(uint64_t& value);
			void SetFlag(uint64_t flag, bool enabled);
			bool Flag(uint64_t flag) const;
			void SetCommonFlags(uint64_t result, unsigned bits);
			void SetLogicFlags(uint64_t result, unsigned bits);
			void SetAddFlags(uint64_t left, uint64_t right, uint64_t result, unsigned bits);
			void SetAdcFlags(uint64_t left, uint64_t right, uint64_t carry,
				uint64_t result, unsigned bits);
			void SetSubFlags(uint64_t left, uint64_t right, uint64_t result, unsigned bits);
			void SetSbbFlags(uint64_t left, uint64_t right, uint64_t borrow,
				uint64_t result, unsigned bits);
			bool Condition(unsigned condition) const;
			bool ExecuteBinary(uint8_t opcode, Cursor& cursor, const Prefix& prefix,
				uint64_t instructionRip, const char*& mnemonic);
			bool ExecuteGroup1(uint8_t opcode, Cursor& cursor, const Prefix& prefix,
				uint64_t instructionRip, const char*& mnemonic);
			bool ExecuteShift(uint8_t opcode, Cursor& cursor, const Prefix& prefix,
				uint64_t instructionRip, const char*& mnemonic);
			bool Execute0F(Cursor& cursor, const Prefix& prefix, uint64_t instructionRip,
				const char*& mnemonic);
			bool ExecuteCall(uint64_t target, uint64_t returnRip, const char* mnemonic);
			bool ExecuteJump(uint64_t target, uint64_t fallthroughRip, const char* mnemonic);
			bool TrySuppressBranch(uint64_t target, uint64_t fallthroughRip);
			void Trace(uint64_t instructionRip, uint64_t nextRip, const uint8_t* bytes,
				size_t length, const char* mnemonic);
			bool Finish(Vm::StopReason reason, uint64_t rip, const char* mnemonic,
				uint64_t address = 0);

			Vm::State& state_;
			const Vm::Options& options_;
			Vm::Result result_{};
			uint32_t callDepth_{};
			uint64_t currentFaultAddress_{};
			uint8_t currentBytes_[15]{};
			size_t currentLength_{};
			size_t fetchedLength_{};
			uint64_t instructionRip_{};
			bool fetchFailed_{};
			bool lengthExceeded_{};
		};

		Vm::Result Runner::Run()
		{
			if (!options_.read || !options_.write)
			{
				Finish(Vm::StopReason::InvalidInstruction, state_.rip, "bad-options");
				return result_;
			}

			const uint32_t limit = options_.instructionLimit ? options_.instructionLimit : 1u;
			while (result_.instructionCount < limit)
			{
				if (options_.stopRip && state_.rip == options_.stopRip)
				{
					Finish(Vm::StopReason::StopRip, state_.rip, "stop-rip");
					return result_;
				}
				if (!Step())
					return result_;
			}

			Finish(Vm::StopReason::InstructionLimit, state_.rip, "instruction-limit");
			return result_;
		}

		bool Runner::Fetch(uint64_t address, uint8_t bytes[15], size_t& available)
		{
			available = 0;
			if (options_.fetch)
			{
				for (size_t size = 15u; size != 0u; --size)
				{
					if (options_.fetch(options_.memoryContext, address, bytes, size))
					{
						available = size;
						return true;
					}
				}
				currentFaultAddress_ = address;
				return false;
			}
			/* Prefer a full 15-byte window so ModRM/SIB/imm decode sees the whole
			 * instruction. The window may cross a 4KB page boundary. */
			if (options_.read(options_.memoryContext, address, bytes, 15u))
			{
				available = 15u;
				return true;
			}
			/* Fallback: assemble byte-by-byte so a failed cross-page read
			 * still yields the bytes available on the first page. */
			for (size_t i = 0; i < 15u; ++i)
			{
				if (!options_.read(options_.memoryContext, address + i, bytes + i, 1u))
					break;
				available = i + 1u;
			}
			if (available == 0u)
			{
				currentFaultAddress_ = address;
				return false;
			}
			return true;
		}

		bool Runner::EnsureFetch(size_t required)
		{
			if (required > sizeof(currentBytes_))
			{
				lengthExceeded_ = true;
				return false;
			}
			while (fetchedLength_ < required)
			{
				const uint64_t address = instructionRip_ + fetchedLength_;
				if (address < instructionRip_
					|| !options_.fetch(options_.memoryContext, address,
						currentBytes_ + fetchedLength_, 1))
				{
					fetchFailed_ = true;
					currentFaultAddress_ = address;
					return false;
				}
				++fetchedLength_;
			}
			return true;
		}

		bool Runner::ParsePrefixes(Cursor& cursor, Prefix& prefix)
		{
			while (cursor.Ensure(1))
			{
				const uint8_t byte = cursor.bytes[cursor.at];
				switch (byte)
				{
				case 0x66: prefix.operand16 = true; ++cursor.at; continue;
				case 0x67: prefix.address32 = true; ++cursor.at; continue;
				case 0xF2: prefix.repne = true; ++cursor.at; continue;
				case 0xF3: prefix.rep = true; ++cursor.at; continue;
				case 0x26:
				case 0x2E:
				case 0x36:
				case 0x3E:
					prefix.segment = 0;
					++cursor.at;
					continue;
				case 0xF0:
					++cursor.at;
					continue;
				case 0x64:
				case 0x65:
					if (options_.rejectSegmentOverrides) return false;
					prefix.segment = byte;
					++cursor.at;
					continue;
				default:
					if ((byte & 0xF0u) == 0x40u)
					{
						prefix.rexPresent = true;
						prefix.rex = byte;
						++cursor.at;
						continue;
					}
					return cursor.at < 15u;
				}
			}
			return false;
		}

		bool Runner::DecodeModRm(Cursor& cursor, const Prefix& prefix, ModRm& modrm)
		{
			if (prefix.address32 && !options_.useSegmentBases)
				return false;
			if (!cursor.Read8(modrm.raw))
				return false;
			modrm.mod = modrm.raw >> 6u;
			modrm.regRaw = (modrm.raw >> 3u) & 7u;
			modrm.regIndex = modrm.regRaw | (prefix.R() ? 8u : 0u);
			const uint8_t rmRaw = modrm.raw & 7u;
			modrm.rm.rawRegister = rmRaw;
			modrm.rm.registerIndex = rmRaw | (prefix.B() ? 8u : 0u);
			modrm.rm.segment = prefix.segment;
			modrm.rm.address32 = prefix.address32;

			if (modrm.mod == 3u)
			{
				modrm.rm.isRegister = true;
				return true;
			}

			if (rmRaw == 4u)
			{
				uint8_t sib = 0;
				if (!cursor.Read8(sib))
					return false;
				modrm.rm.scale = uint8_t(1u << (sib >> 6u));
				const uint8_t indexRaw = (sib >> 3u) & 7u;
				const uint8_t baseRaw = sib & 7u;
				if (indexRaw != 4u || prefix.X())
				{
					modrm.rm.hasIndex = true;
					modrm.rm.index = indexRaw | (prefix.X() ? 8u : 0u);
				}
				if (modrm.mod == 0u && baseRaw == 5u)
				{
					uint32_t disp = 0;
					if (!cursor.Read32(disp))
						return false;
					modrm.rm.displacement = int32_t(disp);
				}
				else
				{
					modrm.rm.hasBase = true;
					modrm.rm.base = baseRaw | (prefix.B() ? 8u : 0u);
				}
			}
			else if (modrm.mod == 0u && rmRaw == 5u)
			{
				uint32_t disp = 0;
				if (!cursor.Read32(disp))
					return false;
				modrm.rm.ripRelative = true;
				modrm.rm.displacement = int32_t(disp);
			}
			else
			{
				modrm.rm.hasBase = true;
				modrm.rm.base = rmRaw | (prefix.B() ? 8u : 0u);
			}

			if (modrm.mod == 1u)
			{
				uint8_t disp = 0;
				if (!cursor.Read8(disp))
					return false;
				modrm.rm.displacement += int8_t(disp);
			}
			else if (modrm.mod == 2u)
			{
				uint32_t disp = 0;
				if (!cursor.Read32(disp))
					return false;
				modrm.rm.displacement += int32_t(disp);
			}
			return true;
		}

		uint64_t Runner::EffectiveAddress(const RmOperand& operand, uint64_t nextRip,
			bool linear) const
		{
			uint64_t address = uint64_t(operand.displacement);
			if (operand.ripRelative)
				address += nextRip;
			if (operand.hasBase)
				address += state_.gpr[operand.base];
			if (operand.hasIndex)
				address += state_.gpr[operand.index] * operand.scale;
			if (options_.useSegmentBases && operand.address32)
				address = uint32_t(address);
			if (options_.useSegmentBases && linear)
			{
				if (operand.segment == 0x64) address += state_.fsBase;
				if (operand.segment == 0x65) address += state_.gsBase;
			}
			return address;
		}

		uint64_t Runner::ReadRegister(uint8_t raw, uint8_t index, unsigned bits,
			bool rexPresent) const
		{
			if (bits == 8u && !rexPresent && index < 8u && raw >= 4u)
				return (state_.gpr[raw - 4u] >> 8u) & 0xFFu;
			return state_.gpr[index] & MaskFor(bits);
		}

		void Runner::WriteRegister(uint8_t raw, uint8_t index, unsigned bits,
			bool rexPresent, uint64_t value)
		{
			value &= MaskFor(bits);
			if (bits == 64u)
				state_.gpr[index] = value;
			else if (bits == 32u)
				state_.gpr[index] = uint32_t(value);
			else if (bits == 16u)
				state_.gpr[index] = (state_.gpr[index] & ~0xFFFFull) | value;
			else if (!rexPresent && index < 8u && raw >= 4u)
			{
				const uint8_t highBase = raw - 4u;
				state_.gpr[highBase] = (state_.gpr[highBase] & ~0xFF00ull) | (value << 8u);
			}
			else
				state_.gpr[index] = (state_.gpr[index] & ~0xFFull) | value;
		}

		bool Runner::ReadMemory(uint64_t address, void* dst, size_t size)
		{
			if (options_.read(options_.memoryContext, address, dst, size))
				return true;
			currentFaultAddress_ = address;
			return false;
		}

		bool Runner::WriteMemory(uint64_t address, const void* src, size_t size)
		{
			if (options_.write(options_.memoryContext, address, src, size))
				return true;
			currentFaultAddress_ = address;
			return false;
		}

		bool Runner::ReadRm(const ModRm& modrm, const Prefix& prefix, unsigned bits,
			uint64_t nextRip, uint64_t& value)
		{
			if (modrm.rm.isRegister)
			{
				value = ReadRegister(modrm.rm.rawRegister, modrm.rm.registerIndex,
					bits, prefix.rexPresent);
				return true;
			}
			value = 0;
			return ReadMemory(EffectiveAddress(modrm.rm, nextRip), &value, bits / 8u);
		}

		bool Runner::WriteRm(const ModRm& modrm, const Prefix& prefix, unsigned bits,
			uint64_t nextRip, uint64_t value)
		{
			if (modrm.rm.isRegister)
			{
				WriteRegister(modrm.rm.rawRegister, modrm.rm.registerIndex,
					bits, prefix.rexPresent, value);
				return true;
			}
			return WriteMemory(EffectiveAddress(modrm.rm, nextRip), &value, bits / 8u);
		}

		bool Runner::ReadVectorRm(const ModRm& modrm, uint64_t nextRip, uint8_t value[16])
		{
			if (modrm.rm.isRegister)
			{
				memcpy(value, state_.ymm[modrm.rm.registerIndex], 16u);
				return true;
			}
			return ReadMemory(EffectiveAddress(modrm.rm, nextRip), value, 16u);
		}

		bool Runner::WriteVectorRm(const ModRm& modrm, uint64_t nextRip,
			const uint8_t value[16])
		{
			if (modrm.rm.isRegister)
			{
				memcpy(state_.ymm[modrm.rm.registerIndex], value, 16u);
				return true;
			}
			return WriteMemory(EffectiveAddress(modrm.rm, nextRip), value, 16u);
		}

		bool Runner::Push(uint64_t value)
		{
			state_.gpr[Vm::RSP] -= 8u;
			return WriteMemory(state_.gpr[Vm::RSP], &value, sizeof(value));
		}

		bool Runner::Pop(uint64_t& value)
		{
			if (!ReadMemory(state_.gpr[Vm::RSP], &value, sizeof(value)))
				return false;
			state_.gpr[Vm::RSP] += 8u;
			return true;
		}

		void Runner::SetFlag(uint64_t flag, bool enabled)
		{
			if (enabled)
				state_.rflags |= flag;
			else
				state_.rflags &= ~flag;
			state_.rflags |= 2u;
		}

		bool Runner::Flag(uint64_t flag) const
		{
			return (state_.rflags & flag) != 0;
		}

		void Runner::SetCommonFlags(uint64_t value, unsigned bits)
		{
			value &= MaskFor(bits);
			SetFlag(Vm::FlagZF, value == 0);
			SetFlag(Vm::FlagSF, (value & SignFor(bits)) != 0);
			SetFlag(Vm::FlagPF, EvenParity(uint8_t(value)));
		}

		void Runner::SetLogicFlags(uint64_t value, unsigned bits)
		{
			SetCommonFlags(value, bits);
			SetFlag(Vm::FlagCF, false);
			SetFlag(Vm::FlagOF, false);
			SetFlag(Vm::FlagAF, false);
		}

		void Runner::SetAddFlags(uint64_t left, uint64_t right, uint64_t result,
			unsigned bits)
		{
			SetAdcFlags(left, right, 0u, result, bits);
		}

		void Runner::SetAdcFlags(uint64_t left, uint64_t right, uint64_t carry,
			uint64_t result, unsigned bits)
		{
			const uint64_t mask = MaskFor(bits);
			const uint64_t sign = SignFor(bits);
			left &= mask;
			right &= mask;
			carry &= 1u;
			result &= mask;
			const uint64_t partial = (left + right) & mask;
			SetCommonFlags(result, bits);
			SetFlag(Vm::FlagCF, partial < left || (carry != 0u && result < partial));
			SetFlag(Vm::FlagAF, ((left ^ right ^ result) & 0x10u) != 0);
			SetFlag(Vm::FlagOF, ((~(left ^ right) & (left ^ result)) & sign) != 0);
		}

		void Runner::SetSubFlags(uint64_t left, uint64_t right, uint64_t result,
			unsigned bits)
		{
			SetSbbFlags(left, right, 0u, result, bits);
		}

		void Runner::SetSbbFlags(uint64_t left, uint64_t right, uint64_t borrow,
			uint64_t result, unsigned bits)
		{
			const uint64_t mask = MaskFor(bits);
			const uint64_t sign = SignFor(bits);
			left &= mask;
			right &= mask;
			borrow &= 1u;
			result &= mask;
			const uint64_t partial = (left - right) & mask;
			SetCommonFlags(result, bits);
			SetFlag(Vm::FlagCF, left < right || partial < borrow);
			SetFlag(Vm::FlagAF, ((left ^ right ^ result) & 0x10u) != 0);
			SetFlag(Vm::FlagOF, (((left ^ right) & (left ^ result)) & sign) != 0);
		}

		bool Runner::Condition(unsigned condition) const
		{
			const bool cf = Flag(Vm::FlagCF);
			const bool pf = Flag(Vm::FlagPF);
			const bool zf = Flag(Vm::FlagZF);
			const bool sf = Flag(Vm::FlagSF);
			const bool of = Flag(Vm::FlagOF);
			switch (condition & 15u)
			{
			case 0: return of;
			case 1: return !of;
			case 2: return cf;
			case 3: return !cf;
			case 4: return zf;
			case 5: return !zf;
			case 6: return cf || zf;
			case 7: return !cf && !zf;
			case 8: return sf;
			case 9: return !sf;
			case 10: return pf;
			case 11: return !pf;
			case 12: return sf != of;
			case 13: return sf == of;
			case 14: return zf || (sf != of);
			default: return !zf && (sf == of);
			}
		}

		bool Runner::Finish(Vm::StopReason reason, uint64_t rip, const char* mnemonic,
			uint64_t address)
		{
			if (fetchFailed_ || lengthExceeded_)
			{
				reason = fetchFailed_ ? Vm::StopReason::ReadFault
					: Vm::StopReason::InvalidInstruction;
				rip = instructionRip_;
				mnemonic = fetchFailed_ ? "fetch" : "instruction-too-long";
				address = fetchFailed_ ? currentFaultAddress_ : 0;
				currentLength_ = fetchedLength_;
			}
			else if (options_.fetch && currentLength_ == 0u)
				currentLength_ = fetchedLength_;
			result_.reason = reason;
			result_.rip = rip;
			result_.faultAddress = address;
			result_.callDepth = callDepth_;
			result_.mnemonic = mnemonic;
			result_.length = uint8_t(currentLength_);
			memcpy(result_.bytes, currentBytes_, sizeof(result_.bytes));
			return false;
		}

		void Runner::Trace(uint64_t instructionRip, uint64_t nextRip,
			const uint8_t* bytes, size_t length, const char* mnemonic)
		{
			if (!options_.trace)
				return;
			Vm::TraceRecord record{};
			record.rip = instructionRip;
			record.nextRip = nextRip;
			record.length = uint8_t(length);
			memcpy(record.bytes, bytes, length <= sizeof(record.bytes)
				? length : sizeof(record.bytes));
			record.mnemonic = mnemonic;
			record.state = &state_;
			options_.trace(options_.traceContext, &record);
		}

		bool Runner::ExecuteBinary(uint8_t opcode, Cursor& cursor, const Prefix& prefix,
			uint64_t instructionRip, const char*& mnemonic)
		{
			ModRm modrm{};
			if (!DecodeModRm(cursor, prefix, modrm))
				return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "modrm");
			currentLength_ = cursor.at;
			const uint64_t nextRip = instructionRip + cursor.at;
			const unsigned form = opcode & 3u;
			const unsigned bits = (form == 0u || form == 2u)
				? 8u : (prefix.W() ? 64u : (prefix.operand16 ? 16u : 32u));
			const unsigned operation = (opcode >> 3u) & 7u;
			const bool destinationIsRm = form < 2u;

			uint64_t rmValue = 0;
			if (!ReadRm(modrm, prefix, bits, nextRip, rmValue))
				return Finish(Vm::StopReason::ReadFault, instructionRip, "read-rm",
					currentFaultAddress_);
			const uint64_t regValue = ReadRegister(modrm.regRaw, modrm.regIndex,
				bits, prefix.rexPresent);
			const uint64_t left = destinationIsRm ? rmValue : regValue;
			const uint64_t right = destinationIsRm ? regValue : rmValue;
			const uint64_t mask = MaskFor(bits);
			uint64_t value = left;
			bool write = true;

			switch (operation)
			{
			case 0:
				mnemonic = "add";
				value = (left + right) & mask;
				SetAddFlags(left, right, value, bits);
				break;
			case 1:
				mnemonic = "or";
				value = (left | right) & mask;
				SetLogicFlags(value, bits);
				break;
			case 2:
			{
				mnemonic = "adc";
				const uint64_t carry = Flag(Vm::FlagCF) ? 1u : 0u;
				value = (left + right + carry) & mask;
				SetAdcFlags(left, right, carry, value, bits);
				break;
			}
			case 3:
			{
				mnemonic = "sbb";
				const uint64_t borrow = Flag(Vm::FlagCF) ? 1u : 0u;
				value = (left - right - borrow) & mask;
				SetSbbFlags(left, right, borrow, value, bits);
				break;
			}
			case 4:
				mnemonic = "and";
				value = (left & right) & mask;
				SetLogicFlags(value, bits);
				break;
			case 5:
				mnemonic = "sub";
				value = (left - right) & mask;
				SetSubFlags(left, right, value, bits);
				break;
			case 6:
				mnemonic = "xor";
				value = (left ^ right) & mask;
				SetLogicFlags(value, bits);
				break;
			default:
				mnemonic = "cmp";
				value = (left - right) & mask;
				SetSubFlags(left, right, value, bits);
				write = false;
				break;
			}

			if (write)
			{
				if (destinationIsRm)
				{
					if (!WriteRm(modrm, prefix, bits, nextRip, value))
						return Finish(Vm::StopReason::WriteFault, instructionRip, mnemonic,
							currentFaultAddress_);
				}
				else
					WriteRegister(modrm.regRaw, modrm.regIndex, bits,
						prefix.rexPresent, value);
			}
			state_.rip = nextRip;
			return true;
		}

		bool Runner::ExecuteGroup1(uint8_t opcode, Cursor& cursor, const Prefix& prefix,
			uint64_t instructionRip, const char*& mnemonic)
		{
			ModRm modrm{};
			if (!DecodeModRm(cursor, prefix, modrm))
				return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "group1-modrm");
			const unsigned bits = opcode == 0x80u ? 8u
				: (prefix.W() ? 64u : (prefix.operand16 ? 16u : 32u));
			uint64_t immediate = 0;
			if (opcode == 0x80u || opcode == 0x83u)
			{
				uint8_t byte = 0;
				if (!cursor.Read8(byte))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "group1-imm8");
				immediate = opcode == 0x83u ? SignExtend(byte, 8u) : byte;
			}
			else if (bits == 16u)
			{
				uint16_t word = 0;
				if (!cursor.Read16(word))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "group1-imm16");
				immediate = word;
			}
			else
			{
				uint32_t dword = 0;
				if (!cursor.Read32(dword))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "group1-imm32");
				immediate = bits == 64u ? SignExtend(dword, 32u) : dword;
			}
			currentLength_ = cursor.at;
			const uint64_t nextRip = instructionRip + cursor.at;
			uint64_t left = 0;
			if (!ReadRm(modrm, prefix, bits, nextRip, left))
				return Finish(Vm::StopReason::ReadFault, instructionRip, "group1-read",
					currentFaultAddress_);
			const uint64_t mask = MaskFor(bits);
			const uint64_t right = immediate & mask;
			uint64_t value = left;
			bool write = true;

			switch (modrm.regRaw)
			{
			case 0:
				mnemonic = "add";
				value = (left + right) & mask;
				SetAddFlags(left, right, value, bits);
				break;
			case 1:
				mnemonic = "or";
				value = (left | right) & mask;
				SetLogicFlags(value, bits);
				break;
			case 2:
			{
				mnemonic = "adc";
				const uint64_t carry = Flag(Vm::FlagCF) ? 1u : 0u;
				value = (left + right + carry) & mask;
				SetAdcFlags(left, right, carry, value, bits);
				break;
			}
			case 3:
			{
				mnemonic = "sbb";
				const uint64_t borrow = Flag(Vm::FlagCF) ? 1u : 0u;
				value = (left - right - borrow) & mask;
				SetSbbFlags(left, right, borrow, value, bits);
				break;
			}
			case 4:
				mnemonic = "and";
				value = (left & right) & mask;
				SetLogicFlags(value, bits);
				break;
			case 5:
				mnemonic = "sub";
				value = (left - right) & mask;
				SetSubFlags(left, right, value, bits);
				break;
			case 6:
				mnemonic = "xor";
				value = (left ^ right) & mask;
				SetLogicFlags(value, bits);
				break;
			default:
				mnemonic = "cmp";
				value = (left - right) & mask;
				SetSubFlags(left, right, value, bits);
				write = false;
				break;
			}

			if (write && !WriteRm(modrm, prefix, bits, nextRip, value))
				return Finish(Vm::StopReason::WriteFault, instructionRip, mnemonic,
					currentFaultAddress_);
			state_.rip = nextRip;
			return true;
		}

		bool Runner::ExecuteShift(uint8_t opcode, Cursor& cursor, const Prefix& prefix,
			uint64_t instructionRip, const char*& mnemonic)
		{
			ModRm modrm{};
			if (!DecodeModRm(cursor, prefix, modrm))
				return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "shift-modrm");
			const unsigned bits = (opcode == 0xC0u || opcode == 0xD0u || opcode == 0xD2u)
				? 8u : (prefix.W() ? 64u : (prefix.operand16 ? 16u : 32u));
			uint8_t count = 1u;
			if (opcode == 0xC0u || opcode == 0xC1u)
			{
				if (!cursor.Read8(count))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "shift-count");
			}
			else if (opcode == 0xD2u || opcode == 0xD3u)
				count = uint8_t(state_.gpr[Vm::RCX]);
			count &= bits == 64u ? 0x3Fu : 0x1Fu;
			currentLength_ = cursor.at;
			const uint64_t nextRip = instructionRip + cursor.at;
			uint64_t value = 0;
			if (!ReadRm(modrm, prefix, bits, nextRip, value))
				return Finish(Vm::StopReason::ReadFault, instructionRip, "shift-read",
					currentFaultAddress_);
			const uint64_t mask = MaskFor(bits);
			value &= mask;
			const uint64_t original = value;
			if (count)
			{
				switch (modrm.regRaw)
				{
				case 0:
					mnemonic = "rol";
					count = uint8_t(count % bits);
					if (count)
					{
						value = ((value << count) | (value >> (bits - count))) & mask;
						SetFlag(Vm::FlagCF, (value & 1u) != 0);
						if (count == 1u)
							SetFlag(Vm::FlagOF, ((value & SignFor(bits)) != 0) != Flag(Vm::FlagCF));
					}
					break;
				case 1:
					mnemonic = "ror";
					count = uint8_t(count % bits);
					if (count)
					{
						value = ((value >> count) | (value << (bits - count))) & mask;
						SetFlag(Vm::FlagCF, (value & SignFor(bits)) != 0);
						if (count == 1u)
							SetFlag(Vm::FlagOF, ((value >> (bits - 1u)) ^
								(value >> (bits - 2u))) & 1u);
					}
					break;
				case 4:
				case 6:
					mnemonic = "shl";
					if (count <= bits)
						SetFlag(Vm::FlagCF, ((original >> (bits - count)) & 1u) != 0);
					value = count < bits ? (original << count) & mask : 0;
					SetCommonFlags(value, bits);
					if (count == 1u)
						SetFlag(Vm::FlagOF, ((value & SignFor(bits)) != 0) != Flag(Vm::FlagCF));
					break;
				case 5:
					mnemonic = "shr";
					if (count <= bits)
						SetFlag(Vm::FlagCF, ((original >> (count - 1u)) & 1u) != 0);
					value = count < bits ? original >> count : 0;
					SetCommonFlags(value, bits);
					if (count == 1u)
						SetFlag(Vm::FlagOF, (original & SignFor(bits)) != 0);
					break;
				case 7:
					mnemonic = "sar";
					if (count <= bits)
						SetFlag(Vm::FlagCF, ((original >> (count - 1u)) & 1u) != 0);
					if (count >= bits)
						value = (original & SignFor(bits)) != 0 ? mask : 0u;
					else
					{
						value = original >> count;
						if ((original & SignFor(bits)) != 0)
							value |= mask ^ (mask >> count);
					}
					value &= mask;
					SetCommonFlags(value, bits);
					if (count == 1u)
						SetFlag(Vm::FlagOF, false);
					break;
				default:
					return Finish(Vm::StopReason::UnsupportedInstruction, instructionRip,
						"rcl-rcr");
				}
			}
			else
				mnemonic = "shift-zero";

			if (!WriteRm(modrm, prefix, bits, nextRip, value))
				return Finish(Vm::StopReason::WriteFault, instructionRip, mnemonic,
					currentFaultAddress_);
			state_.rip = nextRip;
			return true;
		}

#if defined(__clang__)
		__attribute__((target("aes")))
#endif
		static void AesImc128(uint8_t block[16]);
#if defined(__clang__)
		__attribute__((target("aes")))
#endif
		static void AesDec128(uint8_t state[16], const uint8_t roundKey[16], bool last);
#if defined(__clang__)
		__attribute__((target("aes")))
#endif
		static void AesEnc128(uint8_t state[16], const uint8_t roundKey[16], bool last);

		bool Runner::Execute0F(Cursor& cursor, const Prefix& prefix,
			uint64_t instructionRip, const char*& mnemonic)
		{
			uint8_t opcode = 0;
			if (!cursor.Read8(opcode))
				return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "0f-opcode");

			if (opcode == 0x05u)
			{
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				mnemonic = "syscall";
				if (!options_.syscallHook)
					return Finish(Vm::StopReason::UnsupportedInstruction,
						instructionRip, mnemonic);
				const Vm::CallHookResult hook = options_.syscallHook(
					options_.syscallHookContext, instructionRip, &state_);
				if (hook == Vm::CallHookResult::Handled)
				{
					/* SYSCALL saves the following RIP/RFLAGS in RCX/R11. Both registers
					 * are volatile in the Windows x64 syscall ABI.
					 * The hook may redirect RIP; keep its choice if it did. */
					state_.gpr[Vm::RCX] = nextRip;
					state_.gpr[Vm::R11] = state_.rflags;
					if (state_.rip == instructionRip)
						state_.rip = nextRip;
					return true;
				}
				if (hook == Vm::CallHookResult::Stop)
					return Finish(Vm::StopReason::HookStop, instructionRip, mnemonic);
				if (hook == Vm::CallHookResult::Fault)
					return Finish(Vm::StopReason::HookFault, instructionRip, mnemonic);
				return Finish(Vm::StopReason::UnsupportedInstruction,
					instructionRip, mnemonic);
			}

			if (opcode >= 0x80u && opcode <= 0x8Fu)
			{
				uint32_t rel = 0;
				if (!cursor.Read32(rel))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "jcc-rel32");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				const uint64_t target = nextRip + int32_t(rel);
				mnemonic = "jcc";
				result_.branchTarget = target;
				if (Condition(opcode & 15u))
				{
					if (TrySuppressBranch(target, nextRip))
						return true;
					state_.rip = target;
				}
				else
					state_.rip = nextRip;
				return true;
			}

			if (opcode >= 0x90u && opcode <= 0x9Fu)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "setcc-modrm");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				mnemonic = "setcc";
				if (!WriteRm(modrm, prefix, 8u, nextRip,
					Condition(opcode & 15u) ? 1u : 0u))
					return Finish(Vm::StopReason::WriteFault, instructionRip, mnemonic,
						currentFaultAddress_);
				state_.rip = nextRip;
				return true;
			}

			if (opcode >= 0x40u && opcode <= 0x4Fu)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "cmov-modrm");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				const unsigned bits = prefix.W() ? 64u : (prefix.operand16 ? 16u : 32u);
				uint64_t value = 0;
				if (!ReadRm(modrm, prefix, bits, nextRip, value))
					return Finish(Vm::StopReason::ReadFault, instructionRip, "cmov-read",
						currentFaultAddress_);
				mnemonic = "cmovcc";
				if (Condition(opcode & 15u))
					WriteRegister(modrm.regRaw, modrm.regIndex, bits,
						prefix.rexPresent, value);
				state_.rip = nextRip;
				return true;
			}

			if (opcode == 0x1Fu)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "nop-modrm");
				currentLength_ = cursor.at;
				mnemonic = "nop";
				state_.rip = instructionRip + cursor.at;
				return true;
			}

			/* Bit-test group used by the current Native slot.  The register form
			 * (for example `66 0F BA E3 0C`) keeps the operand untouched for BT and
			 * uses the old bit as CF for all four variants.  The memory form also
			 * follows Intel's immediate-index addressing rule: bits beyond the
			 * operand width select a subsequent natural-sized word. */
			if (opcode == 0xBAu)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm)
					|| modrm.regRaw < 4u || modrm.regRaw > 7u)
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip,
						"bt-group-modrm");
				uint8_t immediate = 0;
				if (!cursor.Read8(immediate))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip,
						"bt-group-imm8");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				const unsigned bits = prefix.W() ? 64u
					: (prefix.operand16 ? 16u : 32u);
				const unsigned byteWidth = bits / 8u;
				uint64_t value = 0;
				uint64_t bit = 0;
				if (modrm.rm.isRegister)
				{
					bit = uint64_t(immediate) & uint64_t(bits - 1u);
					value = ReadRegister(modrm.rm.rawRegister,
						modrm.rm.registerIndex, bits, prefix.rexPresent);
				}
				else
				{
					const uint64_t bitIndex = immediate;
					const uint64_t wordOffset = (bitIndex / bits) * byteWidth;
					bit = bitIndex % bits;
					const uint64_t address = EffectiveAddress(modrm.rm, nextRip)
						+ wordOffset;
					if (!ReadMemory(address, &value, byteWidth))
						return Finish(Vm::StopReason::ReadFault, instructionRip,
							"bt-group-read", currentFaultAddress_);
				}
				const uint64_t mask = 1ull << bit;
				SetFlag(Vm::FlagCF, (value & mask) != 0u);
				if (modrm.regRaw != 4u)
				{
					if (modrm.regRaw == 5u)
						value |= mask;
					else if (modrm.regRaw == 6u)
						value &= ~mask;
					else
						value ^= mask;
					if (modrm.rm.isRegister)
						WriteRegister(modrm.rm.rawRegister,
							modrm.rm.registerIndex, bits, prefix.rexPresent, value);
					else
					{
						const uint64_t address = EffectiveAddress(modrm.rm, nextRip)
							+ (uint64_t(immediate) / bits) * byteWidth;
						if (!WriteMemory(address, &value, byteWidth))
							return Finish(Vm::StopReason::WriteFault, instructionRip,
								"bt-group-write", currentFaultAddress_);
					}
				}
				mnemonic = modrm.regRaw == 4u ? "bt"
					: (modrm.regRaw == 5u ? "bts"
						: (modrm.regRaw == 6u ? "btr" : "btc"));
				state_.rip = nextRip;
				return true;
			}

			/* Register bit-test forms
			 * (for example `48 0F A3 C6` / `bt rsi, rax`).  Immediate forms are
			 * handled above via 0F BA; these four opcodes carry the bit index in
			 * the register operand instead. */
			if (opcode == 0xA3u || opcode == 0xABu || opcode == 0xB3u
				|| opcode == 0xBBu)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip,
						"bt-reg-modrm");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				const unsigned bits = prefix.W() ? 64u
					: (prefix.operand16 ? 16u : 32u);
				const unsigned byteWidth = bits / 8u;
				const uint64_t bitOffset = ReadRegister(modrm.regRaw, modrm.regIndex,
					bits, prefix.rexPresent);
				uint64_t value = 0;
				uint64_t bit = 0;
				uint64_t address = 0;
				if (modrm.rm.isRegister)
				{
					bit = bitOffset & uint64_t(bits - 1u);
					value = ReadRegister(modrm.rm.rawRegister,
						modrm.rm.registerIndex, bits, prefix.rexPresent);
				}
				else
				{
					bit = bitOffset % bits;
					address = EffectiveAddress(modrm.rm, nextRip)
						+ (bitOffset / bits) * byteWidth;
					if (!ReadMemory(address, &value, byteWidth))
						return Finish(Vm::StopReason::ReadFault, instructionRip,
							"bt-reg-read", currentFaultAddress_);
				}
				const uint64_t mask = 1ull << bit;
				SetFlag(Vm::FlagCF, (value & mask) != 0u);
				if (opcode != 0xA3u)
				{
					if (opcode == 0xABu)
						value |= mask;
					else if (opcode == 0xB3u)
						value &= ~mask;
					else
						value ^= mask;
					if (modrm.rm.isRegister)
						WriteRegister(modrm.rm.rawRegister,
							modrm.rm.registerIndex, bits, prefix.rexPresent, value);
					else if (!WriteMemory(address, &value, byteWidth))
						return Finish(Vm::StopReason::WriteFault, instructionRip,
							"bt-reg-write", currentFaultAddress_);
				}
				mnemonic = opcode == 0xA3u ? "bt"
					: (opcode == 0xABu ? "bts"
						: (opcode == 0xB3u ? "btr" : "btc"));
				state_.rip = nextRip;
				return true;
			}

			/* xadd r/m, reg (e.g. lock xadd qword ptr [r8], r9 =
			 * F0 4D 0F C1 08).  Treat LOCK as a no-op; emulate the
			 * exchange-add and the ADD-style flags. */
			if (opcode == 0xC0u || opcode == 0xC1u)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip,
						"xadd-modrm");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				const unsigned bits = opcode == 0xC0u ? 8u
					: (prefix.W() ? 64u : (prefix.operand16 ? 16u : 32u));
				uint64_t destination = 0;
				if (!ReadRm(modrm, prefix, bits, nextRip, destination))
					return Finish(Vm::StopReason::ReadFault, instructionRip,
						"xadd-read", currentFaultAddress_);
				const uint64_t source = ReadRegister(modrm.regRaw, modrm.regIndex,
					bits, prefix.rexPresent);
				const uint64_t mask = MaskFor(bits);
				const uint64_t sum = (destination + source) & mask;
				SetAddFlags(destination, source, sum, bits);
				if (!WriteRm(modrm, prefix, bits, nextRip, sum))
					return Finish(Vm::StopReason::WriteFault, instructionRip,
						"xadd-write", currentFaultAddress_);
				WriteRegister(modrm.regRaw, modrm.regIndex, bits,
					prefix.rexPresent, destination);
				mnemonic = "xadd";
				state_.rip = nextRip;
				return true;
			}

			/* cmpxchg r/m, reg (e.g. lock cmpxchg qword ptr [rsi], rcx =
			 * F0 48 0F B1 0E), as used by Interlocked* primitives. */
			if (opcode == 0xB0u || opcode == 0xB1u)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip,
						"cmpxchg-modrm");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				const unsigned bits = opcode == 0xB0u ? 8u
					: (prefix.W() ? 64u : (prefix.operand16 ? 16u : 32u));
				const uint64_t mask = MaskFor(bits);
				uint64_t destination = 0;
				if (!ReadRm(modrm, prefix, bits, nextRip, destination))
					return Finish(Vm::StopReason::ReadFault, instructionRip,
						"cmpxchg-read", currentFaultAddress_);
				const uint64_t accumulator = ReadRegister(0u, Vm::RAX, bits,
					prefix.rexPresent);
				const uint64_t compare = (accumulator - destination) & mask;
				SetSubFlags(accumulator, destination, compare, bits);
				if ((accumulator & mask) == (destination & mask))
				{
					const uint64_t source = ReadRegister(modrm.regRaw, modrm.regIndex,
						bits, prefix.rexPresent);
					if (!WriteRm(modrm, prefix, bits, nextRip, source & mask))
						return Finish(Vm::StopReason::WriteFault, instructionRip,
							"cmpxchg-write", currentFaultAddress_);
				}
				else
					WriteRegister(0u, Vm::RAX, bits, prefix.rexPresent, destination);
				mnemonic = "cmpxchg";
				state_.rip = nextRip;
				return true;
			}

			if (opcode == 0xAFu || opcode == 0xB6u || opcode == 0xB7u
				|| opcode == 0xBEu || opcode == 0xBFu)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "0f-gpr-modrm");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				const unsigned destinationBits = prefix.W() ? 64u
					: (prefix.operand16 ? 16u : 32u);
				if (opcode == 0xAFu)
				{
					uint64_t right = 0;
					if (!ReadRm(modrm, prefix, destinationBits, nextRip, right))
						return Finish(Vm::StopReason::ReadFault, instructionRip, "imul-read",
							currentFaultAddress_);
					const uint64_t left = ReadRegister(modrm.regRaw, modrm.regIndex,
						destinationBits, prefix.rexPresent);
					uint64_t value = 0;
					bool overflow = false;
					if (destinationBits == 16u)
					{
						const int32_t product = int16_t(left) * int16_t(right);
						value = uint16_t(product);
						overflow = product != int16_t(value);
					}
					else if (destinationBits == 32u)
					{
						const int64_t product = int64_t(int32_t(left)) * int64_t(int32_t(right));
						value = uint32_t(product);
						overflow = product != int64_t(int32_t(value));
					}
					else
					{
						const int64_t signedLeft = int64_t(left);
						const int64_t signedRight = int64_t(right);
						value = uint64_t(signedLeft) * uint64_t(signedRight);
						if (signedRight == -1)
							overflow = signedLeft == INT64_MIN;
						else if (signedRight != 0)
							overflow = int64_t(value) / signedRight != signedLeft;
					}
					WriteRegister(modrm.regRaw, modrm.regIndex, destinationBits,
						prefix.rexPresent, value);
					SetFlag(Vm::FlagCF, overflow);
					SetFlag(Vm::FlagOF, overflow);
					mnemonic = "imul";
				}
				else
				{
					const unsigned sourceBits = (opcode == 0xB6u || opcode == 0xBEu)
						? 8u : 16u;
					uint64_t value = 0;
					if (!ReadRm(modrm, prefix, sourceBits, nextRip, value))
						return Finish(Vm::StopReason::ReadFault, instructionRip, "movx-read",
							currentFaultAddress_);
					if (opcode == 0xBEu || opcode == 0xBFu)
						value = SignExtend(value, sourceBits);
					WriteRegister(modrm.regRaw, modrm.regIndex, destinationBits,
						prefix.rexPresent, value);
					mnemonic = (opcode == 0xBEu || opcode == 0xBFu) ? "movsx" : "movzx";
				}
				state_.rip = nextRip;
				return true;
			}

			/* 0F 38 three-byte map: legacy (non-VEX) AES-NI,
			 * 66 0F 38 DB/DC/DD/DE/DF (aesimc/aesenc/aesenclast/aesdec/aesdeclast). */
			if (opcode == 0x38u)
			{
				uint8_t secondary = 0;
				if (!cursor.Read8(secondary))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip,
						"0f38-op");
				if (secondary == 0xDBu || secondary == 0xDCu || secondary == 0xDDu
					|| secondary == 0xDEu || secondary == 0xDFu)
				{
					if (!prefix.operand16)
						return Finish(Vm::StopReason::UnsupportedInstruction,
							instructionRip, "aes-need-66");
					ModRm modrm{};
					if (!DecodeModRm(cursor, prefix, modrm))
						return Finish(Vm::StopReason::InvalidInstruction,
							instructionRip, "aes-modrm");
					currentLength_ = cursor.at;
					const uint64_t nextRip = instructionRip + cursor.at;
					uint8_t right[16]{};
					if (!ReadVectorRm(modrm, nextRip, right))
						return Finish(Vm::StopReason::ReadFault, instructionRip,
							"aes-read", currentFaultAddress_);
					uint8_t stateBytes[16]{};
					if (secondary == 0xDBu)
					{
						memcpy(stateBytes, right, sizeof(stateBytes));
						AesImc128(stateBytes);
						mnemonic = "aesimc";
					}
					else
					{
						memcpy(stateBytes, state_.ymm[modrm.regIndex], 16u);
						const bool last = secondary == 0xDFu || secondary == 0xDDu;
						const bool enc = secondary == 0xDCu || secondary == 0xDDu;
						if (enc)
							AesEnc128(stateBytes, right, last);
						else
							AesDec128(stateBytes, right, last);
						mnemonic = secondary == 0xDCu ? "aesenc"
							: (secondary == 0xDDu ? "aesenclast"
								: (secondary == 0xDEu ? "aesdec" : "aesdeclast"));
					}
					memcpy(state_.ymm[modrm.regIndex], stateBytes, 16u);
					memset(state_.ymm[modrm.regIndex] + 16u, 0, 16u);
					state_.rip = nextRip;
					return true;
				}
				currentLength_ = cursor.at;
				return Finish(Vm::StopReason::UnsupportedInstruction, instructionRip,
					"0f38-unsupported");
			}

			const bool vectorOpcode = opcode == 0x10u || opcode == 0x11u
				|| opcode == 0x12u || opcode == 0x28u || opcode == 0x29u || opcode == 0x50u
				|| opcode == 0x54u || opcode == 0x56u || opcode == 0x57u
				|| opcode == 0x6Cu || opcode == 0x6Eu || opcode == 0x6Fu
				|| opcode == 0x6Du
				|| opcode == 0x70u || opcode == 0x76u || opcode == 0x7Eu
				|| opcode == 0x7Fu || opcode == 0xC6u || opcode == 0xD6u || opcode == 0xEFu
				|| opcode == 0xDBu || opcode == 0xEBu || opcode == 0xD4u;
			if (!vectorOpcode)
			{
				/* Keep the 0F xx prefix+opcode bytes so the caller can see which
				 * secondary opcode still needs coverage. */
				currentLength_ = cursor.at;
				return Finish(Vm::StopReason::UnsupportedInstruction, instructionRip,
					"0f-unsupported");
			}

			ModRm modrm{};
			if (!DecodeModRm(cursor, prefix, modrm))
				return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "sse-modrm");
			uint8_t shuffle = 0;
			if ((opcode == 0x70u || opcode == 0xC6u) && !cursor.Read8(shuffle))
				return Finish(Vm::StopReason::InvalidInstruction, instructionRip,
					opcode == 0xC6u ? "shufps-imm" : "pshufd-imm");
			currentLength_ = cursor.at;
			const uint64_t nextRip = instructionRip + cursor.at;

            // Legacy SSE MOVSS/MOVSD use 4/8-byte memory operands, not m128.
            // Memory loads clear the rest of XMM; register moves merge only
            // the scalar lane. Legacy encodings preserve the upper YMM half.
            if ((opcode == 0x10u || opcode == 0x11u)
                && (prefix.rep || prefix.repne))
            {
                const size_t bytes = prefix.rep ? 4u : 8u;
                uint8_t scalar[8]{};
                if (opcode == 0x10u)
                {
                    if (modrm.rm.isRegister)
                        memcpy(scalar, state_.ymm[modrm.rm.registerIndex], bytes);
                    else
                    {
                        const uint64_t address = EffectiveAddress(modrm.rm, nextRip);
                        if (!ReadMemory(address, scalar, bytes))
                            return Finish(Vm::StopReason::ReadFault, instructionRip,
                                "scalar-xmm-read", currentFaultAddress_);
                        memset(state_.ymm[modrm.regIndex], 0, 16u);
                    }
                    memcpy(state_.ymm[modrm.regIndex], scalar, bytes);
                }
                else
                {
                    memcpy(scalar, state_.ymm[modrm.regIndex], bytes);
                    if (modrm.rm.isRegister)
                        memcpy(state_.ymm[modrm.rm.registerIndex], scalar, bytes);
                    else
                    {
                        const uint64_t address = EffectiveAddress(modrm.rm, nextRip);
                        if (!WriteMemory(address, scalar, bytes))
                            return Finish(Vm::StopReason::WriteFault, instructionRip,
                                "scalar-xmm-write", currentFaultAddress_);
                    }
                }
                mnemonic = prefix.rep ? "movss" : "movsd";
            }
            else if (opcode == 0x6Fu || opcode == 0x10u || opcode == 0x28u)
			{
				uint8_t value[16]{};
				if (!ReadVectorRm(modrm, nextRip, value))
					return Finish(Vm::StopReason::ReadFault, instructionRip, "xmm-read",
						currentFaultAddress_);
				memcpy(state_.ymm[modrm.regIndex], value, sizeof(value));
				mnemonic = opcode == 0x6Fu ? "movdqa" : (opcode == 0x28u ? "movaps" : "movups");
			}
			else if (opcode == 0x7Fu || opcode == 0x11u || opcode == 0x29u)
			{
				if (!WriteVectorRm(modrm, nextRip, state_.ymm[modrm.regIndex]))
					return Finish(Vm::StopReason::WriteFault, instructionRip, "xmm-write",
						currentFaultAddress_);
				mnemonic = opcode == 0x7Fu ? "movdqa" : (opcode == 0x29u ? "movaps" : "movups");
			}
			else if (opcode == 0x12u)
			{
				if (modrm.rm.isRegister)
				{
					/* MOVHLPS xmm1, xmm2: low qword <- src high qword. */
					memcpy(state_.ymm[modrm.regIndex],
						state_.ymm[modrm.rm.registerIndex] + 8u, 8u);
					mnemonic = "movhlps";
				}
				else
				{
					uint8_t value[8]{};
					const uint64_t address = EffectiveAddress(modrm.rm, nextRip);
					if (!ReadMemory(address, value, sizeof(value)))
						return Finish(Vm::StopReason::ReadFault, instructionRip,
							"movlps-read", currentFaultAddress_);
					memcpy(state_.ymm[modrm.regIndex], value, sizeof(value));
					mnemonic = "movlps";
				}
			}
			else if (opcode == 0xEFu || opcode == 0x57u || opcode == 0x76u
				|| opcode == 0x54u || opcode == 0x56u
				|| opcode == 0xDBu || opcode == 0xEBu)
			{
				uint8_t right[16]{};
				if (!ReadVectorRm(modrm, nextRip, right))
					return Finish(Vm::StopReason::ReadFault, instructionRip, "xmm-op-read",
						currentFaultAddress_);
				if (opcode == 0x76u)
				{
					for (unsigned i = 0; i < 4u; ++i)
					{
						uint32_t leftWord = 0, rightWord = 0, resultWord = 0;
						memcpy(&leftWord, state_.ymm[modrm.regIndex] + i * 4u, 4u);
						memcpy(&rightWord, right + i * 4u, 4u);
						resultWord = leftWord == rightWord ? 0xFFFFFFFFu : 0u;
						memcpy(state_.ymm[modrm.regIndex] + i * 4u, &resultWord, 4u);
					}
					mnemonic = "pcmpeqd";
				}
				else
				{
					for (unsigned i = 0; i < 16u; ++i)
					{
						uint8_t& left = state_.ymm[modrm.regIndex][i];
						if (opcode == 0x54u || opcode == 0xDBu)
							left &= right[i];
						else if (opcode == 0x56u || opcode == 0xEBu)
							left |= right[i];
						else
							left ^= right[i];
					}
					mnemonic = opcode == 0xEFu ? "pxor"
						: (opcode == 0x54u ? "andps"
							: (opcode == 0x56u ? "orps"
								: (opcode == 0xDBu ? "pand"
									: (opcode == 0xEBu ? "por" : "xorps"))));
				}
			}
			else if (opcode == 0xD4u)
			{
				/* paddq xmm1, xmm2/m128 */
				if (!prefix.operand16)
					return Finish(Vm::StopReason::UnsupportedInstruction,
						instructionRip, "paddq-prefix");
				uint8_t right[16]{};
				if (!ReadVectorRm(modrm, nextRip, right))
					return Finish(Vm::StopReason::ReadFault, instructionRip,
						"paddq-read", currentFaultAddress_);
				for (unsigned i = 0; i < 2u; ++i)
				{
					uint64_t a = 0, b = 0;
					memcpy(&a, state_.ymm[modrm.regIndex] + i * 8u, 8u);
					memcpy(&b, right + i * 8u, 8u);
					a += b;
					memcpy(state_.ymm[modrm.regIndex] + i * 8u, &a, 8u);
				}
				mnemonic = "paddq";
			}
			else if (opcode == 0x70u || opcode == 0xC6u)
			{
				uint8_t source[16]{}, output[16]{};
				if (!ReadVectorRm(modrm, nextRip, source))
					return Finish(Vm::StopReason::ReadFault, instructionRip,
						opcode == 0xC6u ? "shufps-read" : "pshufd-read",
						currentFaultAddress_);
				if (opcode == 0xC6u)
				{
					uint8_t destination[16]{};
					memcpy(destination, state_.ymm[modrm.regIndex], sizeof(destination));
					for (unsigned i = 0; i < 2u; ++i)
						memcpy(output + i * 4u,
							destination + ((shuffle >> (i * 2u)) & 3u) * 4u, 4u);
					for (unsigned i = 0; i < 2u; ++i)
						memcpy(output + (i + 2u) * 4u,
							source + ((shuffle >> ((i + 2u) * 2u)) & 3u) * 4u, 4u);
					mnemonic = "shufps";
				}
				else
				{
					for (unsigned i = 0; i < 4u; ++i)
						memcpy(output + i * 4u, source + ((shuffle >> (i * 2u)) & 3u) * 4u, 4u);
					mnemonic = "pshufd";
				}
				memcpy(state_.ymm[modrm.regIndex], output, sizeof(output));
			}
			else if (opcode == 0x6Cu)
			{
				if (!prefix.operand16)
					return Finish(Vm::StopReason::UnsupportedInstruction,
						instructionRip, "punpcklqdq-prefix");
				uint8_t source[16]{};
				if (!ReadVectorRm(modrm, nextRip, source))
					return Finish(Vm::StopReason::ReadFault, instructionRip,
						"punpcklqdq-read", currentFaultAddress_);
				memcpy(state_.ymm[modrm.regIndex] + 8u, source, 8u);
				mnemonic = "punpcklqdq";
			}
			else if (opcode == 0x6Du)
			{
				if (!prefix.operand16)
					return Finish(Vm::StopReason::UnsupportedInstruction,
						instructionRip, "punpckhqdq-prefix");
				/* PUNPCKHQDQ consumes the high qword from both operands and
				 * writes them to the low 128-bit destination.  Keep both source
				 * snapshots first because the common register form can alias. */
				uint8_t destination[16]{};
				uint8_t source[16]{};
				memcpy(destination, state_.ymm[modrm.regIndex], sizeof(destination));
				if (!ReadVectorRm(modrm, nextRip, source))
					return Finish(Vm::StopReason::ReadFault, instructionRip,
						"punpckhqdq-read", currentFaultAddress_);
				memcpy(state_.ymm[modrm.regIndex] + 0u, destination + 8u, 8u);
				memcpy(state_.ymm[modrm.regIndex] + 8u, source + 8u, 8u);
				mnemonic = "punpckhqdq";
			}
			else if (opcode == 0x7Eu || opcode == 0xD6u)
			{
				const unsigned bits = prefix.W() || opcode == 0xD6u ? 64u : 32u;
				uint64_t value = 0;
				memcpy(&value, state_.ymm[modrm.regIndex], bits / 8u);
				if (!WriteRm(modrm, prefix, bits, nextRip, value))
					return Finish(Vm::StopReason::WriteFault, instructionRip, "movq-write",
						currentFaultAddress_);
				mnemonic = bits == 64u ? "movq" : "movd";
			}
			else if (opcode == 0x6Eu)
			{
				const unsigned bits = prefix.W() ? 64u : 32u;
				uint64_t value = 0;
				if (!ReadRm(modrm, prefix, bits, nextRip, value))
					return Finish(Vm::StopReason::ReadFault, instructionRip, "movq-read",
						currentFaultAddress_);
				memset(state_.ymm[modrm.regIndex], 0, 16u);
				memcpy(state_.ymm[modrm.regIndex], &value, bits / 8u);
				mnemonic = bits == 64u ? "movq" : "movd";
			}
			else if (opcode == 0x50u)
			{
				uint8_t source[16]{};
				if (!ReadVectorRm(modrm, nextRip, source))
					return Finish(Vm::StopReason::ReadFault, instructionRip, "movmsk-read",
						currentFaultAddress_);
				uint32_t mask = 0;
				for (unsigned i = 0; i < 4u; ++i)
					mask |= uint32_t((source[i * 4u + 3u] >> 7u) & 1u) << i;
				WriteRegister(modrm.regRaw, modrm.regIndex, 32u,
					prefix.rexPresent, mask);
				mnemonic = "movmskps";
			}

			state_.rip = nextRip;
			return true;
		}

#if defined(__clang__)
		__attribute__((target("aes")))
#endif
		static void AesImc128(uint8_t block[16])
		{
			__m128i s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block));
			s = _mm_aesimc_si128(s);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(block), s);
		}

#if defined(__clang__)
		__attribute__((target("aes")))
#endif
		static void AesDec128(uint8_t state[16], const uint8_t roundKey[16],
			bool last)
		{
			__m128i s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(state));
			__m128i k = _mm_loadu_si128(reinterpret_cast<const __m128i*>(roundKey));
			s = last ? _mm_aesdeclast_si128(s, k) : _mm_aesdec_si128(s, k);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(state), s);
		}

#if defined(__clang__)
		__attribute__((target("aes")))
#endif
		static void AesEnc128(uint8_t state[16], const uint8_t roundKey[16],
			bool last)
		{
			__m128i s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(state));
			__m128i k = _mm_loadu_si128(reinterpret_cast<const __m128i*>(roundKey));
			s = last ? _mm_aesenclast_si128(s, k) : _mm_aesenc_si128(s, k);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(state), s);
		}

#include "MiniX64VmVex.inl"

		bool Runner::TrySuppressBranch(uint64_t target, uint64_t fallthroughRip)
		{
			if (!options_.branchHook)
				return false;
			return options_.branchHook(options_.branchHookContext, target,
				fallthroughRip, &state_);
		}

		bool Runner::ExecuteCall(uint64_t target, uint64_t returnRip, const char* mnemonic)
		{
			result_.branchTarget = target;
			if (TrySuppressBranch(target, returnRip))
				return true;
			if (options_.callHook)
			{
				const Vm::CallHookResult hook = options_.callHook(
					options_.callHookContext, target, &state_);
				if (hook == Vm::CallHookResult::Handled)
				{
					state_.rip = returnRip;
					return true;
				}
				if (hook == Vm::CallHookResult::Stop)
					return Finish(Vm::StopReason::HookStop, state_.rip, mnemonic, target);
				if (hook == Vm::CallHookResult::Fault)
					return Finish(Vm::StopReason::HookFault, state_.rip, mnemonic, target);
			}

			const uint32_t depthLimit = options_.callDepthLimit ? options_.callDepthLimit : 1u;
			if (callDepth_ >= depthLimit)
				return Finish(Vm::StopReason::CallDepthLimit, state_.rip, mnemonic, target);
			if (!Push(returnRip))
				return Finish(Vm::StopReason::WriteFault, state_.rip, mnemonic,
					currentFaultAddress_);
			++callDepth_;
			state_.rip = target;
			return true;
		}

		bool Runner::ExecuteJump(uint64_t target, uint64_t fallthroughRip,
			const char* mnemonic)
		{
			result_.branchTarget = target;
			if (TrySuppressBranch(target, fallthroughRip))
				return true;
			if (options_.callHook)
			{
				const Vm::CallHookResult hook = options_.callHook(
					options_.callHookContext, target, &state_);
				/* Handled on jmp means an external/API transfer was stubbed.
				 * Import thunks are `jmp ntdll`; emulate an immediate return to
				 * the stacked caller instead of entering the system image. */
				if (hook == Vm::CallHookResult::Handled)
				{
					uint64_t returnRip = 0;
					if (!Pop(returnRip))
						return Finish(Vm::StopReason::ReadFault, state_.rip, mnemonic,
							currentFaultAddress_);
					if (callDepth_ > 0u)
						--callDepth_;
					state_.rip = returnRip;
					return true;
				}
				if (hook == Vm::CallHookResult::Stop)
					return Finish(Vm::StopReason::HookStop, state_.rip, mnemonic, target);
				if (hook == Vm::CallHookResult::Fault)
					return Finish(Vm::StopReason::HookFault, state_.rip, mnemonic, target);
			}
			state_.rip = target;
			return true;
		}

		bool Runner::Step()
		{
			const uint64_t instructionRip = state_.rip;
			memset(currentBytes_, 0, sizeof(currentBytes_));
			currentLength_ = 0;
			currentFaultAddress_ = 0;
			fetchedLength_ = 0;
			instructionRip_ = instructionRip;
			fetchFailed_ = false;
			lengthExceeded_ = false;
			size_t available = 0;
			if (!Fetch(instructionRip, currentBytes_, available))
				return Finish(Vm::StopReason::ReadFault, instructionRip, "fetch",
					currentFaultAddress_);

			Cursor cursor{ currentBytes_, available, 0 };
			if (options_.fetch)
			{
				fetchedLength_ = available;
				cursor.context = this;
				cursor.ensure = [](void* context, size_t required)
				{
					return static_cast<Runner*>(context)->EnsureFetch(required);
				};
			}
			Prefix prefix{};
			if (!ParsePrefixes(cursor, prefix))
				return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "prefix");
			uint8_t opcode = 0;
			if (!cursor.Read8(opcode))
				return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "opcode");
			if (options_.useSegmentBases && opcode != 0x8D
				&& ((prefix.segment == 0x64 && !state_.fsBase)
					|| (prefix.segment == 0x65 && !state_.gsBase)))
			{
				currentLength_ = cursor.at;
				return Finish(Vm::StopReason::InvalidInstruction, instructionRip,
					prefix.segment == 0x65 ? "gs-base-missing" : "fs-base-missing");
			}
			const char* mnemonic = "unknown";
			bool ok = true;

			const uint8_t binaryBase = opcode & 0xF8u;
			if ((opcode & 7u) <= 3u && (binaryBase == 0x00u || binaryBase == 0x08u
				|| binaryBase == 0x10u || binaryBase == 0x18u || binaryBase == 0x20u
				|| binaryBase == 0x28u || binaryBase == 0x30u || binaryBase == 0x38u))
			{
				ok = ExecuteBinary(opcode, cursor, prefix, instructionRip, mnemonic);
			}
			else if (opcode == 0x80u || opcode == 0x81u || opcode == 0x83u)
			{
				ok = ExecuteGroup1(opcode, cursor, prefix, instructionRip, mnemonic);
			}
			else if (opcode == 0xC0u || opcode == 0xC1u || opcode == 0xD0u
				|| opcode == 0xD1u || opcode == 0xD2u || opcode == 0xD3u)
			{
				ok = ExecuteShift(opcode, cursor, prefix, instructionRip, mnemonic);
			}
			else if (opcode == 0x0Fu)
			{
				ok = Execute0F(cursor, prefix, instructionRip, mnemonic);
			}
			else if (opcode == 0xC4u || opcode == 0xC5u)
			{
				ok = ExecuteVex(opcode, cursor, instructionRip, mnemonic);
			}
			else if (opcode >= 0x50u && opcode <= 0x57u)
			{
				const uint8_t index = uint8_t(opcode - 0x50u) | (prefix.B() ? 8u : 0u);
				currentLength_ = cursor.at;
				mnemonic = "push";
				if (!Push(state_.gpr[index]))
					return Finish(Vm::StopReason::WriteFault, instructionRip, mnemonic,
						currentFaultAddress_);
				state_.rip = instructionRip + cursor.at;
			}
			else if (opcode >= 0x58u && opcode <= 0x5Fu)
			{
				const uint8_t index = uint8_t(opcode - 0x58u) | (prefix.B() ? 8u : 0u);
				uint64_t value = 0;
				currentLength_ = cursor.at;
				mnemonic = "pop";
				if (!Pop(value))
					return Finish(Vm::StopReason::ReadFault, instructionRip, mnemonic,
						currentFaultAddress_);
				state_.gpr[index] = value;
				state_.rip = instructionRip + cursor.at;
			}
			else if (opcode == 0x68u || opcode == 0x6Au)
			{
				uint64_t value = 0;
				if (opcode == 0x6Au)
				{
					uint8_t immediate = 0;
					if (!cursor.Read8(immediate))
						return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "push-imm8");
					value = SignExtend(immediate, 8u);
				}
				else
				{
					uint32_t immediate = 0;
					if (!cursor.Read32(immediate))
						return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "push-imm32");
					value = SignExtend(immediate, 32u);
				}
				currentLength_ = cursor.at;
				mnemonic = "push";
				if (!Push(value))
					return Finish(Vm::StopReason::WriteFault, instructionRip, mnemonic,
						currentFaultAddress_);
				state_.rip = instructionRip + cursor.at;
			}
			else if (opcode >= 0x70u && opcode <= 0x7Fu)
			{
				uint8_t rel = 0;
				if (!cursor.Read8(rel))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "jcc-rel8");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				const uint64_t target = nextRip + int8_t(rel);
				result_.branchTarget = target;
				mnemonic = "jcc";
				if (Condition(opcode & 15u))
				{
					if (TrySuppressBranch(target, nextRip))
						ok = true;
					else
					{
						state_.rip = target;
						ok = true;
					}
				}
				else
				{
					state_.rip = nextRip;
					ok = true;
				}
			}
			else if (opcode == 0x84u || opcode == 0x85u)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "test-modrm");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				const unsigned bits = opcode == 0x84u ? 8u
					: (prefix.W() ? 64u : (prefix.operand16 ? 16u : 32u));
				uint64_t rmValue = 0;
				if (!ReadRm(modrm, prefix, bits, nextRip, rmValue))
					return Finish(Vm::StopReason::ReadFault, instructionRip, "test-read",
						currentFaultAddress_);
				const uint64_t regValue = ReadRegister(modrm.regRaw, modrm.regIndex,
					bits, prefix.rexPresent);
				SetLogicFlags(rmValue & regValue, bits);
				mnemonic = "test";
				state_.rip = nextRip;
			}
			else if (opcode == 0x86u || opcode == 0x87u)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "xchg-modrm");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				const unsigned bits = opcode == 0x86u ? 8u
					: (prefix.W() ? 64u : (prefix.operand16 ? 16u : 32u));
				uint64_t rmValue = 0;
				if (!ReadRm(modrm, prefix, bits, nextRip, rmValue))
					return Finish(Vm::StopReason::ReadFault, instructionRip, "xchg-read",
						currentFaultAddress_);
				const uint64_t regValue = ReadRegister(modrm.regRaw, modrm.regIndex,
					bits, prefix.rexPresent);
				if (!WriteRm(modrm, prefix, bits, nextRip, regValue))
					return Finish(Vm::StopReason::WriteFault, instructionRip, "xchg-write",
						currentFaultAddress_);
				WriteRegister(modrm.regRaw, modrm.regIndex, bits, prefix.rexPresent, rmValue);
				mnemonic = "xchg";
				state_.rip = nextRip;
			}
			else if (opcode >= 0x88u && opcode <= 0x8Bu)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "mov-modrm");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				const bool byte = opcode == 0x88u || opcode == 0x8Au;
				const unsigned bits = byte ? 8u
					: (prefix.W() ? 64u : (prefix.operand16 ? 16u : 32u));
				const bool destinationIsRm = opcode == 0x88u || opcode == 0x89u;
				uint64_t value = 0;
				if (destinationIsRm)
				{
					value = ReadRegister(modrm.regRaw, modrm.regIndex, bits, prefix.rexPresent);
					if (!WriteRm(modrm, prefix, bits, nextRip, value))
						return Finish(Vm::StopReason::WriteFault, instructionRip, "mov-write",
							currentFaultAddress_);
				}
				else
				{
					if (!ReadRm(modrm, prefix, bits, nextRip, value))
						return Finish(Vm::StopReason::ReadFault, instructionRip, "mov-read",
							currentFaultAddress_);
					WriteRegister(modrm.regRaw, modrm.regIndex, bits, prefix.rexPresent, value);
				}
				mnemonic = "mov";
				state_.rip = nextRip;
			}
			else if (opcode == 0x8Du)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm) || modrm.rm.isRegister)
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "lea-modrm");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				const unsigned bits = prefix.W() ? 64u : (prefix.operand16 ? 16u : 32u);
				WriteRegister(modrm.regRaw, modrm.regIndex, bits, prefix.rexPresent,
					EffectiveAddress(modrm.rm, nextRip));
				mnemonic = "lea";
				state_.rip = nextRip;
			}
			else if (opcode == 0x8Fu)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm) || modrm.regRaw != 0u)
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "pop-rm");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				uint64_t value = 0;
				if (!Pop(value))
					return Finish(Vm::StopReason::ReadFault, instructionRip, "pop-read",
						currentFaultAddress_);
				if (!WriteRm(modrm, prefix, 64u, nextRip, value))
					return Finish(Vm::StopReason::WriteFault, instructionRip, "pop-write",
						currentFaultAddress_);
				mnemonic = "pop";
				state_.rip = nextRip;
			}
			else if (opcode == 0x90u)
			{
				currentLength_ = cursor.at;
				mnemonic = "nop";
				state_.rip = instructionRip + cursor.at;
			}
			else if (opcode == 0xA8u || opcode == 0xA9u)
			{
				/* TEST accumulator, immediate. The current EncMgr method uses
				 * test al,1 immediately before its conditional branch. */
				const unsigned bits = opcode == 0xA8u ? 8u
					: (prefix.W() ? 64u : (prefix.operand16 ? 16u : 32u));
				uint64_t immediate = 0;
				if (opcode == 0xA8u)
				{
					uint8_t byte = 0;
					if (!cursor.Read8(byte))
						return Finish(Vm::StopReason::InvalidInstruction, instructionRip,
							"test-acc-imm8");
					immediate = byte;
				}
				else if (bits == 64u)
				{
					uint32_t dword = 0;
					if (!cursor.Read32(dword))
						return Finish(Vm::StopReason::InvalidInstruction, instructionRip,
							"test-acc-imm32");
					immediate = dword;
				}
				else if (bits == 16u)
				{
					uint16_t word = 0;
					if (!cursor.Read16(word))
						return Finish(Vm::StopReason::InvalidInstruction, instructionRip,
							"test-acc-imm16");
					immediate = word;
				}
				else
				{
					uint32_t dword = 0;
					if (!cursor.Read32(dword))
						return Finish(Vm::StopReason::InvalidInstruction, instructionRip,
							"test-acc-imm32");
					immediate = dword;
				}
				const uint64_t value = ReadRegister(0u, 0u, bits, prefix.rexPresent);
				SetLogicFlags(value & immediate, bits);
				currentLength_ = cursor.at;
				mnemonic = "test";
				state_.rip = instructionRip + cursor.at;
			}
			else if (opcode == 0x04u || opcode == 0x05u
				|| opcode == 0x0Cu || opcode == 0x0Du
				|| opcode == 0x14u || opcode == 0x15u
				|| opcode == 0x1Cu || opcode == 0x1Du
				|| opcode == 0x24u || opcode == 0x25u
				|| opcode == 0x2Cu || opcode == 0x2Du
				|| opcode == 0x34u || opcode == 0x35u
				|| opcode == 0x3Cu || opcode == 0x3Du)
			{
				/* AL/AX/EAX/RAX, immediate (e.g. 05 = ADD EAX, imm32). */
				const bool byteForm = (opcode & 1u) == 0u;
				const unsigned bits = byteForm ? 8u
					: (prefix.W() ? 64u : (prefix.operand16 ? 16u : 32u));
				uint64_t immediate = 0;
				if (byteForm)
				{
					uint8_t value = 0;
					if (!cursor.Read8(value))
						return Finish(Vm::StopReason::InvalidInstruction,
							instructionRip, "acc-imm8");
					immediate = value;
				}
				else if (bits == 64u)
				{
					uint32_t value = 0;
					if (!cursor.Read32(value))
						return Finish(Vm::StopReason::InvalidInstruction,
							instructionRip, "acc-imm32");
					immediate = SignExtend(value, 32u);
				}
				else if (bits == 16u)
				{
					uint16_t value = 0;
					if (!cursor.Read16(value))
						return Finish(Vm::StopReason::InvalidInstruction,
							instructionRip, "acc-imm16");
					immediate = value;
				}
				else
				{
					uint32_t value = 0;
					if (!cursor.Read32(value))
						return Finish(Vm::StopReason::InvalidInstruction,
							instructionRip, "acc-imm32");
					immediate = value;
				}
				const uint64_t left = ReadRegister(0u, 0u, bits,
					prefix.rexPresent);
				const uint64_t mask = MaskFor(bits);
				uint64_t value = left;
				const unsigned operation = (opcode >> 3u) & 7u;
				switch (operation)
				{
				case 0u:
					value = (left + (immediate & mask)) & mask;
					SetAddFlags(left, immediate & mask, value, bits);
					mnemonic = "add";
					break;
				case 1u:
					value = (left | immediate) & mask;
					SetLogicFlags(value, bits);
					mnemonic = "or";
					break;
				case 2u:
				{
					const uint64_t carry = Flag(Vm::FlagCF) ? 1u : 0u;
					value = (left + (immediate & mask) + carry) & mask;
					SetAdcFlags(left, immediate & mask, carry, value, bits);
					mnemonic = "adc";
					break;
				}
				case 3u:
				{
					const uint64_t borrow = Flag(Vm::FlagCF) ? 1u : 0u;
					value = (left - (immediate & mask) - borrow) & mask;
					SetSbbFlags(left, immediate & mask, borrow, value, bits);
					mnemonic = "sbb";
					break;
				}
				case 4u:
					value = (left & immediate) & mask;
					SetLogicFlags(value, bits);
					mnemonic = "and";
					break;
				case 5u:
					value = (left - (immediate & mask)) & mask;
					SetSubFlags(left, immediate & mask, value, bits);
					mnemonic = "sub";
					break;
				case 6u:
					value = (left ^ immediate) & mask;
					SetLogicFlags(value, bits);
					mnemonic = "xor";
					break;
				default:
					value = (left - (immediate & mask)) & mask;
					SetSubFlags(left, immediate & mask, value, bits);
					mnemonic = "cmp";
					break;
				}
				if (operation != 7u)
					WriteRegister(0u, 0u, bits, prefix.rexPresent, value);
				currentLength_ = cursor.at;
				state_.rip = instructionRip + cursor.at;
			}
			else if (opcode >= 0xB0u && opcode <= 0xB7u)
			{
				uint8_t immediate = 0;
				if (!cursor.Read8(immediate))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "mov-imm8");
				const uint8_t raw = opcode - 0xB0u;
				const uint8_t index = raw | (prefix.B() ? 8u : 0u);
				WriteRegister(raw, index, 8u, prefix.rexPresent, immediate);
				currentLength_ = cursor.at;
				mnemonic = "mov";
				state_.rip = instructionRip + cursor.at;
			}
			else if (opcode >= 0xB8u && opcode <= 0xBFu)
			{
				const uint8_t raw = opcode - 0xB8u;
				const uint8_t index = raw | (prefix.B() ? 8u : 0u);
				const unsigned bits = prefix.W() ? 64u : (prefix.operand16 ? 16u : 32u);
				uint64_t immediate = 0;
				if (bits == 64u)
				{
					if (!cursor.Read64(immediate))
						return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "mov-imm64");
				}
				else if (bits == 16u)
				{
					uint16_t word = 0;
					if (!cursor.Read16(word))
						return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "mov-imm16");
					immediate = word;
				}
				else
				{
					uint32_t dword = 0;
					if (!cursor.Read32(dword))
						return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "mov-imm32");
					immediate = dword;
				}
				WriteRegister(raw, index, bits, prefix.rexPresent, immediate);
				currentLength_ = cursor.at;
				mnemonic = "mov";
				state_.rip = instructionRip + cursor.at;
			}
			else if (opcode == 0xC2u || opcode == 0xC3u)
			{
				uint16_t stackAdjust = 0;
				if (opcode == 0xC2u && !cursor.Read16(stackAdjust))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "ret-imm16");
				currentLength_ = cursor.at;
				mnemonic = "ret";
				if (callDepth_ == 0u && !options_.strictReturn)
				{
					state_.rip = instructionRip + cursor.at;
					++result_.instructionCount;
					Trace(instructionRip, state_.rip, currentBytes_, cursor.at, mnemonic);
					return Finish(Vm::StopReason::Completed, state_.rip, mnemonic);
				}
				uint64_t target = 0;
				if (!Pop(target))
					return Finish(Vm::StopReason::ReadFault, instructionRip, "ret-read",
						currentFaultAddress_);
				state_.gpr[Vm::RSP] += stackAdjust;
				if (callDepth_) --callDepth_;
				state_.rip = target;
				result_.branchTarget = target;
			}
			else if (opcode == 0xC6u || opcode == 0xC7u)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm) || modrm.regRaw != 0u)
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "mov-rm-imm");
				const unsigned bits = opcode == 0xC6u ? 8u
					: (prefix.W() ? 64u : (prefix.operand16 ? 16u : 32u));
				uint64_t immediate = 0;
				if (opcode == 0xC6u)
				{
					uint8_t byte = 0;
					if (!cursor.Read8(byte))
						return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "mov-rm-imm8");
					immediate = byte;
				}
				else if (bits == 16u)
				{
					uint16_t word = 0;
					if (!cursor.Read16(word))
						return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "mov-rm-imm16");
					immediate = word;
				}
				else
				{
					uint32_t dword = 0;
					if (!cursor.Read32(dword))
						return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "mov-rm-imm32");
					immediate = bits == 64u ? SignExtend(dword, 32u) : dword;
				}
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				if (!WriteRm(modrm, prefix, bits, nextRip, immediate))
					return Finish(Vm::StopReason::WriteFault, instructionRip, "mov-rm-write",
						currentFaultAddress_);
				mnemonic = "mov";
				state_.rip = nextRip;
			}
			else if (opcode == 0xC9u)
			{
				state_.gpr[Vm::RSP] = state_.gpr[Vm::RBP];
				uint64_t value = 0;
				if (!Pop(value))
					return Finish(Vm::StopReason::ReadFault, instructionRip, "leave-read",
						currentFaultAddress_);
				state_.gpr[Vm::RBP] = value;
				currentLength_ = cursor.at;
				mnemonic = "leave";
				state_.rip = instructionRip + cursor.at;
			}
			else if (opcode == 0xE8u || opcode == 0xE9u)
			{
				uint32_t rel = 0;
				if (!cursor.Read32(rel))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "rel32");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				const uint64_t target = nextRip + int32_t(rel);
				if (opcode == 0xE8u)
				{
					mnemonic = "call";
					ok = ExecuteCall(target, nextRip, mnemonic);
				}
				else
				{
					mnemonic = "jmp";
					ok = ExecuteJump(target, nextRip, mnemonic);
				}
			}
			else if (opcode == 0xEBu)
			{
				uint8_t rel = 0;
				if (!cursor.Read8(rel))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "jmp-rel8");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				const uint64_t target = nextRip + int8_t(rel);
				mnemonic = "jmp";
				ok = ExecuteJump(target, nextRip, mnemonic);
			}
			else if (opcode == 0xF6u || opcode == 0xF7u)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "group3-modrm");
				const unsigned bits = opcode == 0xF6u ? 8u
					: (prefix.W() ? 64u : (prefix.operand16 ? 16u : 32u));
				uint64_t immediate = 0;
				if (modrm.regRaw == 0u)
				{
					if (bits == 8u)
					{
						uint8_t byte = 0;
						if (!cursor.Read8(byte))
							return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "test-imm8");
						immediate = byte;
					}
					else if (bits == 16u)
					{
						uint16_t word = 0;
						if (!cursor.Read16(word))
							return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "test-imm16");
						immediate = word;
					}
					else
					{
						uint32_t dword = 0;
						if (!cursor.Read32(dword))
							return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "test-imm32");
						immediate = dword;
					}
				}
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				uint64_t value = 0;
				if (!ReadRm(modrm, prefix, bits, nextRip, value))
					return Finish(Vm::StopReason::ReadFault, instructionRip, "group3-read",
						currentFaultAddress_);
				const uint64_t mask = MaskFor(bits);
				if (modrm.regRaw == 0u)
				{
					SetLogicFlags(value & immediate, bits);
					mnemonic = "test";
				}
				else if (modrm.regRaw == 2u)
				{
					value = (~value) & mask;
					mnemonic = "not";
					if (!WriteRm(modrm, prefix, bits, nextRip, value))
						return Finish(Vm::StopReason::WriteFault, instructionRip, mnemonic,
							currentFaultAddress_);
				}
				else if (modrm.regRaw == 3u)
				{
					const uint64_t original = value;
					value = (0u - value) & mask;
					SetSubFlags(0, original, value, bits);
					mnemonic = "neg";
					if (!WriteRm(modrm, prefix, bits, nextRip, value))
						return Finish(Vm::StopReason::WriteFault, instructionRip, mnemonic,
							currentFaultAddress_);
				}
				else if (modrm.regRaw == 4u || modrm.regRaw == 5u)
				{
					/* MUL / IMUL r/m */
					const bool isSigned = modrm.regRaw == 5u;
					mnemonic = isSigned ? "imul" : "mul";
					if (bits == 8u)
					{
						const uint64_t left = state_.gpr[Vm::RAX] & 0xFFu;
						if (isSigned)
						{
							const int16_t product = int16_t(int8_t(left))
								* int16_t(int8_t(value));
							state_.gpr[Vm::RAX] = (state_.gpr[Vm::RAX] & ~0xFFFFull)
								| uint16_t(product);
							const bool wide = int16_t(int8_t(product)) != product;
							SetFlag(Vm::FlagCF, wide);
							SetFlag(Vm::FlagOF, wide);
						}
						else
						{
							const uint16_t product = uint16_t(uint8_t(left))
								* uint16_t(uint8_t(value));
							state_.gpr[Vm::RAX] = (state_.gpr[Vm::RAX] & ~0xFFFFull)
								| product;
							const bool wide = (product >> 8u) != 0u;
							SetFlag(Vm::FlagCF, wide);
							SetFlag(Vm::FlagOF, wide);
						}
					}
					else
					{
						const uint64_t left = state_.gpr[Vm::RAX] & mask;
						uint64_t high = 0;
						uint64_t low = 0;
						if (bits == 16u)
						{
							if (isSigned)
							{
								const int32_t product = int32_t(int16_t(left))
									* int32_t(int16_t(value));
								low = uint16_t(product);
								high = uint16_t(uint32_t(product) >> 16u);
								const bool wide = int32_t(int16_t(product)) != product;
								SetFlag(Vm::FlagCF, wide);
								SetFlag(Vm::FlagOF, wide);
							}
							else
							{
								const uint32_t product = uint32_t(uint16_t(left))
									* uint32_t(uint16_t(value));
								low = uint16_t(product);
								high = uint16_t(product >> 16u);
								const bool wide = high != 0u;
								SetFlag(Vm::FlagCF, wide);
								SetFlag(Vm::FlagOF, wide);
							}
							state_.gpr[Vm::RAX] = (state_.gpr[Vm::RAX] & ~0xFFFFull) | low;
							state_.gpr[Vm::RDX] = (state_.gpr[Vm::RDX] & ~0xFFFFull) | high;
						}
						else if (bits == 32u)
						{
							if (isSigned)
							{
								const int64_t product = int64_t(int32_t(left))
									* int64_t(int32_t(value));
								low = uint32_t(product);
								high = uint32_t(uint64_t(product) >> 32u);
								const bool wide = int64_t(int32_t(product)) != product;
								SetFlag(Vm::FlagCF, wide);
								SetFlag(Vm::FlagOF, wide);
							}
							else
							{
								const uint64_t product = uint64_t(uint32_t(left))
									* uint64_t(uint32_t(value));
								low = uint32_t(product);
								high = uint32_t(product >> 32u);
								const bool wide = high != 0u;
								SetFlag(Vm::FlagCF, wide);
								SetFlag(Vm::FlagOF, wide);
							}
							state_.gpr[Vm::RAX] = low;
							state_.gpr[Vm::RDX] = high;
						}
						else
						{
							if (isSigned)
							{
								const int64_t a = int64_t(left);
								const int64_t b = int64_t(value);
								const bool neg = (a < 0) != (b < 0);
								const uint64_t ua = a < 0 ? uint64_t(-a) : uint64_t(a);
								const uint64_t ub = b < 0 ? uint64_t(-b) : uint64_t(b);
								uint64_t hi = 0;
								uint64_t lo = 0;
								Mul64x64(ua, ub, &hi, &lo);
								if (neg && (lo | hi))
								{
									const uint64_t oldLo = lo;
									lo = uint64_t(0) - lo;
									hi = uint64_t(0) - hi - (oldLo != 0u ? 1u : 0u);
								}
								const bool wide = hi != (lo >> 63 ? ~0ull : 0ull);
								SetFlag(Vm::FlagCF, wide);
								SetFlag(Vm::FlagOF, wide);
								state_.gpr[Vm::RAX] = lo;
								state_.gpr[Vm::RDX] = hi;
							}
							else
							{
								uint64_t hi = 0;
								uint64_t lo = 0;
								Mul64x64(left, value, &hi, &lo);
								const bool wide = hi != 0u;
								SetFlag(Vm::FlagCF, wide);
								SetFlag(Vm::FlagOF, wide);
								state_.gpr[Vm::RAX] = lo;
								state_.gpr[Vm::RDX] = hi;
							}
						}
					}
				}
				else if (modrm.regRaw == 6u || modrm.regRaw == 7u)
				{
					/* DIV / IDIV r/m (e.g. `49 F7 F1` = div r9). */
					const bool isSigned = modrm.regRaw == 7u;
					mnemonic = isSigned ? "idiv" : "div";
					if (value == 0u)
						return Finish(Vm::StopReason::InvalidInstruction, instructionRip,
							"div-zero");
					if (bits == 8u)
					{
						const uint16_t dividend = uint16_t(state_.gpr[Vm::RAX] & 0xFFFFu);
						if (isSigned)
						{
							const int16_t num = int16_t(dividend);
							const int8_t den = int8_t(value);
							const int16_t quot = num / den;
							const int16_t rem = num % den;
							if (quot != int16_t(int8_t(quot)))
								return Finish(Vm::StopReason::InvalidInstruction,
									instructionRip, "idiv-overflow");
							state_.gpr[Vm::RAX] = (state_.gpr[Vm::RAX] & ~0xFFFFull)
								| (uint16_t(uint8_t(rem)) << 8u)
								| uint8_t(quot);
						}
						else
						{
							const uint16_t quot = dividend / uint8_t(value);
							const uint16_t rem = dividend % uint8_t(value);
							if (quot > 0xFFu)
								return Finish(Vm::StopReason::InvalidInstruction,
									instructionRip, "div-overflow");
							state_.gpr[Vm::RAX] = (state_.gpr[Vm::RAX] & ~0xFFFFull)
								| (uint16_t(uint8_t(rem)) << 8u)
								| uint8_t(quot);
						}
					}
					else if (bits == 16u)
					{
						const uint32_t dividend =
							(uint32_t(state_.gpr[Vm::RDX] & 0xFFFFu) << 16u)
							| uint32_t(state_.gpr[Vm::RAX] & 0xFFFFu);
						if (isSigned)
						{
							const int32_t num = int32_t(dividend);
							const int16_t den = int16_t(value);
							const int32_t quot = num / den;
							const int32_t rem = num % den;
							if (quot != int32_t(int16_t(quot)))
								return Finish(Vm::StopReason::InvalidInstruction,
									instructionRip, "idiv-overflow");
							state_.gpr[Vm::RAX] = (state_.gpr[Vm::RAX] & ~0xFFFFull)
								| uint16_t(quot);
							state_.gpr[Vm::RDX] = (state_.gpr[Vm::RDX] & ~0xFFFFull)
								| uint16_t(rem);
						}
						else
						{
							const uint32_t quot = dividend / uint16_t(value);
							const uint32_t rem = dividend % uint16_t(value);
							if (quot > 0xFFFFu)
								return Finish(Vm::StopReason::InvalidInstruction,
									instructionRip, "div-overflow");
							state_.gpr[Vm::RAX] = (state_.gpr[Vm::RAX] & ~0xFFFFull)
								| uint16_t(quot);
							state_.gpr[Vm::RDX] = (state_.gpr[Vm::RDX] & ~0xFFFFull)
								| uint16_t(rem);
						}
					}
					else if (bits == 32u)
					{
						const uint64_t dividend =
							(uint64_t(uint32_t(state_.gpr[Vm::RDX])) << 32u)
							| uint32_t(state_.gpr[Vm::RAX]);
						if (isSigned)
						{
							const int64_t num = int64_t(dividend);
							const int32_t den = int32_t(value);
							const int64_t quot = num / den;
							const int64_t rem = num % den;
							if (quot != int64_t(int32_t(quot)))
								return Finish(Vm::StopReason::InvalidInstruction,
									instructionRip, "idiv-overflow");
							state_.gpr[Vm::RAX] = uint32_t(quot);
							state_.gpr[Vm::RDX] = uint32_t(rem);
						}
						else
						{
							const uint64_t quot = dividend / uint32_t(value);
							const uint64_t rem = dividend % uint32_t(value);
							if (quot > 0xFFFFFFFFull)
								return Finish(Vm::StopReason::InvalidInstruction,
									instructionRip, "div-overflow");
							state_.gpr[Vm::RAX] = uint32_t(quot);
							state_.gpr[Vm::RDX] = uint32_t(rem);
						}
					}
					else
					{
						const uint64_t hi = state_.gpr[Vm::RDX];
						const uint64_t lo = state_.gpr[Vm::RAX];
						uint64_t quot = 0;
						uint64_t rem = 0;
						if (isSigned)
						{
							const bool numNeg = int64_t(hi) < 0;
							const bool denNeg = int64_t(value) < 0;
							uint64_t absHi = hi;
							uint64_t absLo = lo;
							if (numNeg)
							{
								const uint64_t oldLo = absLo;
								absLo = uint64_t(0) - absLo;
								absHi = uint64_t(0) - absHi - (oldLo != 0u ? 1u : 0u);
							}
							const uint64_t absDen = denNeg
								? uint64_t(-int64_t(value)) : value;
							if (!Div128By64(absHi, absLo, absDen, &quot, &rem))
								return Finish(Vm::StopReason::InvalidInstruction,
									instructionRip, "idiv-overflow");
							if ((numNeg != denNeg) && quot)
								quot = uint64_t(0) - quot;
							if (numNeg && rem)
								rem = uint64_t(0) - rem;
						}
						else if (!Div128By64(hi, lo, value, &quot, &rem))
							return Finish(Vm::StopReason::InvalidInstruction,
								instructionRip, "div-overflow");
						state_.gpr[Vm::RAX] = quot;
						state_.gpr[Vm::RDX] = rem;
					}
				}
				else
					return Finish(Vm::StopReason::UnsupportedInstruction, instructionRip,
						"group3-unsupported");
				state_.rip = nextRip;
			}
			else if (opcode == 0xFFu)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "ff-modrm");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				const bool incDec = modrm.regRaw == 0u || modrm.regRaw == 1u;
				const unsigned operandBits = incDec
					? (prefix.W() ? 64u : (prefix.operand16 ? 16u : 32u))
					: 64u;
				uint64_t value = 0;
				if (!ReadRm(modrm, prefix, operandBits, nextRip, value))
					return Finish(Vm::StopReason::ReadFault, instructionRip, "ff-read",
						currentFaultAddress_);
				if (modrm.regRaw == 2u)
				{
					mnemonic = "call";
					ok = ExecuteCall(value, nextRip, mnemonic);
				}
				else if (modrm.regRaw == 4u)
				{
					mnemonic = "jmp";
					ok = ExecuteJump(value, nextRip, mnemonic);
				}
				else if (modrm.regRaw == 6u)
				{
					mnemonic = "push";
					if (!Push(value))
						return Finish(Vm::StopReason::WriteFault, instructionRip, mnemonic,
							currentFaultAddress_);
					state_.rip = nextRip;
				}
				else if (incDec)
				{
					const uint64_t oldCf = state_.rflags & Vm::FlagCF;
					const uint64_t old = value;
					const uint64_t operandMask = MaskFor(operandBits);
					value = (modrm.regRaw == 0u ? value + 1u : value - 1u)
						& operandMask;
					if (modrm.regRaw == 0u)
						SetAddFlags(old, 1u, value, operandBits);
					else
						SetSubFlags(old, 1u, value, operandBits);
					SetFlag(Vm::FlagCF, oldCf != 0);
					if (!WriteRm(modrm, prefix, operandBits, nextRip, value))
						return Finish(Vm::StopReason::WriteFault, instructionRip, "incdec-write",
							currentFaultAddress_);
					mnemonic = modrm.regRaw == 0u ? "inc" : "dec";
					state_.rip = nextRip;
				}
				else
					return Finish(Vm::StopReason::UnsupportedInstruction, instructionRip,
						"ff-unsupported");
			}
			else if (opcode == 0x63u)
			{
				ModRm modrm{};
				if (!DecodeModRm(cursor, prefix, modrm))
					return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "movsxd-modrm");
				currentLength_ = cursor.at;
				const uint64_t nextRip = instructionRip + cursor.at;
				uint64_t value = 0;
				if (!ReadRm(modrm, prefix, 32u, nextRip, value))
					return Finish(Vm::StopReason::ReadFault, instructionRip, "movsxd-read",
						currentFaultAddress_);
				WriteRegister(modrm.regRaw, modrm.regIndex, prefix.W() ? 64u : 32u,
					prefix.rexPresent, SignExtend(value, 32u));
				mnemonic = "movsxd";
				state_.rip = nextRip;
			}
			else
			{
				currentLength_ = cursor.at;
				return Finish(Vm::StopReason::UnsupportedInstruction, instructionRip,
					"unsupported");
			}

			if (!ok)
				return false;
			++result_.instructionCount;
			Trace(instructionRip, state_.rip, currentBytes_, cursor.at, mnemonic);
			return true;
		}

		struct TestRange
		{
			uint64_t base{};
			uint8_t* data{};
			size_t size{};
			bool writable{};
		};

		struct TestMemory
		{
			TestRange ranges[4]{};
			size_t count{};
		};

		static bool TestRangeContains(const TestRange& range, uint64_t address, size_t size)
		{
			if (address < range.base || size > range.size)
				return false;
			return address - range.base <= range.size - size;
		}

		static bool TestRead(void* context, uint64_t address, void* dst, size_t size)
		{
			TestMemory* memory = static_cast<TestMemory*>(context);
			if (!memory || !dst || !size)
				return false;
			uint8_t* out = static_cast<uint8_t*>(dst);
			for (size_t i = 0; i < memory->count; ++i)
			{
				const TestRange& range = memory->ranges[i];
				if (!TestRangeContains(range, address, size))
					continue;
				memcpy(out, range.data + size_t(address - range.base), size);
				return true;
			}
			/* Cross-page stitch (same contract as CurrentCoordRuntime::ReadCode). */
			for (size_t got = 0; got < size; ++got)
			{
				bool hit = false;
				for (size_t i = 0; i < memory->count; ++i)
				{
					const TestRange& range = memory->ranges[i];
					if (!TestRangeContains(range, address + got, 1u))
						continue;
					out[got] = range.data[size_t(address + got - range.base)];
					hit = true;
					break;
				}
				if (!hit)
					return false;
			}
			return true;
		}

		static bool TestWrite(void* context, uint64_t address, const void* src, size_t size)
		{
			TestMemory* memory = static_cast<TestMemory*>(context);
			if (!memory || !src || !size)
				return false;
			for (size_t i = 0; i < memory->count; ++i)
			{
				TestRange& range = memory->ranges[i];
				if (!range.writable || !TestRangeContains(range, address, size))
					continue;
				memcpy(range.data + size_t(address - range.base), src, size);
				return true;
			}
			return false;
		}

		static bool RunCode(const uint8_t* source, size_t sourceSize, Vm::State& state,
			Vm::Result& result, uint8_t* data = nullptr, size_t dataSize = 0,
			uint64_t dataBase = 0, uint32_t instructionLimit = 256u,
			Vm::SyscallHookFn syscallHook = nullptr,
			void* syscallHookContext = nullptr)
		{
			if (!source || !sourceSize || sourceSize > 512u)
				return false;
			uint8_t code[512]{};
			uint8_t stack[1024]{};
			memcpy(code, source, sourceSize);
			constexpr uint64_t codeBase = 0x10000000ull;
			constexpr uint64_t stackBase = 0x20000000ull;
			TestMemory memory{};
			memory.ranges[memory.count++] = { codeBase, code, sizeof(code), false };
			memory.ranges[memory.count++] = { stackBase, stack, sizeof(stack), true };
			if (data && dataSize)
				memory.ranges[memory.count++] = { dataBase, data, dataSize, true };
			state.rip = codeBase;
			state.gpr[Vm::RSP] = stackBase + sizeof(stack) - 0x20u;
			Vm::Options options{};
			options.read = TestRead;
			options.write = TestWrite;
			options.memoryContext = &memory;
			options.syscallHook = syscallHook;
			options.syscallHookContext = syscallHookContext;
			options.instructionLimit = instructionLimit;
			result = Vm::Run(&state, &options);
			return true;
		}
	}

	MiniX64Vm::Result MiniX64Vm::Run(State* state, const Options* options)
	{
		if (!state || !options)
		{
			Result result{};
			result.reason = StopReason::InvalidInstruction;
			result.mnemonic = "bad-options";
			return result;
		}
		Runner runner(*state, *options);
		return runner.Run();
	}

	const char* MiniX64Vm::StopReasonName(StopReason reason)
	{
		switch (reason)
		{
		case StopReason::Completed: return "completed";
		case StopReason::StopRip: return "stop-rip";
		case StopReason::InstructionLimit: return "instruction-limit";
		case StopReason::ReadFault: return "read-fault";
		case StopReason::WriteFault: return "write-fault";
		case StopReason::UnsupportedInstruction: return "unsupported-instruction";
		case StopReason::InvalidInstruction: return "invalid-instruction";
		case StopReason::CallDepthLimit: return "call-depth-limit";
		case StopReason::HookStop: return "hook-stop";
		case StopReason::HookFault: return "hook-fault";
		default: return "unknown";
		}
	}

	bool MiniX64Vm::RunSelfTests(char* detail, size_t detailCapacity)
	{
		auto fail = [&](const char* name, const Result& result) {
			if (detail && detailCapacity)
				std::snprintf(detail, detailCapacity,
					"fail:%s reason=%s rip=0x%llX ins=%u", name,
					StopReasonName(result.reason),
					static_cast<unsigned long long>(result.rip),
					static_cast<unsigned>(result.instructionCount));
			return false;
		};

		{
			const uint8_t code[] = {
				0x48, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
				0xB8, 0x78, 0x56, 0x34, 0x12, 0xC3
			};
			State state{};
			Result result{};
			RunCode(code, sizeof(code), state, result);
			if (result.reason != StopReason::Completed
				|| state.gpr[RAX] != 0x12345678ull)
				return fail("eax-zero-extend", result);
		}

		{
			const uint8_t code[] = {
				0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
				0x48, 0xC1, 0xC0, 0x09, 0xC3
			};
			State state{};
			Result result{};
			RunCode(code, sizeof(code), state, result);
			if (result.reason != StopReason::Completed
				|| state.gpr[RAX] != 0x446688AACCEF1022ull)
				return fail("rol64", result);
		}

		{
			const uint8_t code[] = { 0xC1, 0xF8, 0x01, 0xC3 };
			State state{};
			state.gpr[RAX] = 0xAAAAAAAA80000001ull;
			state.rflags = 2u | FlagOF;
			Result result{};
			RunCode(code, sizeof(code), state, result);
			const uint64_t required = FlagCF | FlagPF | FlagSF;
			const uint64_t forbidden = FlagZF | FlagOF;
			if (result.reason != StopReason::Completed
				|| state.gpr[RAX] != 0x00000000C0000000ull
				|| (state.rflags & required) != required
				|| (state.rflags & forbidden) != 0u)
				return fail("sar32-sign-fill", result);
		}

		{
			const uint8_t code[] = {
				0x48, 0x8B, 0x05, 0x01, 0x00, 0x00, 0x00, 0xC3,
				0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11
			};
			State state{};
			Result result{};
			RunCode(code, sizeof(code), state, result);
			if (result.reason != StopReason::Completed
				|| state.gpr[RAX] != 0x1122334455667788ull)
				return fail("rip-relative", result);
		}

		{
			const uint8_t code[] = {
				0xB8, 0x00, 0x00, 0x00, 0x00,
				0xE8, 0x01, 0x00, 0x00, 0x00,
				0xC3,
				0xB8, 0x2A, 0x00, 0x00, 0x00, 0xC3
			};
			State state{};
			Result result{};
			RunCode(code, sizeof(code), state, result);
			if (result.reason != StopReason::Completed || state.gpr[RAX] != 42u)
				return fail("call-ret", result);
		}

		{
			const uint8_t code[] = { 0xFF, 0xC0, 0xC3 };
			State state{};
			state.gpr[RAX] = 0x11223344FFFFFFFFull;
			state.rflags = 2u | FlagCF | FlagOF;
			Result result{};
			RunCode(code, sizeof(code), state, result);
			const uint64_t required = FlagCF | FlagPF | FlagAF | FlagZF;
			const uint64_t forbidden = FlagSF | FlagOF;
			if (result.reason != StopReason::Completed || state.gpr[RAX] != 0u
				|| (state.rflags & required) != required
				|| (state.rflags & forbidden) != 0u)
				return fail("ff-inc32-width", result);
		}

		{
			const uint8_t code[] = { 0x66, 0xFF, 0xC8, 0xC3 };
			State state{};
			state.gpr[RAX] = 0x1122334455660000ull;
			state.rflags = 2u | FlagCF | FlagOF;
			Result result{};
			RunCode(code, sizeof(code), state, result);
			const uint64_t required = FlagCF | FlagPF | FlagAF | FlagSF;
			const uint64_t forbidden = FlagZF | FlagOF;
			if (result.reason != StopReason::Completed
				|| state.gpr[RAX] != 0x112233445566FFFFull
				|| (state.rflags & required) != required
				|| (state.rflags & forbidden) != 0u)
				return fail("ff-dec16-width", result);
		}

		{
			const uint8_t code[] = { 0xFF, 0x00, 0xC3 };
			constexpr uint64_t dataBase = 0x30000000ull;
			uint32_t data = 0xFFFFFFFFu;
			State state{};
			state.gpr[RAX] = dataBase;
			Result result{};
			RunCode(code, sizeof(code), state, result,
				reinterpret_cast<uint8_t*>(&data), sizeof(data), dataBase);
			if (result.reason != StopReason::Completed || data != 0u)
				return fail("ff-inc32-memory-width", result);
		}

		{
			const uint8_t code[] = { 0x48, 0x11, 0xC8, 0xC3 };
			State state{};
			state.gpr[RAX] = 0u;
			state.gpr[RCX] = ~0ull;
			state.rflags = 2u | FlagCF | FlagOF | FlagSF;
			Result result{};
			RunCode(code, sizeof(code), state, result);
			const uint64_t required = FlagCF | FlagPF | FlagAF | FlagZF;
			const uint64_t forbidden = FlagSF | FlagOF;
			if (result.reason != StopReason::Completed || state.gpr[RAX] != 0u
				|| (state.rflags & required) != required
				|| (state.rflags & forbidden) != 0u)
				return fail("adc64-carry-wrap", result);
		}

		{
			const uint8_t code[] = { 0x48, 0x11, 0xC8, 0xC3 };
			State state{};
			state.gpr[RAX] = 0x7FFFFFFFFFFFFFFFull;
			state.gpr[RCX] = 0x7FFFFFFFFFFFFFFFull;
			state.rflags = 2u | FlagCF;
			Result result{};
			RunCode(code, sizeof(code), state, result);
			const uint64_t required = FlagAF | FlagSF | FlagOF;
			const uint64_t forbidden = FlagCF | FlagZF;
			if (result.reason != StopReason::Completed || state.gpr[RAX] != ~0ull
				|| (state.rflags & required) != required
				|| (state.rflags & forbidden) != 0u)
				return fail("adc64-overflow", result);
		}

		{
			const uint8_t code[] = { 0x83, 0xD0, 0xFF, 0xC3 };
			State state{};
			state.gpr[RAX] = 0xAAAAAAAA00000000ull;
			state.rflags = 2u | FlagCF;
			Result result{};
			RunCode(code, sizeof(code), state, result);
			if (result.reason != StopReason::Completed || state.gpr[RAX] != 0u
				|| (state.rflags & FlagCF) == 0u)
				return fail("adc32-imm-carry-wrap", result);
		}

		{
			const uint8_t code[] = { 0x48, 0x19, 0xC8, 0xC3 };
			State state{};
			state.gpr[RAX] = 0u;
			state.gpr[RCX] = ~0ull;
			state.rflags = 2u | FlagCF | FlagOF | FlagSF;
			Result result{};
			RunCode(code, sizeof(code), state, result);
			const uint64_t required = FlagCF | FlagPF | FlagAF | FlagZF;
			const uint64_t forbidden = FlagSF | FlagOF;
			if (result.reason != StopReason::Completed || state.gpr[RAX] != 0u
				|| (state.rflags & required) != required
				|| (state.rflags & forbidden) != 0u)
				return fail("sbb64-borrow-wrap", result);
		}

		{
			const uint8_t code[] = { 0x48, 0x19, 0xC8, 0xC3 };
			State state{};
			state.gpr[RAX] = 0x8000000000000000ull;
			state.gpr[RCX] = 0x7FFFFFFFFFFFFFFFull;
			state.rflags = 2u | FlagCF;
			Result result{};
			RunCode(code, sizeof(code), state, result);
			const uint64_t required = FlagPF | FlagAF | FlagZF | FlagOF;
			const uint64_t forbidden = FlagCF | FlagSF;
			if (result.reason != StopReason::Completed || state.gpr[RAX] != 0u
				|| (state.rflags & required) != required
				|| (state.rflags & forbidden) != 0u)
				return fail("sbb64-overflow", result);
		}

		{
			const uint8_t code[] = { 0x83, 0xD8, 0xFF, 0xC3 };
			State state{};
			state.gpr[RAX] = 0xAAAAAAAA00000000ull;
			state.rflags = 2u | FlagCF;
			Result result{};
			RunCode(code, sizeof(code), state, result);
			if (result.reason != StopReason::Completed || state.gpr[RAX] != 0u
				|| (state.rflags & FlagCF) == 0u)
				return fail("sbb32-imm-borrow-wrap", result);
		}

		{
			const uint8_t code[] = { 0xA8, 0x01, 0xC3 };
			State state{};
			state.gpr[RAX] = 0x100u;
			Result result{};
			RunCode(code, sizeof(code), state, result);
			if (result.reason != StopReason::Completed
				|| (state.rflags & (FlagZF | FlagSF | FlagCF | FlagOF)) != FlagZF)
				return fail("test-acc-imm8", result);
		}

		{
			/* This is the exact bit-test form reached by the 37117 Native slot. */
			const uint8_t code[] = {
				0x66, 0x0F, 0xBA, 0xE3, 0x0C, /* bt bx, 12 */
				0x66, 0x0F, 0xBA, 0xEB, 0x0D, /* bts bx, 13 */
				0x66, 0x0F, 0xBA, 0xF3, 0x0D, /* btr bx, 13 */
				0x66, 0x0F, 0xBA, 0xFB, 0x0C, /* btc bx, 12 */
				0xC3
			};
			State state{};
			state.gpr[RBX] = 0x1122334455661000ull;
			Result result{};
			RunCode(code, sizeof(code), state, result);
			if (result.reason != StopReason::Completed
				|| state.gpr[RBX] != 0x1122334455660000ull
				|| (state.rflags & FlagCF) == 0u)
				return fail("bt-group-register", result);
		}

		{
			/* Register-index BT/BTS (`48 0F A3 C8`, `48 0F AB C8`). */
			const uint8_t code[] = {
				0x48, 0x0F, 0xA3, 0xC8, /* bt rax, rcx */
				0x48, 0x0F, 0xAB, 0xC8, /* bts rax, rcx */
				0xC3
			};
			State state{};
			state.gpr[RAX] = 0x10ull;
			state.gpr[RCX] = 4ull;
			Result result{};
			RunCode(code, sizeof(code), state, result);
			if (result.reason != StopReason::Completed
				|| state.gpr[RAX] != 0x10ull
				|| (state.rflags & FlagCF) == 0u)
				return fail("bt-reg-form", result);
		}

		{
			/* Native post-TLS path uses movhlps / shufps / andps. */
			const uint8_t code[] = {
				0x0F, 0x12, 0xCA,             /* movhlps xmm1, xmm2 */
				0x0F, 0xC6, 0xC1, 0xC4,       /* shufps xmm0, xmm1, 0xC4 */
				0x0F, 0x54, 0xC1,             /* andps xmm0, xmm1 */
				0xC3
			};
			State state{};
			const uint32_t xmm0In[4] = { 1u, 2u, 3u, 4u };
			const uint32_t xmm1In[4] = { 5u, 6u, 7u, 8u };
			const uint32_t xmm2In[4] = { 9u, 10u, 11u, 12u };
			memcpy(state.ymm[0], xmm0In, sizeof(xmm0In));
			memcpy(state.ymm[1], xmm1In, sizeof(xmm1In));
			memcpy(state.ymm[2], xmm2In, sizeof(xmm2In));
			Result result{};
			RunCode(code, sizeof(code), state, result);
			uint32_t xmm0Out[4]{};
			uint32_t xmm1Out[4]{};
			memcpy(xmm0Out, state.ymm[0], sizeof(xmm0Out));
			memcpy(xmm1Out, state.ymm[1], sizeof(xmm1Out));
			/* movhlps xmm1,xmm2 -> xmm1 = {11,12,7,8} then
			 * shufps xmm0,xmm1,0xC4 -> {1,2,11,8}
			 * andps xmm0,xmm1 -> {1,2,11,8} & {11,12,7,8} = {1,0,3,8} */
			if (result.reason != StopReason::Completed
				|| xmm1Out[0] != 11u || xmm1Out[1] != 12u
				|| xmm1Out[2] != 7u || xmm1Out[3] != 8u
				|| xmm0Out[0] != 1u || xmm0Out[1] != 0u
				|| xmm0Out[2] != 3u || xmm0Out[3] != 8u)
				return fail("sse-movhlps-shufps-andps", result);
		}

		{
			/* lock xadd qword ptr [r8], r9 */
			const uint8_t code[] = {
				0xF0, 0x4D, 0x0F, 0xC1, 0x08, /* lock xadd [r8], r9 */
				0xC3
			};
			uint8_t memory[16]{};
			uint64_t initial = 7ull;
			memcpy(memory, &initial, sizeof(initial));
			State state{};
			state.gpr[R8] = 0x1000ull;
			state.gpr[R9] = 1ull;
			Result result{};
			RunCode(code, sizeof(code), state, result,
				memory, sizeof(memory), 0x1000ull);
			uint64_t stored = 0;
			memcpy(&stored, memory, sizeof(stored));
			if (result.reason != StopReason::Completed
				|| stored != 8ull || state.gpr[R9] != 7ull)
				return fail("lock-xadd64", result);
		}

		{
			/* 223108 stop: lock cmpxchg qword ptr [rsi], rcx */
			const uint8_t code[] = {
				0xF0, 0x48, 0x0F, 0xB1, 0x0E, /* lock cmpxchg [rsi], rcx */
				0xC3
			};
			uint8_t memory[16]{};
			uint64_t initial = 0ull;
			memcpy(memory, &initial, sizeof(initial));
			State state{};
			state.gpr[RSI] = 0x1000ull;
			state.gpr[RAX] = 0ull;
			state.gpr[RCX] = 0x11ull;
			Result result{};
			RunCode(code, sizeof(code), state, result,
				memory, sizeof(memory), 0x1000ull);
			uint64_t stored = 0;
			memcpy(&stored, memory, sizeof(stored));
			if (result.reason != StopReason::Completed
				|| stored != 0x11ull
				|| (state.rflags & FlagZF) == 0u
				|| state.gpr[RAX] != 0ull)
				return fail("lock-cmpxchg64-match", result);
		}

		{
			/* 49 F7 F1 = div r9 */
			const uint8_t code[] = {
				0x49, 0xF7, 0xF1, /* div r9 */
				0xC3
			};
			State state{};
			state.gpr[RDX] = 0ull;
			state.gpr[RAX] = 100ull;
			state.gpr[R9] = 7ull;
			Result result{};
			RunCode(code, sizeof(code), state, result);
			if (result.reason != StopReason::Completed
				|| state.gpr[RAX] != 14ull || state.gpr[RDX] != 2ull)
				return fail("div64-r9", result);
		}

		{
			const uint8_t code[] = { 0xEB, 0xFE };
			State state{};
			Result result{};
			RunCode(code, sizeof(code), state, result, nullptr, 0, 0, 8u);
			if (result.reason != StopReason::InstructionLimit
				|| result.instructionCount != 8u)
				return fail("instruction-limit", result);
		}

		{
			const uint8_t code[] = { 0x0F, 0x0B };
			State state{};
			Result result{};
			RunCode(code, sizeof(code), state, result);
			if (result.reason != StopReason::UnsupportedInstruction
				|| result.length < 2u
				|| result.bytes[0] != 0x0Fu || result.bytes[1] != 0x0Bu)
				return fail("unsupported", result);
		}

		{
			const uint8_t scalar[] = {
				0x49, 0xB9, 0xA0, 0xF4, 0xA0, 0x64, 0xE3, 0xB7, 0x61, 0x6D,
				0x4D, 0x31, 0xC8, 0x49, 0xC1, 0xC0, 0x09,
				0x49, 0xBB, 0x4A, 0xE6, 0x8B, 0xB4, 0x40, 0x35, 0xC0, 0x92,
				0x4D, 0x01, 0xD8, 0x4D, 0x89, 0xC2, 0x49, 0xC1, 0xEA, 0x20,
				0x41, 0x81, 0xF2, 0x04, 0xE3, 0xA2, 0x0D,
				0xBE, 0x1C, 0x96, 0xAA, 0xE7, 0x44, 0x0F, 0xAF, 0xD6,
				0x4D, 0x31, 0xC2,
				0x49, 0xB8, 0x82, 0x9D, 0xE5, 0x9E, 0x27, 0x81, 0xDD, 0x8C,
				0x4D, 0x01, 0xC2, 0x49, 0xC1, 0xC2, 0x24,
				0x48, 0xBF, 0xAC, 0xAC, 0xD6, 0xE1, 0x3C, 0xA7, 0x20, 0x1E,
				0x49, 0x31, 0xFA, 0x4C, 0x31, 0xC9, 0x48, 0xC1, 0xC1, 0x09,
				0x4C, 0x01, 0xD9, 0x49, 0x89, 0xC9, 0x49, 0xC1, 0xE9, 0x20,
				0x41, 0x81, 0xF1, 0xE2, 0xC6, 0x2E, 0x3E,
				0x44, 0x0F, 0xAF, 0xCE, 0x49, 0x31, 0xC9, 0x4D, 0x01, 0xC1,
				0x49, 0xC1, 0xC1, 0x24, 0x49, 0x31, 0xF9,
				0x4C, 0x89, 0x10, 0x4C, 0x89, 0x48, 0x08, 0xC3
			};
			constexpr uint64_t outputBase = 0x30000000ull;
			uint8_t output[16]{};
			State state{};
			state.gpr[RAX] = outputBase;
			state.gpr[R8] = 0xB96F91770E96B68Cull;
			state.gpr[RCX] = 0x9F4F91770E9B7454ull;
			Result result{};
			RunCode(scalar, sizeof(scalar), state, result,
				output, sizeof(output), outputBase);
			uint64_t qword[2]{};
			memcpy(qword, output, sizeof(qword));
			if (result.reason != StopReason::Completed
				|| qword[0] != 0x9A233C3F5F7B5F7Eull
				|| qword[1] != 0x80E48DFB5F7B5F67ull)
				return fail("op23-scalar", result);
		}

		{
			const uint8_t scalar[] = {
				0x49, 0xB8, 0xB8, 0x45, 0x65, 0x83, 0x85, 0x1B, 0xA3, 0x06,
				0x4D, 0x31, 0xC2, 0x49, 0xC1, 0xC2, 0x3C,
				0x49, 0xB9, 0xA7, 0x6B, 0x7B, 0x06, 0x73, 0x91, 0x72, 0x79,
				0x4D, 0x01, 0xCA,
				0x49, 0x81, 0xF2, 0x67, 0xA8, 0xED, 0x42,
				0x49, 0xBB, 0x0B, 0x1F, 0x41, 0x17, 0x1C, 0x5F, 0x72, 0x65,
				0x4D, 0x01, 0xDA, 0x49, 0xC1, 0xC2, 0x1A,
				0x48, 0xBE, 0xE4, 0x0E, 0xA2, 0xB6, 0x16, 0xE0, 0x13, 0xFA,
				0x49, 0x31, 0xF2,
				0x4C, 0x31, 0xC1, 0x48, 0xC1, 0xC1, 0x3C,
				0x4C, 0x01, 0xC9,
				0x48, 0x81, 0xF1, 0x67, 0xA8, 0xED, 0x42,
				0x4C, 0x01, 0xD9, 0x48, 0xC1, 0xC1, 0x1A,
				0x48, 0x31, 0xF1,
				0x4C, 0x89, 0x10, 0x48, 0x89, 0x48, 0x08, 0xC3
			};
			constexpr uint64_t outputBase = 0x30000000ull;
			uint8_t output[16]{};
			State state{};
			state.gpr[RAX] = outputBase;
			state.gpr[R10] = 0xB96F91770E96B68Cull;
			state.gpr[RCX] = 0x9F4F91770E9B7454ull;
			Result result{};
			RunCode(scalar, sizeof(scalar), state, result,
				output, sizeof(output), outputBase);
			uint64_t qword[2]{};
			memcpy(qword, output, sizeof(qword));
			if (result.reason != StopReason::Completed
				|| qword[0] != 0x0000009196098800ull
				|| qword[1] != 0u)
				return fail("op23-current-scalar", result);
		}

		{
			const uint8_t code[] = {
				0x66, 0x48, 0x0F, 0x6E, 0xC0,
				0x66, 0x48, 0x0F, 0x6E, 0xC9,
				0x66, 0x0F, 0x6C, 0xC1,
				0xC3
			};
			State state{};
			state.gpr[RAX] = 0x1122334455667788ull;
			state.gpr[RCX] = 0x99AABBCCDDEEFF00ull;
			Result result{};
			RunCode(code, sizeof(code), state, result);
			uint64_t qword[2]{};
			memcpy(qword, state.ymm[0], sizeof(qword));
			if (result.reason != StopReason::Completed
				|| qword[0] != state.gpr[RAX] || qword[1] != state.gpr[RCX])
				return fail("punpcklqdq", result);
		}

		{
			const uint8_t code[] = { 0x66, 0x0F, 0x6D, 0xC1, 0xC3 };
			State state{};
			const uint64_t dst[2]{ 0x1111222233334444ull, 0x5555666677778888ull };
			const uint64_t src[2]{ 0x9999AAAABBBBCCCCull, 0xDDDDEEEEFFFF0000ull };
			memcpy(state.ymm[0], dst, sizeof(dst));
			memcpy(state.ymm[1], src, sizeof(src));
			Result result{};
			RunCode(code, sizeof(code), state, result);
			uint64_t qword[2]{};
			memcpy(qword, state.ymm[0], sizeof(qword));
			if (result.reason != StopReason::Completed
				|| qword[0] != dst[1] || qword[1] != src[1])
				return fail("punpckhqdq", result);
		}

		{
			/* VEX subset: vmovaps / vxorps / vpxor / vblendps(+ymm15) /
			 * vinsertf128 / vaesdeclast. */
			const uint8_t code[] = {
				0xC5, 0x78, 0x28, 0xD0,             /* vmovaps xmm10, xmm0 */
				0xC5, 0xF0, 0x57, 0xC9,             /* vxorps xmm1, xmm1, xmm1 */
				0xC4, 0x41, 0x19, 0xEF, 0xE3,       /* vpxor xmm12, xmm12, xmm11 */
				0xC4, 0x63, 0x0D, 0x18, 0xF1, 0x01, /* vinsertf128 ymm14,ymm14,xmm1,1 */
				0xC4, 0x43, 0x0D, 0x0C, 0xF7, 0x10, /* vblendps ymm14,ymm14,ymm15,0x10 */
				0xC4, 0x42, 0x11, 0xDF, 0xEC,       /* vaesdeclast xmm13,xmm13,xmm12 */
				0xC3
			};
			State state{};
			const uint32_t xmm0In[4] = { 1u, 2u, 3u, 4u };
			const uint32_t xmm11In[4] = { 0x11111111u, 0x22222222u, 0x33333333u,
				0x44444444u };
			const uint32_t xmm12In[4] = { 0xAAAAAAAAu, 0xBBBBBBBBu, 0xCCCCCCCCu,
				0xDDDDDDDDu };
			const uint32_t ymm14In[8] = { 10u, 20u, 30u, 40u, 50u, 60u, 70u, 80u };
			const uint32_t ymm15In[8] = { 0u, 0u, 0u, 0u, 0xDEADBEEFu, 0u, 0u, 0u };
			memcpy(state.ymm[0], xmm0In, sizeof(xmm0In));
			memcpy(state.ymm[11], xmm11In, sizeof(xmm11In));
			memcpy(state.ymm[12], xmm12In, sizeof(xmm12In));
			memcpy(state.ymm[13], xmm0In, sizeof(xmm0In));
			memcpy(state.ymm[14], ymm14In, sizeof(ymm14In));
			memcpy(state.ymm[15], ymm15In, sizeof(ymm15In));
			Result result{};
			RunCode(code, sizeof(code), state, result);
			uint32_t xmm10[4]{}, xmm1[4]{}, xmm12[4]{}, ymm14[8]{};
			memcpy(xmm10, state.ymm[10], sizeof(xmm10));
			memcpy(xmm1, state.ymm[1], sizeof(xmm1));
			memcpy(xmm12, state.ymm[12], sizeof(xmm12));
			memcpy(ymm14, state.ymm[14], sizeof(ymm14));
			if (result.reason != StopReason::Completed
				|| xmm10[0] != 1u || xmm10[3] != 4u
				|| xmm1[0] != 0u || xmm1[3] != 0u
				|| xmm12[0] != (0xAAAAAAAAu ^ 0x11111111u)
				|| ymm14[0] != 10u || ymm14[4] != 0xDEADBEEFu
				|| ymm14[5] != 0u)
				return fail("vex-subset", result);
			/* vaesdeclast must have mutated xmm13 low lane. */
			uint32_t xmm13[4]{};
			memcpy(xmm13, state.ymm[13], sizeof(xmm13));
			if (xmm13[0] == xmm0In[0] && xmm13[1] == xmm0In[1]
				&& xmm13[2] == xmm0In[2] && xmm13[3] == xmm0In[3])
				return fail("vaesdeclast-noop", result);
		}

		{
			/* vptest xmm12, xmm12 followed by jne/je on ZF. */
			const uint8_t codeZero[] = {
				0xC4, 0x42, 0x79, 0x17, 0xE4, /* vptest xmm12, xmm12 */
				0x75, 0x01,                   /* jne +1 */
				0xC3,                         /* ret (ZF path) */
				0xCC                          /* int3 if ZF clear */
			};
			State state{};
			Result result{};
			RunCode(codeZero, sizeof(codeZero), state, result);
			if (result.reason != StopReason::Completed
				|| (state.rflags & FlagZF) == 0u
				|| (state.rflags & FlagCF) == 0u)
				return fail("vptest-zero", result);

			const uint8_t codeNz[] = {
				0xC4, 0x42, 0x79, 0x17, 0xE4,
				0x74, 0x01,                   /* je +1 */
				0xC3,
				0xCC
			};
			state = State{};
			const uint32_t nz[4] = { 1u, 0u, 0u, 0u };
			memcpy(state.ymm[12], nz, sizeof(nz));
			result = Result{};
			RunCode(codeNz, sizeof(codeNz), state, result);
			if (result.reason != StopReason::Completed
				|| (state.rflags & FlagZF) != 0u)
				return fail("vptest-nonzero", result);
		}

		{
			const uint8_t code[] = {
				0xC4, 0xC1, 0x19, 0x72, 0xD3, 0x04, /* vpsrld xmm12, xmm11, 4 (vvvv=12, rm=11) */
				0xC4, 0x41, 0x19, 0xFE, 0xE4,       /* vpaddd xmm12, xmm12, xmm12 */
				0xC4, 0x41, 0x79, 0x70, 0xE4, 0x1B, /* vpshufd xmm12, xmm12, 0x1B */
				0xC3
			};
			State state{};
			const uint32_t xmm11In[4] = { 0x100u, 0x200u, 0x300u, 0x400u };
			memcpy(state.ymm[11], xmm11In, sizeof(xmm11In));
			Result result{};
			RunCode(code, sizeof(code), state, result);
			uint32_t xmm12[4]{};
			memcpy(xmm12, state.ymm[12], sizeof(xmm12));
			/* >>4 => {0x10,0x20,0x30,0x40}; +self => {0x20,0x40,0x60,0x80};
			 * pshufd 1B => {0x80,0x60,0x40,0x20} */
			if (result.reason != StopReason::Completed
				|| xmm12[0] != 0x80u || xmm12[1] != 0x60u
				|| xmm12[2] != 0x40u || xmm12[3] != 0x20u)
				return fail("vex-shift-add-shuf", result);
		}

		{
			struct SyscallProbe
			{
				uint32_t calls{};
				uint64_t rip{};
				uint64_t number{};
				uint64_t arg0{};
			} probe{};
			const uint8_t code[] = {
				0xB8, 0x34, 0x12, 0x00, 0x00, /* mov eax,1234h */
				0x0F, 0x05,                   /* syscall */
				0x83, 0xC0, 0x01,             /* add eax,1 */
				0xC3
			};
			State state{};
			state.gpr[R10] = 0x1122334455667788ull;
			Result result{};
			const SyscallHookFn hook = [](void* context, uint64_t rip,
				State* hookedState) -> CallHookResult
			{
				SyscallProbe* p = static_cast<SyscallProbe*>(context);
				if (!p || !hookedState)
					return CallHookResult::Fault;
				++p->calls;
				p->rip = rip;
				p->number = hookedState->gpr[RAX];
				p->arg0 = hookedState->gpr[R10];
				hookedState->gpr[RAX] = 0x29u;
				return CallHookResult::Handled;
			};
			RunCode(code, sizeof(code), state, result, nullptr, 0, 0, 256u,
				hook, &probe);
			if (result.reason != StopReason::Completed || probe.calls != 1u
				|| probe.rip != 0x10000005ull || probe.number != 0x1234u
				|| probe.arg0 != 0x1122334455667788ull
				|| uint32_t(state.gpr[RAX]) != 0x2Au
				|| state.gpr[RCX] != 0x10000007ull)
				return fail("syscall-hook", result);
		}

		{
			/* vaesenc / vpcmpeqd. */
			const uint8_t code[] = {
				0xC4, 0xE2, 0x71, 0xDC, 0xC1, /* vaesenc xmm0, xmm1, xmm1 */
				0xC5, 0xF9, 0x76, 0xC0,       /* vpcmpeqd xmm0, xmm0, xmm0 */
				0xC3
			};
			State state{};
			const uint32_t ones[4] = { 1u, 2u, 3u, 4u };
			memcpy(state.ymm[0], ones, sizeof(ones));
			memcpy(state.ymm[1], ones, sizeof(ones));
			Result result{};
			RunCode(code, sizeof(code), state, result);
			uint32_t out[4]{};
			memcpy(out, state.ymm[0], sizeof(out));
			if (result.reason != StopReason::Completed
				|| out[0] != 0xFFFFFFFFu || out[3] != 0xFFFFFFFFu)
				return fail("vaesenc-vpcmpeqd", result);
		}

		{
			/* 66 0F 38 DC = aesenc (legacy, non-VEX). */
			const uint8_t code[] = {
				0x66, 0x0F, 0x38, 0xDC, 0xC1, /* aesenc xmm0, xmm1 */
				0xC3
			};
			State state{};
			memset(state.ymm[0], 0x11, 16u);
			memset(state.ymm[1], 0x22, 16u);
			Result result{};
			RunCode(code, sizeof(code), state, result);
			if (result.reason != StopReason::Completed)
				return fail("legacy-aesenc", result);
		}

		{
			/* Cross-page instruction: vmovaps straddles two 4KB pages.
			 * Stack return must land on stopRip; a bare C3 with garbage RIP
			 * would fault after the vector op and mask a successful stitch. */
			uint8_t pageA[0x1000]{};
			uint8_t pageB[0x1000]{};
			pageA[0x0FFE] = 0xC5;
			pageA[0x0FFF] = 0xF8;
			pageB[0x0000] = 0x28;
			pageB[0x0001] = 0xC1; /* vmovaps xmm0, xmm1 */
			pageB[0x0002] = 0xC3;
			TestMemory memory{};
			memory.ranges[memory.count++] = { 0x3000F000ull, pageA, sizeof(pageA), false };
			memory.ranges[memory.count++] = { 0x30010000ull, pageB, sizeof(pageB), false };
			uint8_t stack[256]{};
			const uint64_t stackBase = 0x40000000ull;
			const uint64_t initialRsp = stackBase + 0xF0ull;
			const uint64_t stopRip = stackBase + 0xE0ull;
			memcpy(stack + 0xF0u, &stopRip, sizeof(stopRip));
			memory.ranges[memory.count++] = { stackBase, stack, sizeof(stack), true };
			State state{};
			const uint32_t xmm1In[4] = { 9u, 8u, 7u, 6u };
			memcpy(state.ymm[1], xmm1In, sizeof(xmm1In));
			state.rip = 0x3000FFFEull;
			state.gpr[RSP] = initialRsp;
			Vm::Options options{};
			options.read = TestRead;
			options.write = TestWrite;
			options.memoryContext = &memory;
			options.stopRip = stopRip;
			options.instructionLimit = 16u;
			Result result = Vm::Run(&state, &options);
			uint32_t xmm0[4]{};
			memcpy(xmm0, state.ymm[0], sizeof(xmm0));
			if ((result.reason != StopReason::Completed
					&& result.reason != StopReason::StopRip)
				|| xmm0[0] != 9u || xmm0[3] != 6u)
				return fail("cross-page-vmovaps", result);
		}

		if (detail && detailCapacity)
			std::snprintf(detail, detailCapacity, "ok cases=44");
		return true;
	}
}
