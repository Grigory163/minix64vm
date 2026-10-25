/* VEX (C4/C5) instruction subset, including VAES.
 * Included inside MiniX64Vm.cpp anonymous namespace (Runner methods). */

bool Runner::ReadYmmRm(const ModRm& modrm, uint64_t nextRip, uint8_t* value,
	size_t bytes)
{
	if (!value || (bytes != 16u && bytes != 32u))
		return false;
	if (modrm.rm.isRegister)
	{
		memcpy(value, state_.ymm[modrm.rm.registerIndex], bytes);
		return true;
	}
	return ReadMemory(EffectiveAddress(modrm.rm, nextRip), value, bytes);
}

bool Runner::WriteYmmRm(const ModRm& modrm, uint64_t nextRip, const uint8_t* value,
	size_t bytes)
{
	if (!value || (bytes != 16u && bytes != 32u))
		return false;
	if (modrm.rm.isRegister)
	{
		memcpy(state_.ymm[modrm.rm.registerIndex], value, bytes);
		if (bytes == 16u)
			memset(state_.ymm[modrm.rm.registerIndex] + 16u, 0, 16u);
		return true;
	}
	return WriteMemory(EffectiveAddress(modrm.rm, nextRip), value, bytes);
}

void Runner::WriteYmmDest(uint8_t index, const uint8_t* value, size_t bytes)
{
	if (index >= 16u || !value)
		return;
	if (bytes >= 32u)
	{
		memcpy(state_.ymm[index], value, 32u);
		return;
	}
	memcpy(state_.ymm[index], value, 16u);
	memset(state_.ymm[index] + 16u, 0, 16u);
}

bool Runner::ExecuteVex(uint8_t lead, Cursor& cursor, uint64_t instructionRip,
	const char*& mnemonic)
{
	uint8_t map = 1u;
	uint8_t pp = 0;
	bool L = false;
	bool W = false;
	Prefix prefix{};
	prefix.rexPresent = true;
	uint8_t vvvv = 0;

	if (lead == 0xC5u)
	{
		uint8_t b1 = 0;
		if (!cursor.Read8(b1))
			return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "vex2");
		if ((b1 & 0x80u) == 0u)
			prefix.rex |= 4u; /* R */
		vvvv = uint8_t((~uint8_t((b1 >> 3) & 0x0Fu)) & 0x0Fu);
		L = (b1 & 4u) != 0u;
		pp = uint8_t(b1 & 3u);
		map = 1u;
	}
	else if (lead == 0xC4u)
	{
		uint8_t b1 = 0, b2 = 0;
		if (!cursor.Read8(b1) || !cursor.Read8(b2))
			return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "vex3");
		if ((b1 & 0x80u) == 0u)
			prefix.rex |= 4u; /* R */
		if ((b1 & 0x40u) == 0u)
			prefix.rex |= 2u; /* X */
		if ((b1 & 0x20u) == 0u)
			prefix.rex |= 1u; /* B */
		map = uint8_t(b1 & 0x1Fu);
		W = (b2 & 0x80u) != 0u;
		if (W)
			prefix.rex |= 8u;
		vvvv = uint8_t((~uint8_t((b2 >> 3) & 0x0Fu)) & 0x0Fu);
		L = (b2 & 4u) != 0u;
		pp = uint8_t(b2 & 3u);
	}
	else
		return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "vex-lead");

	if (pp == 1u)
		prefix.operand16 = true; /* embedded 66 */
	else if (pp == 2u)
		prefix.rep = true; /* F3 */
	else if (pp == 3u)
		prefix.repne = true; /* F2 */

	uint8_t opcode = 0;
	if (!cursor.Read8(opcode))
		return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "vex-op");

	const size_t width = L ? 32u : 16u;

	/* VEX.66.0F.WIG 72 /2 ib: destination=vvvv, source=r/m.
	 * Unlike most VEX instructions ModRM.reg is the opcode extension. */
	if (map == 1u && opcode == 0x72u && pp == 1u)
	{
		ModRm modrm{};
		if (!DecodeModRm(cursor, prefix, modrm))
			return Finish(Vm::StopReason::InvalidInstruction, instructionRip,
				"vpsrld-modrm");
		uint8_t imm = 0;
		if (!cursor.Read8(imm))
			return Finish(Vm::StopReason::InvalidInstruction, instructionRip,
				"vpsrld-imm");
		currentLength_ = cursor.at;
		const uint64_t nextRip = instructionRip + cursor.at;
		if (modrm.regRaw != 2u)
			return Finish(Vm::StopReason::UnsupportedInstruction, instructionRip,
				"vex-72-group");
		uint8_t source[32]{};
		if (!ReadYmmRm(modrm, nextRip, source, width))
			return Finish(Vm::StopReason::ReadFault, instructionRip,
				"vpsrld-read", currentFaultAddress_);
		uint8_t output[32]{};
		const unsigned shift = imm > 31u ? 32u : imm;
		const unsigned dwords = unsigned(width / 4u);
		for (unsigned i = 0; i < dwords; ++i)
		{
			uint32_t word = 0;
			memcpy(&word, source + i * 4u, 4u);
			word = shift >= 32u ? 0u : (word >> shift);
			memcpy(output + i * 4u, &word, 4u);
		}
		WriteYmmDest(vvvv, output, width);
		mnemonic = "vpsrld";
		state_.rip = nextRip;
		return true;
	}

	ModRm modrm{};
	if (!DecodeModRm(cursor, prefix, modrm))
		return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "vex-modrm");

	uint8_t imm = 0;
	const bool needsImm = (map == 1u && opcode == 0x70u)
		|| (map == 3u && (opcode == 0x0Cu || opcode == 0x16u
			|| opcode == 0x18u || opcode == 0x19u));
	if (needsImm && !cursor.Read8(imm))
		return Finish(Vm::StopReason::InvalidInstruction, instructionRip, "vex-imm");
	currentLength_ = cursor.at;
	const uint64_t nextRip = instructionRip + cursor.at;

	auto readRight = [&](uint8_t* out, size_t bytes) -> bool {
		return ReadYmmRm(modrm, nextRip, out, bytes);
	};

	/* map1 = legacy 0F */
	if (map == 1u)
	{
		if (opcode == 0x28u || opcode == 0x10u || opcode == 0x6Fu)
		{
			uint8_t value[32]{};
			if (!readRight(value, width))
				return Finish(Vm::StopReason::ReadFault, instructionRip, "vmov-read",
					currentFaultAddress_);
			WriteYmmDest(modrm.regIndex, value, width);
			mnemonic = opcode == 0x6Fu ? "vmovdqa" : "vmovaps";
			state_.rip = nextRip;
			return true;
		}
		if (opcode == 0x29u || opcode == 0x11u || opcode == 0x7Fu)
		{
			if (!WriteYmmRm(modrm, nextRip, state_.ymm[modrm.regIndex], width))
				return Finish(Vm::StopReason::WriteFault, instructionRip, "vmov-write",
					currentFaultAddress_);
			mnemonic = opcode == 0x7Fu ? "vmovdqa" : "vmovaps";
			state_.rip = nextRip;
			return true;
		}
		if (opcode == 0x57u || opcode == 0xEFu || opcode == 0x54u || opcode == 0x56u)
		{
			uint8_t right[32]{};
			if (!readRight(right, width))
				return Finish(Vm::StopReason::ReadFault, instructionRip, "vxor-read",
					currentFaultAddress_);
			uint8_t output[32]{};
			for (size_t i = 0; i < width; ++i)
			{
				const uint8_t left = state_.ymm[vvvv][i];
				if (opcode == 0x54u)
					output[i] = uint8_t(left & right[i]);
				else if (opcode == 0x56u)
					output[i] = uint8_t(left | right[i]);
				else
					output[i] = uint8_t(left ^ right[i]);
			}
			WriteYmmDest(modrm.regIndex, output, width);
			mnemonic = opcode == 0xEFu ? "vpxor"
				: (opcode == 0x54u ? "vandps"
					: (opcode == 0x56u ? "vorps" : "vxorps"));
			state_.rip = nextRip;
			return true;
		}
		if (opcode == 0xFEu)
		{
			uint8_t right[32]{};
			if (!readRight(right, width))
				return Finish(Vm::StopReason::ReadFault, instructionRip, "vpaddd-read",
					currentFaultAddress_);
			uint8_t output[32]{};
			const unsigned dwords = unsigned(width / 4u);
			for (unsigned i = 0; i < dwords; ++i)
			{
				uint32_t a = 0, b = 0, c = 0;
				memcpy(&a, state_.ymm[vvvv] + i * 4u, 4u);
				memcpy(&b, right + i * 4u, 4u);
				c = a + b;
				memcpy(output + i * 4u, &c, 4u);
			}
			WriteYmmDest(modrm.regIndex, output, width);
			mnemonic = "vpaddd";
			state_.rip = nextRip;
			return true;
		}
		if (opcode == 0x70u)
		{
			uint8_t source[32]{};
			if (!readRight(source, 16u))
				return Finish(Vm::StopReason::ReadFault, instructionRip, "vpshufd-read",
					currentFaultAddress_);
			uint8_t output[32]{};
			for (unsigned i = 0; i < 4u; ++i)
				memcpy(output + i * 4u, source + ((imm >> (i * 2u)) & 3u) * 4u, 4u);
			WriteYmmDest(modrm.regIndex, output, 16u);
			mnemonic = "vpshufd";
			state_.rip = nextRip;
			return true;
		}
		if (opcode == 0x76u)
		{
			uint8_t right[32]{};
			if (!readRight(right, width))
				return Finish(Vm::StopReason::ReadFault, instructionRip,
					"vpcmpeqd-read", currentFaultAddress_);
			uint8_t output[32]{};
			const unsigned dwords = unsigned(width / 4u);
			for (unsigned i = 0; i < dwords; ++i)
			{
				uint32_t a = 0, b = 0, c = 0;
				memcpy(&a, state_.ymm[vvvv] + i * 4u, 4u);
				memcpy(&b, right + i * 4u, 4u);
				c = a == b ? 0xFFFFFFFFu : 0u;
				memcpy(output + i * 4u, &c, 4u);
			}
			WriteYmmDest(modrm.regIndex, output, width);
			mnemonic = "vpcmpeqd";
			state_.rip = nextRip;
			return true;
		}
		if (opcode == 0x74u && pp == 1u)
		{
			uint8_t right[32]{};
			if (!readRight(right, width))
				return Finish(Vm::StopReason::ReadFault, instructionRip,
					"vpcmpeqb-read", currentFaultAddress_);
			uint8_t output[32]{};
			for (size_t i = 0; i < width; ++i)
				output[i] = state_.ymm[vvvv][i] == right[i] ? 0xFFu : 0u;
			WriteYmmDest(modrm.regIndex, output, width);
			mnemonic = "vpcmpeqb";
			state_.rip = nextRip;
			return true;
		}
		if (opcode == 0xD1u && pp == 1u)
		{
			/* vpsrlw xmm/ymm, xmm/ymm, xmm/m128 */
			uint8_t right[32]{};
			if (!readRight(right, width))
				return Finish(Vm::StopReason::ReadFault, instructionRip,
					"vpsrlw-read", currentFaultAddress_);
			uint64_t count = 0;
			memcpy(&count, right, sizeof(count));
			uint8_t output[32]{};
			for (size_t offset = 0; offset < width; offset += 2u)
			{
				uint16_t value = 0;
				memcpy(&value, state_.ymm[vvvv] + offset, sizeof(value));
				value = count >= 16u ? 0u : uint16_t(value >> count);
				memcpy(output + offset, &value, sizeof(value));
			}
			WriteYmmDest(modrm.regIndex, output, width);
			mnemonic = "vpsrlw";
			state_.rip = nextRip;
			return true;
		}
		if (opcode == 0x17u && pp == 1u)
		{
			/* vtestps: same ZF/CF contract as vptest on bit masks. */
			uint8_t right[32]{};
			if (!readRight(right, width))
				return Finish(Vm::StopReason::ReadFault, instructionRip,
					"vtestps-read", currentFaultAddress_);
			const uint8_t* left = state_.ymm[modrm.regIndex];
			bool zf = true;
			bool cf = true;
			for (size_t i = 0; i < width; ++i)
			{
				if ((uint8_t(left[i] & right[i])) != 0u)
					zf = false;
				if ((uint8_t((~left[i]) & right[i])) != 0u)
					cf = false;
			}
			SetFlag(Vm::FlagZF, zf);
			SetFlag(Vm::FlagCF, cf);
			SetFlag(Vm::FlagOF, false);
			SetFlag(Vm::FlagSF, false);
			SetFlag(Vm::FlagAF, false);
			SetFlag(Vm::FlagPF, false);
			mnemonic = "vtestps";
			state_.rip = nextRip;
			return true;
		}
		if (opcode == 0xD7u && pp == 1u)
		{
			/* vpmovmskb r32, xmm/ymm */
			uint8_t source[32]{};
			if (!readRight(source, width))
				return Finish(Vm::StopReason::ReadFault, instructionRip,
					"vpmovmskb-read", currentFaultAddress_);
			uint32_t mask = 0;
			const unsigned bytes = unsigned(width);
			for (unsigned i = 0; i < bytes; ++i)
				mask |= uint32_t((source[i] >> 7u) & 1u) << i;
			WriteRegister(modrm.regRaw, modrm.regIndex, 32u, true, mask);
			mnemonic = "vpmovmskb";
			state_.rip = nextRip;
			return true;
		}
		if (opcode == 0x6Eu)
		{
			const unsigned bits = W ? 64u : 32u;
			uint64_t value = 0;
			if (!ReadRm(modrm, prefix, bits, nextRip, value))
				return Finish(Vm::StopReason::ReadFault, instructionRip, "vmovq-read",
					currentFaultAddress_);
			uint8_t output[16]{};
			memcpy(output, &value, bits / 8u);
			WriteYmmDest(modrm.regIndex, output, 16u);
			mnemonic = bits == 64u ? "vmovq" : "vmovd";
			state_.rip = nextRip;
			return true;
		}
		if (opcode == 0x7Eu || opcode == 0xD6u)
		{
			const unsigned bits = (W || opcode == 0xD6u) ? 64u : 32u;
			uint64_t value = 0;
			memcpy(&value, state_.ymm[modrm.regIndex], bits / 8u);
			if (!WriteRm(modrm, prefix, bits, nextRip, value))
				return Finish(Vm::StopReason::WriteFault, instructionRip,
					"vmovq-write", currentFaultAddress_);
			mnemonic = bits == 64u ? "vmovq" : "vmovd";
			state_.rip = nextRip;
			return true;
		}
		return Finish(Vm::StopReason::UnsupportedInstruction, instructionRip,
			"vex-0f-unsupported");
	}

	/* map2 = 0F38 */
	if (map == 2u)
	{
		if (opcode == 0x40u)
		{
			uint8_t right[32]{};
			if (!readRight(right, width))
				return Finish(Vm::StopReason::ReadFault, instructionRip, "vpmulld-read",
					currentFaultAddress_);
			uint8_t output[32]{};
			const unsigned dwords = unsigned(width / 4u);
			for (unsigned i = 0; i < dwords; ++i)
			{
				uint32_t a = 0, b = 0, c = 0;
				memcpy(&a, state_.ymm[vvvv] + i * 4u, 4u);
				memcpy(&b, right + i * 4u, 4u);
				c = uint32_t(uint64_t(a) * uint64_t(b));
				memcpy(output + i * 4u, &c, 4u);
			}
			WriteYmmDest(modrm.regIndex, output, width);
			mnemonic = "vpmulld";
			state_.rip = nextRip;
			return true;
		}
		if (opcode == 0x17u && pp == 1u)
		{
			/* vptest xmm/ymm, xmm/ymm/m - ZF=(dst&src)==0; CF=(~dst&src)==0. */
			uint8_t right[32]{};
			if (!readRight(right, width))
				return Finish(Vm::StopReason::ReadFault, instructionRip,
					"vptest-read", currentFaultAddress_);
			const uint8_t* left = state_.ymm[modrm.regIndex];
			bool zf = true;
			bool cf = true;
			for (size_t i = 0; i < width; ++i)
			{
				if ((uint8_t(left[i] & right[i])) != 0u)
					zf = false;
				if ((uint8_t((~left[i]) & right[i])) != 0u)
					cf = false;
			}
			SetFlag(Vm::FlagZF, zf);
			SetFlag(Vm::FlagCF, cf);
			SetFlag(Vm::FlagOF, false);
			SetFlag(Vm::FlagSF, false);
			SetFlag(Vm::FlagAF, false);
			SetFlag(Vm::FlagPF, false);
			mnemonic = "vptest";
			state_.rip = nextRip;
			return true;
		}
		if ((opcode == 0xDEu || opcode == 0xDFu || opcode == 0xDBu
			|| opcode == 0xDCu || opcode == 0xDDu) && pp == 1u)
		{
			uint8_t stateBytes[16]{};
			uint8_t key[16]{};
			if (opcode == 0xDBu)
			{
				/* vaesimc xmm, xmm/m128 - dest=reg, src=rm; vvvv unused */
				if (!readRight(key, 16u))
					return Finish(Vm::StopReason::ReadFault, instructionRip,
						"vaesimc-read", currentFaultAddress_);
				memcpy(stateBytes, key, sizeof(stateBytes));
				AesImc128(stateBytes);
				WriteYmmDest(modrm.regIndex, stateBytes, 16u);
				mnemonic = "vaesimc";
			}
			else
			{
				memcpy(stateBytes, state_.ymm[vvvv], 16u);
				if (!readRight(key, 16u))
					return Finish(Vm::StopReason::ReadFault, instructionRip,
						"vaes-round-read", currentFaultAddress_);
				const bool last = opcode == 0xDFu || opcode == 0xDDu;
				const bool enc = opcode == 0xDCu || opcode == 0xDDu;
				if (enc)
					AesEnc128(stateBytes, key, last);
				else
					AesDec128(stateBytes, key, last);
				WriteYmmDest(modrm.regIndex, stateBytes, 16u);
				mnemonic = opcode == 0xDCu ? "vaesenc"
					: (opcode == 0xDDu ? "vaesenclast"
						: (opcode == 0xDEu ? "vaesdec" : "vaesdeclast"));
			}
			state_.rip = nextRip;
			return true;
		}
		return Finish(Vm::StopReason::UnsupportedInstruction, instructionRip,
			"vex-0f38-unsupported");
	}

	/* map3 = 0F3A */
	if (map == 3u)
	{
		if (opcode == 0x0Cu && pp == 1u)
		{
			uint8_t right[32]{};
			if (!readRight(right, width))
				return Finish(Vm::StopReason::ReadFault, instructionRip,
					"vblendps-read", currentFaultAddress_);
			uint8_t output[32]{};
			const unsigned dwords = unsigned(width / 4u);
			for (unsigned i = 0; i < dwords; ++i)
			{
				const uint8_t* src = ((imm >> i) & 1u)
					? right : state_.ymm[vvvv];
				memcpy(output + i * 4u, src + i * 4u, 4u);
			}
			WriteYmmDest(modrm.regIndex, output, width);
			mnemonic = "vblendps";
			state_.rip = nextRip;
			return true;
		}
		if (opcode == 0x18u && pp == 1u)
		{
			/* vinsertf128 ymm1, ymm2, xmm3/m128, imm8 */
			uint8_t lane[16]{};
			if (!readRight(lane, 16u))
				return Finish(Vm::StopReason::ReadFault, instructionRip,
					"vinsertf128-read", currentFaultAddress_);
			uint8_t output[32]{};
			memcpy(output, state_.ymm[vvvv], 32u);
			memcpy(output + ((imm & 1u) ? 16u : 0u), lane, 16u);
			WriteYmmDest(modrm.regIndex, output, 32u);
			mnemonic = "vinsertf128";
			state_.rip = nextRip;
			return true;
		}
		if (opcode == 0x19u && pp == 1u)
		{
			/* vextractf128 xmm/m128, ymm2, imm8 - dest=r/m, src=reg */
			uint8_t lane[16]{};
			memcpy(lane, state_.ymm[modrm.regIndex] + ((imm & 1u) ? 16u : 0u), 16u);
			if (!WriteYmmRm(modrm, nextRip, lane, 16u))
				return Finish(Vm::StopReason::WriteFault, instructionRip,
					"vextractf128-write", currentFaultAddress_);
			mnemonic = "vextractf128";
			state_.rip = nextRip;
			return true;
		}
		if (opcode == 0x16u && pp == 1u && W)
		{
			/* vpextrq r/m64, xmm, imm8 */
			uint64_t value = 0;
			memcpy(&value, state_.ymm[modrm.regIndex] + ((imm & 1u) ? 8u : 0u), 8u);
			if (!WriteRm(modrm, prefix, 64u, nextRip, value))
				return Finish(Vm::StopReason::WriteFault, instructionRip,
					"vpextrq-write", currentFaultAddress_);
			mnemonic = "vpextrq";
			state_.rip = nextRip;
			return true;
		}
		return Finish(Vm::StopReason::UnsupportedInstruction, instructionRip,
			"vex-0f3a-unsupported");
	}

	return Finish(Vm::StopReason::UnsupportedInstruction, instructionRip,
		"vex-map-unsupported");
}
