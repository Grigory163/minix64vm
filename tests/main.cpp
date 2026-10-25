#include "MiniX64Vm.h"

#include <cstdio>

int main()
{
	char detail[256]{};
	const bool ok = MiniVm::MiniX64Vm::RunSelfTests(detail, sizeof(detail));
	std::printf("%s: %s\n", ok ? "PASS" : "FAIL", detail);
	return ok ? 0 : 1;
}
