#include <dependencies/Orochi/Orochi/OrochiUtils.h>
#include <src/Error.h>
#include <src/Kernel.h>
#include <src/Timer.h>

#include <vector>
#include <algorithm>
#include <filesystem>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <optional>
#include <cstdlib> 
#include <ctime>
#include <cassert>

using namespace GpuOptimizationProject;

using u32 = unsigned int;
using u64 = unsigned long;
enum Error
{
	Success,
	Failure
};

static std::string readSourceCode(const std::filesystem::path& path, std::optional<std::vector<std::filesystem::path>*> includes)
{
	std::string	  src;
	std::ifstream file(path);
	if (!file.is_open())
	{
		std::string msg = "Unable to open " + path.string();
		throw std::runtime_error(msg);
	}
	size_t sizeFile;
	file.seekg(0, std::ifstream::end);
	size_t size = sizeFile = static_cast<size_t>(file.tellg());
	file.seekg(0, std::ifstream::beg);
	if (includes.has_value())
	{
		std::string line;
		while (std::getline(file, line))
		{
			if (line.find("#include") != std::string::npos)
			{
				size_t		pa = line.find("<");
				size_t		pb = line.find(">");
				std::string buf = line.substr(pa + 1, pb - pa - 1);
				includes.value()->push_back(buf);
				src += line + '\n';
			}
			src += line + '\n';
		}
	}
	else
	{
		src.resize(size, ' ');
		file.read(&src[0], size);
	}
	return src;
}

Error buildKernelFromSrc(Kernel& kernel, oroDevice& device, const std::filesystem::path& srcPath, const std::string& functionName, std::optional<std::vector<const char*>> opts)
{
	oroFunction function = nullptr;
	std::vector<char> codec;
	std::vector<const char*> options;
	if (opts)
	{
		options = *opts;
	}
#if defined(HLT_DEBUG_GPU)
	options.push_back("-G");
#endif
	const bool isAmd = oroGetCurAPI(0) == ORO_API_HIP;
	std::string sarg;
	if (isAmd)
	{
		oroDeviceProp props;
		oroGetDeviceProperties(&props, device);
		sarg = std::string("--gpu-architecture=") + props.gcnArchName;
		options.push_back(sarg.c_str());
	}
	else
	{
		options.push_back("--device-c");
		options.push_back("-arch=compute_60");
	}
	options.push_back("-I../dependencies/Orochi/");
	options.push_back("-I../");
	options.push_back("-std=c++17");

	std::vector<std::filesystem::path> includeNamesData;
	std::string srcCode = readSourceCode(srcPath, &includeNamesData);
	if (srcCode.empty())
	{
		std::cerr << "Unable to open '" + srcPath.string() + "'" + "\n";
		return Error::Failure;
	}
	orortcProgram prog = nullptr;
	CHECK_ORORTC(orortcCreateProgram(&prog, srcCode.data(), functionName.c_str(), 0, nullptr, nullptr));

	orortcResult e = orortcCompileProgram(prog, static_cast<int>(options.size()), options.data());
	if (e != ORORTC_SUCCESS)
	{
		size_t logSize;
		CHECK_ORORTC(orortcGetProgramLogSize(prog, &logSize));

		if (logSize)
		{
			std::string log(logSize, '\0');
			CHECK_ORORTC(orortcGetProgramLog(prog, &log[0]));
			std::cerr << log;
			return Error::Failure;
		}
	}

	size_t codeSize = 0;
	orortcGetCodeSize(prog, &codeSize);

	codec.resize(codeSize);
	orortcGetCode(prog, codec.data());

	orortcDestroyProgram(&prog);

	oroModule module = nullptr;
	oroModuleLoadData(&module, codec.data());
	oroModuleGetFunction(&function, module, functionName.c_str());

	kernel.setFunction(function);

	return Error::Success;
}

int main(int argc, char* argv[])
{
	try
	{
		oroDevice	orochiDevice;
		oroCtx		orochiCtxt;
		Timer timer;
		enum {
			reductionTime = 0
		};

		CHECK_ORO((oroError)oroInitialize((oroApi)(ORO_API_HIP | ORO_API_CUDA), 0));
		CHECK_ORO(oroInit(0));
		CHECK_ORO(oroDeviceGet(&orochiDevice, 0)); // deviceId should be taken from user?
		CHECK_ORO(oroCtxCreate(&orochiCtxt, 0, orochiDevice));

		Kernel		reductionKernel;
		std::string functionName = "reduce_0";

		buildKernelFromSrc(
			reductionKernel,
			orochiDevice,
			"../src/ReductionKernel.h",
			functionName.c_str(),
			std::nullopt);

		//prepare host data 
		std::srand((unsigned int)std::time(0));
		const u32 dataSize = std::pow(2, 22) * 40; // 160m ints
		
		auto genRandomNumbers = [&]() -> auto {
			return (std::rand() % 10 + 1);
		};
		
		std::vector<int> h_inData(dataSize);
		std::generate(h_inData.begin(), h_inData.end(), genRandomNumbers);

		//prepare device data
		int* d_inData = NULL;
		int* d_outData = NULL;
		
		OrochiUtils::malloc(d_inData, dataSize);
		assert(dataSize != NULL);
		OrochiUtils::copyHtoD(d_inData, h_inData.data(), dataSize);

		constexpr int blockSize = 64;
		int gridSize = static_cast<int>((dataSize + blockSize - 1) / blockSize);
		
		OrochiUtils::malloc(d_outData, gridSize);
		OrochiUtils::memset(d_outData, 0, sizeof(int) * gridSize);
		assert(d_outData != NULL);
		
		std::vector<int> test(gridSize, 0);

		while (true)
		{
			reductionKernel.setArgs({ d_inData, d_outData});
			timer.measure(reductionTime, [&]() { reductionKernel.launch(gridSize, 1, 1, blockSize, 1, 1, 0, 0); });
			OrochiUtils::memset(d_inData, 0, sizeof(int) * dataSize);
			OrochiUtils::copyDtoD(d_inData, d_outData, gridSize);
			OrochiUtils::copyDtoH(test.data(), d_inData, gridSize);
			if (gridSize <= 1)
				break;
			gridSize = static_cast<int>((gridSize + blockSize - 1) / blockSize);
		}

		int d_reductionVal = 0;
		OrochiUtils::copyDtoH(&d_reductionVal, d_inData, 1);
		assert(d_reductionVal != 0);

		//calculate reduction on cpu 
		int h_reductionVal = 0;
		for (size_t i = 0; i < h_inData.size(); i++)
		{
			h_reductionVal += h_inData[i];
		}

		assert(h_reductionVal == d_reductionVal);

		std::cout << "Total time in Ms : " << timer.getTimeRecord(reductionTime);

		//Always cleanup if you dont may Satan haunt you in dreams!
		OrochiUtils::free(d_inData);
		OrochiUtils::free(d_outData);
		CHECK_ORO(oroCtxDestroy(orochiCtxt));
	}
	catch (std::exception& e)
	{
		std::cerr << e.what();
		return -1;
	}
	return 0;
}