
using u32 = unsigned int;

extern "C" __global__ void reduce0(int* g_idata, int* g_odata)
{
	const u32 tIdx = threadIdx.x;
	const u32 gIdx = blockIdx.x * blockDim.x + threadIdx.x;
}