
using u32 = unsigned int;

extern "C" __global__ void reduce_0(int* d_inData, int* d_outData)
{
	const u32 tIdx = threadIdx.x;
	const u32 gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	constexpr u32 blockSize = 64;
	__shared__ int s_data[blockSize];
	s_data[tIdx] = d_inData[gIdx];
	__syncthreads();

	for (u32 stride = 1; stride < blockDim.x; stride *= 2)
	{
		if (tIdx % (2 * stride) == 0)
		{
			s_data[tIdx] += s_data[tIdx + stride];
		}
		__syncthreads();
	}

	if (tIdx == 0) d_outData[blockIdx.x] = s_data[0];
}