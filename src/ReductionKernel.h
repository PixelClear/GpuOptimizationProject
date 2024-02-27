
using u32 = unsigned int;

__global__ void reduce0(int* g_idata, int* g_odata, int n) 
{
	u32 gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gIdx >= n) return;

	g_odata[0] = g_idata[0];
	for (u32 tId = 1; tId < n; tId++)
	{
		g_odata[tId] += g_idata[tId];
	}
}