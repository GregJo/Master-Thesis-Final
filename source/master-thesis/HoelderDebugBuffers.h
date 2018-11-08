//#include "DebugHoelder.h"
#include "TestHoelder.h"

#ifdef DEBUG_HOELDER

#include <optix_device.h>

using namespace optix;

// For debug!
rtBuffer<float4, 2>   depth_gradient_buffer;
// For debug!
rtBuffer<float4, 2>   hoelder_alpha_buffer;
// For debug!
rtBuffer<uint4, 2>   total_sample_count_buffer;

//rtBuffer<float4, 2>   adaptive_sample_count_buffer;

#endif // !PATH_TRACE_BUFFERS