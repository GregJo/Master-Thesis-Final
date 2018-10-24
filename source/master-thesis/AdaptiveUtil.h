#ifndef ADAPTIVE_UTIL

#define ADAPTIVE_UTIL

#include <optixu/optixu_math_namespace.h>
#include <optix_device.h>

#include "Buffers.h"

static __device__ uint2 get_current_window_centre(uint2 current_launch_index, uint current_window_size)
{
	size_t2 screen = output_buffer.size();

	uint times_width = screen.x / current_window_size;
	uint times_height = screen.y / current_window_size;

	uint horizontal_padding = static_cast<uint>(0.5f * (screen.x - (times_width * current_window_size)));
	uint vertical_padding = static_cast<uint>(0.5f * (screen.y - (times_height * current_window_size)));

	uint half_window_size = (current_window_size / 2);// +(current_window_size % 2);

	uint2 times_launch_index = make_uint2(((current_launch_index.x / current_window_size) * current_window_size) % screen.x, ((current_launch_index.y / current_window_size) * current_window_size) % screen.y);

	uint2 current_window_center = make_uint2(times_launch_index.x + horizontal_padding + half_window_size, times_launch_index.y + vertical_padding + half_window_size);

	return current_window_center;
};

#endif // !ADAPTIVE_UTIL