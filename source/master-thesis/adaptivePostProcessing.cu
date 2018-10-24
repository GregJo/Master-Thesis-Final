/*
* Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <optixu/optixu_math_namespace.h>
#include "optixPathTracer.h"

#include "random.h"
#include "Buffers.h"
#include "AdaptiveUtil.h"
#include "LevelHoelderAdaptive.h"

using namespace optix;

struct PerRayData_pathtrace
{
	float3 result;
	float3 radiance;
	float3 attenuation;
	float3 origin;
	float3 direction;
	unsigned int seed;
	int depth;
	int countEmitted;
	int done;
	int missed;
	//int isAdaptive;
};

struct PerRayData_pathtrace_shadow
{
	bool inShadow;
};

// Scene wide variables
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(float, far_plane, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );

rtDeclareVariable(unsigned int, current_level_window_size, , );

RT_PROGRAM void pathtrace_camera_adaptive()
{
	uint2 current_window_centre = get_current_window_centre(launch_index, current_level_window_size);

	output_buffer[launch_index] = output_buffer[current_window_centre];
	output_scene_depth_buffer[launch_index] = output_scene_depth_buffer[current_window_centre];
	output_filter_sum_buffer[launch_index] = output_filter_sum_buffer[current_window_centre];
	output_filter_x_sample_sum_buffer[launch_index] = output_filter_x_sample_sum_buffer[current_window_centre];

	output_current_total_rays_buffer[launch_index] = output_current_total_rays_buffer[current_window_centre];

#ifdef DEBUG_HOELDER
	depth_gradient_buffer[launch_index] = depth_gradient_buffer[current_window_centre];
#endif //DEBUG_HOELDER

	object_ids_buffer[launch_index] = object_ids_buffer[current_window_centre];
}
