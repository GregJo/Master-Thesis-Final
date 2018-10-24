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

//#include "VarianceAdaptive.h"
#include "HoelderAdaptive.h"
#include "MitchellFilterDevice.h"

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



//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(unsigned int, frame_number, , );
rtDeclareVariable(unsigned int, num_samples, , );
rtDeclareVariable(unsigned int, sqrt_num_samples, , );
rtDeclareVariable(unsigned int, rr_begin_depth, , );
rtDeclareVariable(unsigned int, pathtrace_ray_type, , );

RT_PROGRAM void pathtrace_camera_adaptive()
{

	// Debug!
	depth_gradient_buffer[launch_index] = make_float4(0.0f);
	size_t2 screen = output_buffer.size();

	float2 inv_screen = 1.0f / make_float2(screen) * 2.f;
	float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

	float2 jitter_scale = inv_screen / sqrt_num_samples;

	initializeHoelderAdaptiveSceneDepthBuffer(launch_index, frame_number, camera_changed);
	initialize_hoelder_adaptive_buffers(launch_index, frame_number, camera_changed, static_cast<uint>(window_size));

	unsigned int adaptive_samples_per_pixel = compute_current_samples_number(launch_index, window_size_buffer[launch_index].x);
	//unsigned int adaptive_samples_per_pixel = 0;
	unsigned int current_samples_per_pixel = adaptive_samples_per_pixel;
	float3 result = make_float3(0.0f);

	unsigned int adaptive_sqrt_num_samples = sqrtf(static_cast<float>(adaptive_samples_per_pixel));

	if (!adaptive_sqrt_num_samples)
	{
		++adaptive_sqrt_num_samples;
	}

	unsigned int seed = tea<16>(screen.x*launch_index.y + launch_index.x, frame_number);

	int num_not_missed_rays = current_samples_per_pixel;

	if (current_samples_per_pixel)
	{
		//post_process_output_buffer[launch_index] = hoelder_adaptive_buffers[input_scene_render_buffer][launch_index];
		do
		{
			//
			// Sample pixel using jittering
			//
			unsigned int x = adaptive_samples_per_pixel % adaptive_sqrt_num_samples;
			unsigned int y = adaptive_samples_per_pixel / adaptive_sqrt_num_samples;
			float2 jitter = make_float2(x - rnd(seed), y - rnd(seed));
			float2 d = pixel + jitter*jitter_scale;
			float3 ray_origin = eye;
			float3 ray_direction = normalize(d.x*U + d.y*V + W);

			//rtPrintf("2D ray direction in screen coordinates: [ %f , %f ]\n", d.x, d.y);

			// Initialze per-ray data
			PerRayData_pathtrace prd;
			prd.result = make_float3(0.f);
			prd.attenuation = make_float3(1.f);
			prd.countEmitted = true;
			prd.done = false;
			prd.seed = seed;
			prd.depth = 0;
			//prd.isAdaptive = 1;
			prd.missed = 0;

			// Each iteration is a segment of the ray path.  The closest hit will
			// return new segments to be traced here.
			for (;;)
			{
				if (prd.depth == 1)
				{
					float ray_length = fabsf(length((prd.origin - eye)));
					float normalized_ray_length = ray_length / far_plane;//2500.0f;

					float a = 1.0f / (float)output_current_total_rays_buffer[launch_index].x;
					float3 old_depth = make_float3(output_scene_depth_buffer[launch_index]);
					output_scene_depth_buffer[launch_index] = make_float4(make_float3(min(normalized_ray_length, old_depth.x)), 1.0f);// make_float4(lerp(old_depth, make_float3(normalized_ray_length), a), 1.0f);

					hoelder_adaptive_scene_depth_buffer[launch_index] = make_float4(make_float3(min(normalized_ray_length, old_depth.x)), 1.0f);//make_float4(lerp(old_depth, make_float3(normalized_ray_length), a), 1.0f);

				}

				Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);
				rtTrace(top_object, ray, prd);

				if (prd.done)
				{
					if (prd.missed && prd.depth == 0)
					{
						num_not_missed_rays--;
					}
					// We have hit the background or a luminaire
					prd.result += prd.radiance * prd.attenuation;
					break;
				}

				// Russian roulette termination 
				if (prd.depth >= rr_begin_depth)
				{
					float pcont = fmaxf(prd.attenuation);
					if (rnd(prd.seed) >= pcont)
						break;
					prd.attenuation /= pcont;
				}

				prd.depth++;
				prd.result += prd.radiance * prd.attenuation;

				// Update ray data for the next path segment
				ray_origin = prd.origin;
				ray_direction = prd.direction;
			}

			float2 sample = make_float2(launch_index.x + jitter.x, launch_index.y + jitter.y);

			seed = prd.seed;
			output_current_total_rays_buffer[launch_index].x++;
			int current_total_rays = output_current_total_rays_buffer[launch_index].x;
			computeMitchellFilterSampleContributionInNeighborhood(sample, launch_index, prd.result, screen, current_total_rays, &output_filter_sum_buffer, &output_filter_x_sample_sum_buffer);
		} while (--current_samples_per_pixel);
	}
	//
	// Update the output buffer
	//

	if (num_not_missed_rays > 0)
	{
		evaluatePixelFileringEquation(launch_index, &output_buffer, &output_filter_sum_buffer, &output_filter_x_sample_sum_buffer);
	}
}

//
// Adaptive version of pathtracing end
//
