#include <optixu/optixu_math_namespace.h>
#include <optix_device.h>
#include "AdaptiveUtil.h"

using namespace optix;

rtBuffer<float, 2>				mitchell_filter_table;
rtDeclareVariable(int, mitchell_filter_table_width, , );
rtDeclareVariable(float2, mitchell_filter_radius, , );
rtDeclareVariable(float2, mitchell_filter_inv_radius, , );
rtDeclareVariable(int, expected_samples_count, , );

static __device__ float computeMitchellFilterSampleContribution(float2 sample, uint2 current_launch_index, float2 scaled_mitchell_filter_radius)
{
	float2 center_pixel_offset = make_float2(0.5f);

	float2 center_pixel = make_float2(current_launch_index.x + center_pixel_offset.x, current_launch_index.y + center_pixel_offset.y);

	float2 center_sample = make_float2(sample.x + center_pixel_offset.x, sample.y + center_pixel_offset.y);

	float2 dist = center_sample - center_pixel;

	float2 normalized_dist = make_float2((dist.x / (scaled_mitchell_filter_radius.x)), (dist.y / (scaled_mitchell_filter_radius.y)));

	int mitchell_filter_half_table_width = 0.5f * mitchell_filter_table_width - 1;

	int2 filter_table_center = make_int2(mitchell_filter_half_table_width + 1);

	int filter_table_width_offset_x = (static_cast<int>(normalized_dist.x * mitchell_filter_half_table_width));
	int filter_table_width_offset_y = (static_cast<int>(normalized_dist.y * mitchell_filter_half_table_width));
	int2 filter_table_width_offset = make_int2(filter_table_width_offset_x, filter_table_width_offset_y);

	// max value should be 0.5f
	float normalized_dist_length = length(normalized_dist);

	float mitchell_filter_weight = 0.0f;

	if (normalized_dist_length <= 1.0f)
	{
		mitchell_filter_weight = mitchell_filter_table[make_uint2((filter_table_center.x + filter_table_width_offset.x), (filter_table_center.y + filter_table_width_offset.y))];
	}
	return mitchell_filter_weight;
};

static __device__ void computeMitchellFilterSampleContributionInNeighborhood(float2 sample, uint2 current_launch_index, float3 prd_result, size_t2 screen, int current_total_samples_buffer,
																				buffer<float4, 2>* filter_sum_buffer,
																				buffer<float4, 2>* filter_x_sample_sum_buffer)
{
	int radiusX = mitchell_filter_radius.x;
	int radiusY = mitchell_filter_radius.y;

	float2 scaled_radius;

	scaled_radius.x = radiusX;
	scaled_radius.y = radiusY;

	uint2 current_launch_index_top_left = make_uint2(current_launch_index.x - static_cast<uint>(scaled_radius.x), current_launch_index.y - static_cast<uint>(scaled_radius.y));

	float3 result = make_float3(0.0f);

	for (int x = 0; x < scaled_radius.x * 2 + 1; x++)
	{
		for (int y = 0; y < scaled_radius.y * 2 + 1; y++)
		{
			uint2 neighborhood_idx = make_uint2(current_launch_index_top_left.x + x, current_launch_index_top_left.y + y);

			if (neighborhood_idx.x < screen.x && neighborhood_idx.y < screen.y/* || neighborhood_idx.x < 0 || neighborhood_idx.y < 0*/)
			{
				float reconstruction_filter_weight = computeMitchellFilterSampleContribution(sample, neighborhood_idx, scaled_radius);

				atomicExch(&(*filter_sum_buffer)[neighborhood_idx].x, (*filter_sum_buffer)[neighborhood_idx].x + reconstruction_filter_weight);

				atomicExch(&(*filter_x_sample_sum_buffer)[neighborhood_idx].x, (*filter_x_sample_sum_buffer)[neighborhood_idx].x + (reconstruction_filter_weight * prd_result.x));
				atomicExch(&(*filter_x_sample_sum_buffer)[neighborhood_idx].y, (*filter_x_sample_sum_buffer)[neighborhood_idx].y + (reconstruction_filter_weight * prd_result.y));
				atomicExch(&(*filter_x_sample_sum_buffer)[neighborhood_idx].z, (*filter_x_sample_sum_buffer)[neighborhood_idx].z + (reconstruction_filter_weight * prd_result.z));
				atomicExch(&(*filter_x_sample_sum_buffer)[neighborhood_idx].w, 1.0f);
			}
		}
	}
}

static __device__ float computeLevelMitchellFilterSampleContribution(uint current_level_window_size, float2 sample, uint2 current_launch_index, float2 scaled_mitchell_filter_radius)
{
	float2 center_pixel_offset = make_float2(0.5f);

	float2 center_pixel = make_float2(current_launch_index.x + center_pixel_offset.x, current_launch_index.y + center_pixel_offset.y);

	float2 center_sample = make_float2(sample.x + center_pixel_offset.x, sample.y + center_pixel_offset.y);

	float2 dist = center_sample - center_pixel;

	float2 normalized_dist = make_float2((dist.x / (scaled_mitchell_filter_radius.x)), (dist.y / (scaled_mitchell_filter_radius.y)));

	int mitchell_filter_half_table_width = 0.5f * mitchell_filter_table_width - 1;

	int2 filter_table_center = make_int2(mitchell_filter_half_table_width + 1);

	int filter_table_width_offset_x = (static_cast<int>(normalized_dist.x * mitchell_filter_half_table_width));
	int filter_table_width_offset_y = (static_cast<int>(normalized_dist.y * mitchell_filter_half_table_width));
	int2 filter_table_width_offset = make_int2(filter_table_width_offset_x, filter_table_width_offset_y);

	// max value should be 0.5f
	float normalized_dist_length = length(normalized_dist);

	float mitchell_filter_weight = 0.0f;

	if (normalized_dist_length * current_level_window_size <= current_level_window_size)
	{
		mitchell_filter_weight = mitchell_filter_table[make_uint2((filter_table_center.x + filter_table_width_offset.x), (filter_table_center.y + filter_table_width_offset.y))];
	}

	return mitchell_filter_weight;
};

static __device__ void computeLevelMitchellFilterSampleContributionInNeighborhood(uint current_level_window_size, float2 sample, uint2 current_launch_index, float3 prd_result, size_t2 screen, int current_total_samples_buffer,
	buffer<float4, 2>* filter_sum_buffer,
	buffer<float4, 2>* filter_x_sample_sum_buffer)
{
	int radiusX = mitchell_filter_radius.x;
	int radiusY = mitchell_filter_radius.y;

	float2 scaled_radius;

	scaled_radius.x = radiusX;
	scaled_radius.y = radiusY;

	uint2 current_launch_index_top_left = make_uint2(current_launch_index.x - static_cast<uint>(scaled_radius.x) * current_level_window_size, current_launch_index.y - static_cast<uint>(scaled_radius.y) * current_level_window_size);

	uint2 current_top_left_window_centre = get_current_window_centre(current_launch_index_top_left, current_level_window_size);

	float3 result = make_float3(0.0f);

	for (int x = 0; x < scaled_radius.x * 2 + 1; x++)
	{
		for (int y = 0; y < scaled_radius.y * 2 + 1; y++)
		{
			uint2 neighborhood_idx = make_uint2(current_top_left_window_centre.x + x * current_level_window_size, current_top_left_window_centre.y + y * current_level_window_size);

			uint2 current_window_centre = get_current_window_centre(neighborhood_idx, current_level_window_size);

			//if (current_launch_index.x == 255 && current_launch_index.y == 255)
			//{
			//	rtPrintf("Neighborhood index: [ %u , %u ], Current window centre: [ %u , %u ]\n", neighborhood_idx.x, neighborhood_idx.y, current_window_centre.x, current_window_centre.y);
			//}

			if (neighborhood_idx.x < screen.x && neighborhood_idx.y < screen.y/* || neighborhood_idx.x < 0 || neighborhood_idx.y < 0*/)
			{
				/*float reconstruction_filter_weight = computeLevelMitchellFilterSampleContribution(current_level_window_size, sample, neighborhood_idx, scaled_radius);*/
				float reconstruction_filter_weight = computeLevelMitchellFilterSampleContribution(current_level_window_size, make_float2(current_window_centre.x, current_window_centre.y), neighborhood_idx, scaled_radius);

				atomicExch(&(*filter_sum_buffer)[neighborhood_idx].x, (*filter_sum_buffer)[neighborhood_idx].x + reconstruction_filter_weight);

				atomicExch(&(*filter_x_sample_sum_buffer)[neighborhood_idx].x, (*filter_x_sample_sum_buffer)[neighborhood_idx].x + (reconstruction_filter_weight * prd_result.x));
				atomicExch(&(*filter_x_sample_sum_buffer)[neighborhood_idx].y, (*filter_x_sample_sum_buffer)[neighborhood_idx].y + (reconstruction_filter_weight * prd_result.y));
				atomicExch(&(*filter_x_sample_sum_buffer)[neighborhood_idx].z, (*filter_x_sample_sum_buffer)[neighborhood_idx].z + (reconstruction_filter_weight * prd_result.z));
				atomicExch(&(*filter_x_sample_sum_buffer)[neighborhood_idx].w, 1.0f);
			}
		}
	}
}

// Evaluate pixel filtering equation.
static __device__ void evaluatePixelFileringEquation(uint2 current_launch_index, buffer<float4, 2>* output_render_buffer, buffer<float4, 2>* filter_sum_buffer, buffer<float4, 2>* filter_x_sample_sum_buffer)
{
	if (current_launch_index.x < output_render_buffer->size().x && current_launch_index.y < output_render_buffer->size().y)
	{
		(*output_render_buffer)[current_launch_index].x = (*filter_x_sample_sum_buffer)[current_launch_index].x / (*filter_sum_buffer)[current_launch_index].x;
		(*output_render_buffer)[current_launch_index].y = (*filter_x_sample_sum_buffer)[current_launch_index].y / (*filter_sum_buffer)[current_launch_index].x;
		(*output_render_buffer)[current_launch_index].z = (*filter_x_sample_sum_buffer)[current_launch_index].z / (*filter_sum_buffer)[current_launch_index].x;
	}
};