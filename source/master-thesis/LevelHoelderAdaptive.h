#include <optixu/optixu_math_namespace.h>
#include <optix_device.h>

#include "Buffers.h"
#include "HoelderDebugBuffers.h"
#include "AdaptiveUtil.h"

using namespace optix;

rtDeclareVariable(unsigned int, window_size, , );
rtDeclareVariable(unsigned int, current_level_adaptive_sample_count, , );
rtDeclareVariable(unsigned int, max_per_frame_samples_budget, , ) = static_cast<uint>(4u);		/* this variable can be written by the user */
rtDeclareVariable(int, camera_changed, , );

rtBuffer<float4, 2>	  hoelder_refinement_buffer;

rtBuffer<int, 2>	  object_ids_buffer;

static __device__ __inline__ float4 compute_level_color_depth_gradient(uint2 current_launch_index, uint window_size_)
{
	uint2 screen = make_uint2(output_buffer.size().x, output_buffer.size().y);

	int up = min((static_cast<int>(current_launch_index.y) + static_cast<int>(window_size_)), static_cast<int>(screen.y));
	int down = max(0, (static_cast<int>(current_launch_index.y) - static_cast<int>(window_size_)));
	int left = max(0, (static_cast<int>(current_launch_index.x) - static_cast<int>(window_size_)));
	int right = min((static_cast<int>(current_launch_index.x) + static_cast<int>(window_size_)), static_cast<int>(screen.x));

	uint2 idx_up = make_uint2(current_launch_index.x, static_cast<uint>(up));
	idx_up = get_current_window_centre(idx_up, window_size_);
	uint2 idx_down = make_uint2(current_launch_index.x, static_cast<uint>(down));
	idx_down = get_current_window_centre(idx_down, window_size_);
	uint2 idx_left = make_uint2(static_cast<uint>(left), current_launch_index.y);
	idx_left = get_current_window_centre(idx_left, window_size_);
	uint2 idx_right = make_uint2(static_cast<uint>(right), current_launch_index.y);
	idx_right = get_current_window_centre(idx_right, window_size_);

	float4 gradient_color_y = output_buffer[idx_up] - output_buffer[idx_down];
	gradient_color_y = 0.5f * /*window_inverse * */gradient_color_y;
	float4 gradient_color_x = output_buffer[idx_right] - output_buffer[idx_left];
	gradient_color_x = 0.5f * /*window_inverse * */gradient_color_x;

	float4 gradient_color_tmp = 0.5f * gradient_color_y + gradient_color_x;

	float3 gradient_color = 0.5f * make_float3( gradient_color_tmp.x, gradient_color_tmp.y, gradient_color_tmp.z);

	float gradient_depth_x = (output_scene_depth_buffer[idx_up].x) - (output_scene_depth_buffer[idx_down].x);
	gradient_depth_x = 0.5f * /*window_inverse * */gradient_depth_x;
	float gradient_depth_y = (output_scene_depth_buffer[idx_right].x) - (output_scene_depth_buffer[idx_left].x);
	gradient_depth_y = 0.5f * /*window_inverse * */gradient_depth_y;

	float gradient_depth = 0.5f * gradient_depth_x + gradient_depth_y;

	//gradient_color = window_size_ * gradient_color;

	float4 combined_gradient = /*50.0f * */make_float4(gradient_color.x, gradient_color.y, gradient_color.z, gradient_depth);

	return combined_gradient;
};

static __device__ float compute_level_window_hoelder(uint2 center, uint window_size_)
{
	uint2 screen = make_uint2(output_buffer.size().x, output_buffer.size().y);

	float alpha = 100.f;

	uint squared_window_size = window_size_ * window_size_;
	uint half_window_size = (window_size_ / 2) + (window_size_ % 2);
	uint2 top_left_window_corner = make_uint2(center.x - half_window_size, center.y - half_window_size);

	float3 center_buffer_val = make_float3(output_buffer[center].x, output_buffer[center].y, output_buffer[center].z);

	float centerColorMean = 1.f / 3.f * (center_buffer_val.x + center_buffer_val.y + center_buffer_val.z);
	float neighborColorMean = 0.0f;

	float4 color_depth_gradient = compute_level_color_depth_gradient(center, window_size_); //compute_color_depth_gradient(center);//

#ifdef DEBUG_HOELDER

	// Debug!
	depth_gradient_buffer[center] = make_float4(color_depth_gradient.w);

#endif // DEBUG_HOELDER

	for (size_t i = 0; i < 12; i++)
	{
		int idx_x = 0;
		int idx_y = 0;

		if (i == 0)
		{
			idx_x = 0;
			idx_y = center.y + window_size_ > screen.y ? 0 : 1;
		}
		if (i == 1)
		{
			idx_x = 0;
			idx_y = static_cast<int>(center.y) - static_cast<int>(window_size_) < 0 ? 0 : -1;
		}
		if (i == 2)
		{
			idx_x = static_cast<int>(center.x) - static_cast<int>(window_size_) < 0 ? 0 : -1;
			idx_y = 0;
		}
		if (i == 3)
		{
			idx_x = center.x + window_size_ > screen.x ? 0 : 1;
			idx_y = 0;
		}

		if (i == 4)
		{
			idx_x = static_cast<int>(center.x) - static_cast<int>(window_size_) < 0 ? 0 : -1;
			idx_y = center.y + window_size_ > screen.y ? 0 : 1;
		}
		if (i == 5)
		{
			idx_x = center.x + window_size_ > screen.x ? 0 : 1;
			idx_y = center.y + window_size_ > screen.y ? 0 : 1;
		}
		if (i == 6)
		{
			idx_x = static_cast<int>(center.x) - static_cast<int>(window_size_) < 0 ? 0 : -1;
			idx_y = static_cast<int>(center.y) - static_cast<int>(window_size_) < 0 ? 0 : -1;
		}
		if (i == 7)
		{
			idx_x = center.x + window_size_ > screen.x ? 0 : 1;
			idx_y = static_cast<int>(center.y) - static_cast<int>(window_size_) < 0 ? 0 : -1;
		}

		if (i == 8)
		{
			idx_x = 0;
			idx_y = center.y + 2 * window_size_ > screen.y ? 0 : 2;
		}
		if (i == 9)
		{
			idx_x = 0;
			idx_y = static_cast<int>(center.y) - 2 * static_cast<int>(window_size_) < 0 ? 0 : -2;
		}
		if (i == 10)
		{
			idx_x = static_cast<int>(center.x) - 2 * static_cast<int>(window_size_) < 0 ? 0 : -2;
			idx_y = 0;
		}
		if (i == 11)
		{
			idx_x = center.x + 2 * window_size_ > screen.x ? 0 : 2;
			idx_y = 0;
		}

		if (idx_x == 0 && idx_y == 0)
		{
			continue;
		}

		uint2 idx = make_uint2(clamp(static_cast<int>(center.x + idx_x * window_size_), int(0), static_cast<int>(screen.x)), clamp(static_cast<int>(center.y + idx_y * window_size_), int(0), static_cast<int>(screen.y)));

		idx = get_current_window_centre(idx, window_size_);

		int center_obj_id = object_ids_buffer[center];
		int neighbor_obj_id = object_ids_buffer[idx];

		if (center_obj_id - neighbor_obj_id != 0)
		{
			alpha = 0.0f;
			//alpha = 50.0f;
			break;
		}

		float neighbor_center_distance = (1.0f / static_cast<float>(window_size_)) * length(/*(1.0f / static_cast<float>(window_size_)) * */(make_float2(idx) - make_float2(center)));

		//float c_base = 1.0f/(sqrtf(2.0f) + 0.1f);
		//float c_base = 1.0f / (sqrtf(2.0f) + 0.5f);
		float c_base = 1.0f / (2.0f * 100.1f);
		//float c_base = 1.0f / (2.0f * window_size_);
		float log_base = log(c_base * fabsf(neighbor_center_distance)/* + 1.0f*/)/**2.0f*/;
		log_base = log_base == 0 ? /*1.0f*/-0.001f : log_base;

		float3 neighbor_buffer_val = make_float3(output_buffer[idx].x, output_buffer[idx].y, output_buffer[idx].z);
		neighborColorMean = length(neighbor_buffer_val);
		float log_x = 0.0f;

		// Decide whether to use smooth or non-smooth regime based on depth/geometry buffer map. 
		// Where there is a very small depth/geometry gradient use smooth regime computation hoelder alpha, 
		// else use non-smooth regime hoelder alpha computation (log_x value makes for that distinction).
		// Smooth regime
		//float c = 10.0f;
		float c = 3.0f;
		//c = 1.0f;// / c;
		if (fabsf(color_depth_gradient.w)/* Value 'w' is depth/geometry gradient */ <= 0.025f/**50.0f*//*0.025f*//* Currently more or less arbitary threshhold for an edge! */)
		{
			//hoelder_alpha_buffer[center] = make_float4(1000.0f, 0.0f, 1000.0f, 1.0f);
			float3 color_gradient = make_float3(color_depth_gradient.x, color_depth_gradient.y, color_depth_gradient.z);
			float mean_of_color_gradient = 1.f / 3.f * (color_gradient.x + color_gradient.y + color_gradient.z);
			log_x = log(/*c hoelder constant, also try value 3 * */fabsf(c * (neighborColorMean - centerColorMean - mean_of_color_gradient * neighbor_center_distance))/* + 1.0f*/);
		}
		// Non-smooth regime
		else
		{
			log_x = log(/*c hoelder constant, also try value 3 * */fabsf(c * (neighborColorMean - centerColorMean)) + 1.0f);
			//output_buffer[center] = make_float4(1000.0f,0.0f,1000.0f,1.0f);
		}

		//if (center.x == (screen.x / 2.0f))
		//{
		//	rtPrintf("log_x(device): %f, current i(device): %d\n", log_x, static_cast<int>(i));
		//	//rtPrintf("log_base(device): %f, current i(device): %d\n", log_base, static_cast<int>(i));
		//	//rtPrintf("log(base)(device): log(%f), current i(device): %d\n", (c_base * fabsf(neighbor_center_distance)), static_cast<int>(i));
		//	//rtPrintf("log_x(device): %f, log_base(device): %f, log_x / log_base(device): %f, current center (device): [ %u , %u ], current i(device): %d\n", log_x, log_base, log_x / log_base, center.x, center.y, static_cast<int>(i));
		//}

		alpha = min(alpha, log_x / log_base);
		alpha = clamp(alpha, 0.0f, 100.f);
	}

	return alpha;
};

static __device__ void hoelder_compute_current_level_samples_count(uint2 current_launch_index, uint2 current_window_center, uint window_size_)
{
	if (window_size_ >= 1)
	{
		float hoelder_alpha = -100.0f;
		float hoelder_alpha_no_refinement_threshhold = 0.2f;//0.05f

#ifdef DEBUG_HOELDER
		hoelder_alpha_buffer[current_window_center] = make_float4(0.0f);
		hoelder_alpha_buffer[current_launch_index] = make_float4(0.0f);
#endif //DEBUG_HOELDER

		if (hoelder_refinement_buffer[current_window_center].x == 1)
		{
			if (output_scene_depth_buffer[current_launch_index].x < 1.0f)
			{
				hoelder_alpha = compute_level_window_hoelder(current_window_center, window_size_);//compute_window_hoelder(current_window_center, window_size_);
																									//hoelder_alpha = compute_window_hoelder(current_window_center, window_size_);
			}
			hoelder_refinement_buffer[current_window_center] = make_float4(0);
			hoelder_refinement_buffer[current_launch_index] = make_float4(0);
		}

		float currentPixelLuminosity = 0.21f * output_buffer[current_launch_index].x + 0.72f * output_buffer[current_launch_index].y + 0.07f * output_buffer[current_launch_index].z;


#ifdef DEBUG_HOELDER
		hoelder_alpha_buffer[current_window_center] = make_float4(hoelder_alpha * 1.0f);// -0.5f*(1.0f - currentPixelLuminosity));
		hoelder_alpha_buffer[current_launch_index] = make_float4(hoelder_alpha * 1.0f);// -0.5f*(1.0f - currentPixelLuminosity));
		//hoelder_alpha_buffer[current_window_center] = make_float4(1.0f - hoelder_alpha * 1000.0f);
		//hoelder_alpha_buffer[current_launch_index] = make_float4(1.0f - hoelder_alpha * 1000.0f);
#endif //DEBUG_HOELDER

		if (hoelder_alpha <= -100.0f)
		{
			hoelder_alpha = 100.0f;
		}

		//hoelder_alpha_buffer[current_launch_index] = make_float4(1.0 - hoelder_alpha * 1.0f);

		// Luminosity conversion: 0.21 R + 0.72 G + 0.07 B
		//float currentPixelLuminosity = 0.21f * output_buffer[current_launch_index].x + 0.72f * output_buffer[current_launch_index].y + 0.07f * output_buffer[current_launch_index].z;

		if (/*hoelder_alpha * 1.0f*/hoelder_alpha * 1.0f/* * (1.0f - currentPixelLuminosity)*/ < hoelder_alpha_no_refinement_threshhold)// && currentPixelLuminosity > 0.0f)// * currentPixelLuminosity)
		{
			hoelder_refinement_buffer[current_window_center] = make_float4(1);
			hoelder_refinement_buffer[current_launch_index] = make_float4(1);
		}
	}
};

static __device__ __inline__ int get_do_refine(uint2 current_launch_index)
{
	return hoelder_refinement_buffer[current_launch_index].x;
}

static __device__ __inline__ void initialize_hoelder_adaptive_buffers(uint2 current_launch_index)
{
	hoelder_refinement_buffer[current_launch_index] = make_float4(1);
	//total_sample_count_buffer[current_launch_index] = make_float4(0.0f);
};