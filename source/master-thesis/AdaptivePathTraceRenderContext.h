#pragma once

#include "PathTraceRenderContext.h"
#include "TrackballCamera.h"
#include "Scenes.h"
#include <sutil.h>

#include "DebugHoelder.h"
#include "TestHoelder.h"

class AdaptivePathTraceContext : public PathTraceRenderContext
{
public:
	
	AdaptivePathTraceContext() : PathTraceRenderContext() {}
	~AdaptivePathTraceContext() {}

	void createAdaptiveContext(const char* const SAMPLE_NAME, const char* fileNameOptixPrograms, const char* fileNameAdaptiveOptixPrograms, const bool use_pbo, int rr_begin_depth)
	{
		createContext(SAMPLE_NAME, fileNameOptixPrograms, use_pbo, rr_begin_depth);

		//const char *adaptive_ptx = sutil::getPtxString(SAMPLE_NAME, fileNameAdaptiveOptixPrograms);

		Buffer hoelder_refinement_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["hoelder_refinement_buffer"]->set(hoelder_refinement_buffer);

#ifdef DEBUG_HOELDER

		// This buffer is for debug
		Buffer depth_gradient_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["depth_gradient_buffer"]->set(depth_gradient_buffer);

		// This buffer is for debug
		Buffer hoelder_alpha_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["hoelder_alpha_buffer"]->set(hoelder_alpha_buffer);

		// This buffer is for debug
		//Buffer total_sample_count_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		//_context["total_sample_count_buffer"]->set(total_sample_count_buffer);

#endif // DEBUG_HOELDER

		//Buffer hoelder_adaptive_scene_depth_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		//_context["hoelder_adaptive_scene_depth_buffer"]->set(hoelder_adaptive_scene_depth_buffer);

		Buffer object_ids_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_INT, _width, _height, use_pbo);
		_context["object_ids_buffer"]->set(object_ids_buffer);

		//Program adaptive_ray_gen_program = _context->createProgramFromPTXString(adaptive_ptx, "pathtrace_camera_adaptive");
		//_context->setRayGenerationProgram(1, adaptive_ray_gen_program);
	}

	void setWindowSize(unsigned int windowSize)
	{
		_windowSize = windowSize;
	}

	void setMaxAdditionalRaysTotal(unsigned int maxAdditionalRaysTotal)
	{
		_maxAdditionalRaysTotal = maxAdditionalRaysTotal;
	}

	void setMaxAdditionalRaysPerRenderRun(unsigned int maxAdditionalRaysPerRenderRun)
	{
		_maxAdditionalRaysPerRenderRun = maxAdditionalRaysPerRenderRun;
	}

	uint getMaxAdditionalRaysPerRenderRun() 
	{
		return _maxAdditionalRaysPerRenderRun;
	}

private:
	uint _windowSize; 
	uint _maxAdditionalRaysTotal;
	uint _maxAdditionalRaysPerRenderRun;
	
	uint _currentLevelAdaptiveSampleCount = 1;
	uint _currentAdaptiveLevel = 1;

private:

	void _setupPerRayTotalBudgetBuffer()
	{
		int* perPerRayBudget = new int[_width * _height * 4];

		// initialize additional rays buffer
		for (unsigned int i = 0; i < _width * _height; i++)
		{
			perPerRayBudget[i * 4] = static_cast<unsigned int>(_maxAdditionalRaysTotal);
			perPerRayBudget[i * 4 + 1] = static_cast<unsigned int>(_maxAdditionalRaysTotal);
			perPerRayBudget[i * 4 + 2] = static_cast<unsigned int>(_maxAdditionalRaysTotal);
			perPerRayBudget[i * 4 + 3] = static_cast<unsigned int>(_maxAdditionalRaysTotal);
		}

		// Additional rays test buffer setup
		Buffer additional_rays_buffer = sutil::createInputOutputBuffer(_context, RT_FORMAT_UNSIGNED_INT4, _width, _height, false);
		memcpy(additional_rays_buffer->map(), perPerRayBudget, sizeof(int) * _width * _height * 4);
		additional_rays_buffer->unmap();
		_context["additional_rays_buffer_input"]->set(additional_rays_buffer);

		delete[] perPerRayBudget;
	}

	void _setupPerRayWindowSizeBuffer()
	{
		int* perPerRayWindow_Size = new int[_width * _height * 4];

		// initialize additional rays buffer
		for (unsigned int i = 0; i < _width * _height; i++)
		{
			perPerRayWindow_Size[i * 4] = static_cast<unsigned int>(_windowSize);
			perPerRayWindow_Size[i * 4 + 1] = static_cast<unsigned int>(_windowSize);
			perPerRayWindow_Size[i * 4 + 2] = static_cast<unsigned int>(_windowSize);
			perPerRayWindow_Size[i * 4 + 3] = static_cast<unsigned int>(_windowSize);
		}

		// Additional rays test buffer setup
		Buffer additional_rays_buffer = sutil::createInputOutputBuffer(_context, RT_FORMAT_UNSIGNED_INT4, _width, _height, false);
		memcpy(additional_rays_buffer->map(), perPerRayWindow_Size, sizeof(int) * _width * _height * 4);
		additional_rays_buffer->unmap();
		_context["window_size_buffer"]->set(additional_rays_buffer);

		delete[] perPerRayWindow_Size;
	}
};

uint firstAdaptiveLevel = 6;
uint adaptiveLevels = maxAdaptiveLevel - firstAdaptiveLevel;

//CommandList commandListAdaptive;

// Variance based adaptive sampling specific
const uint windowSize = std::powf(2, adaptiveLevels);						// Powers of two are your friend.

const uint initialLevelWindowSize = std::powf(2, adaptiveLevels);
uint currentLevelWindowSize = initialLevelWindowSize;
												//const uint maxAdditionalRaysTotal = 50;
const uint maxAdditionalRaysTotal = 0;			// If using hoelder set this to zero. 
const uint maxAdditionalRaysPerRenderRun = 2;//std::powf(2, maxAdaptiveLevel);	// Powers of two are not only your friend, but a MUST here!
//float* perWindowVariance = nullptr;
//int* perPerRayBudget = nullptr;


uint currentLevelAdaptiveSampleCount = std::powf(2, firstAdaptiveLevel);
uint currentFrameSampleCount = min(currentLevelAdaptiveSampleCount, maxAdditionalRaysPerRenderRun);
uint currentAdaptiveLevel = firstAdaptiveLevel;

uint waitFramesNumber = currentLevelAdaptiveSampleCount / maxAdditionalRaysPerRenderRun;

int nextLevelBegin = 0;

void resetAdaptiveLevelVariables()
{
	currentLevelAdaptiveSampleCount = std::powf(2, firstAdaptiveLevel);
	//currentFrameSampleCount = 0;
	currentFrameSampleCount = min(currentLevelAdaptiveSampleCount, maxAdditionalRaysPerRenderRun);
	currentAdaptiveLevel = firstAdaptiveLevel;
	currentLevelWindowSize = initialLevelWindowSize;
	waitFramesNumber = currentLevelAdaptiveSampleCount / maxAdditionalRaysPerRenderRun;

	nextLevelBegin = 0;

#ifdef TEST_HOELDER
	currentTotalSampleCount = 0;
	equalQuantityComparisonDone = 0;
#endif // TEST_HOELDER

}

void setCurrentLevelWindowSize(Context context)
{
	context["current_level_window_size"]->setUint(currentLevelWindowSize);
}

void updateCurrentLevelAdaptiveVariables(Context context, bool cameraChanged)
{
	if (cameraChanged)
	{
		resetAdaptiveLevelVariables();
	}
	if (waitFramesNumber > 0)
	{
		waitFramesNumber--;
		currentFrameSampleCount = maxAdditionalRaysPerRenderRun;
	}
	context["wait_frames_number"]->setUint(waitFramesNumber);
	if (currentAdaptiveLevel < maxAdaptiveLevel && waitFramesNumber == 0)
	{
		currentAdaptiveLevel++;
		currentLevelAdaptiveSampleCount *= 2;
		currentLevelWindowSize /= 2;
		waitFramesNumber = currentLevelAdaptiveSampleCount / maxAdditionalRaysPerRenderRun;

		nextLevelBegin = 1;
		//printf("\n______________________________________________________________________________________________________\n\n");
		//printf("Current level adaptive sample count (host): %u\n", currentLevelAdaptiveSampleCount);
		//printf("\n______________________________________________________________________________________________________\n\n");
	}
	//printf("\n======================================================================================================\n");
	//printf("Current adaptive level (host):				%u\n", currentAdaptiveLevel);
	//printf("Current level adaptive sample count (host):		%u\n", currentLevelAdaptiveSampleCount);
	//printf("Current frame sample count (host):			%u\n", currentFrameSampleCount);
	//printf("Current wait frames number (host):			%u\n", waitFramesNumber);
	//printf("Current level window size (host):			%u\n", currentLevelWindowSize);
	//printf("Next level begin (host):				%u\n", nextLevelBegin);
	//printf("\n======================================================================================================\n");
	context["num_samples"]->setUint(currentFrameSampleCount);
	context["current_level_adaptive_sample_count"]->setUint(currentLevelAdaptiveSampleCount);
	context["current_level_window_size"]->setUint(currentLevelWindowSize);

	context["next_level_begin"]->setInt(nextLevelBegin);

	currentFrameSampleCount = 0;
	nextLevelBegin = 0;

#ifdef TEST_HOELDER
	if (EQUAL_QUANTITY_COMPARISON_ACTIVE && !equalQuantityComparisonDone)
	{
		currentTotalSampleCount += maxAdditionalRaysPerRenderRun;
	}
#endif // TEST_HOELDER

}

int getInitialRenderNumSamples()
{
	int numSamples = 0;

	for (int i = 0; i <= firstAdaptiveLevel; i++)
	{
		numSamples += pow(2,i);
	}

	return numSamples;
}