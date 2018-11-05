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

/////////////////////////////////////////////////////////////////////////////

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#    include <GL/freeglut.h>
#  else
#    include <GL/glut.h>
#  endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include "optixPathTracer.h"
#include <sutil.h>
#include <Arcball.h>

#include <OptiXMesh.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdint.h>
#include <iomanip>

#include "TrackballCamera.h"

#include "MitchellFilter.h"

#include "AdaptivePathTraceRenderContext.h"

#include "Scenes.h"

#include "TestHoelder.h"


using namespace optix;

const char* const SAMPLE_NAME = "master-thesis";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

//AdaptivePathTraceContext* adaptive_PT_context;
Context        context = 0;
uint32_t       width = 512;
uint32_t       height = 512;
bool           use_pbo = true;

int            frame_number = 1;
int            sqrt_num_samples = sqrt(currentLevelAdaptiveSampleCount);
int            num_samples = getInitialRenderNumSamples();
int            rr_begin_depth = 1;
Program        pgram_intersection = 0;
Program        pgram_bounding_box = 0;

// Camera state
TrackballCamera* camera;
sutil::Arcball arcball;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

bool useAdaptivePostProcessing = false;

//------------------------------------------------------------------------------
//
// Forward decls 
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void glutInitialize(int* argc, char** argv);
void glutRun();

void glutDisplay();
void glutKeyboardPress(unsigned char k, int x, int y);
void glutMousePress(int button, int state, int x, int y);
void glutMouseMotion(int x, int y);
void glutResize(int w, int h);

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer()
{
	return context["output_buffer"]->getBuffer();
}

Buffer getPerWindowVarianceBuffer() 
{
	return context["per_window_variance_buffer_input"]->getBuffer();
}

Buffer getPerRayBudgetBuffer()
{
	return context["additional_rays_buffer_input"]->getBuffer();
}

Buffer getPerRayWindowSizeBuffer()
{
	return context["window_size_buffer"]->getBuffer();
}

Buffer getOutputDepthBuffer() 
{
	return context["output_scene_depth_buffer"]->getBuffer();
}

// Debug
Buffer getDepthGradientBuffer()
{
	return context["depth_gradient_buffer"]->getBuffer();
}

// Debug
Buffer getHoelderAlphaBuffer()
{
	return context["hoelder_alpha_buffer"]->getBuffer();
}

// Debug
Buffer getHoelderAdaptiveSceneDepthBuffer() 
{
	return context["hoelder_adaptive_scene_depth_buffer"]->getBuffer();
}

// Debug
Buffer getHoelderRefinementBuffer()
{
	return context["hoelder_refinement_buffer"]->getBuffer();
}

// Debug
Buffer getTotalSampleCountBuffer()
{
	return context["total_sample_count_buffer"]->getBuffer();
}

// Debug
Buffer getPerWindowVarianceBufferOutput()
{
	return context["per_window_variance_buffer_output"]->getBuffer();
}

void destroyContext()
{
	if (context)
	{
		context->destroy();
		context = 0;
	}
}


void registerExitHandler()
{
	// register shutdown handler
#ifdef _WIN32
	glutCloseFunc(destroyContext);  // this function is freeglut-only
#else
	atexit(destroyContext);
#endif
}


void setMaterial(
	GeometryInstance& gi,
	Material material,
	const std::string& color_name,
	const float3& color)
{
	gi->addMaterial(material);
	gi[color_name]->setFloat(color);
}


GeometryInstance createParallelogram(
	Context context,
	const float3& anchor,
	const float3& offset1,
	const float3& offset2)
{
	Geometry parallelogram = context->createGeometry();
	parallelogram->setPrimitiveCount(1u);
	parallelogram->setIntersectionProgram(pgram_intersection);
	parallelogram->setBoundingBoxProgram(pgram_bounding_box);

	float3 normal = normalize(cross(offset1, offset2));
	float d = dot(normal, anchor);
	float4 plane = make_float4(normal, d);

	float3 v1 = offset1 / dot(offset1, offset1);
	float3 v2 = offset2 / dot(offset2, offset2);

	parallelogram["plane"]->setFloat(plane);
	parallelogram["anchor"]->setFloat(anchor);
	parallelogram["v1"]->setFloat(v1);
	parallelogram["v2"]->setFloat(v2);

	GeometryInstance gi = context->createGeometryInstance();
	gi->setGeometry(parallelogram);
	return gi;
}

void loadComplexGeometry(Context context, const Scene scene)
{
	// set up material
	const char *ptx = sutil::getPtxString(SAMPLE_NAME, "optixPathTracer.cu");
	Program diffuse_ch = context->createProgramFromPTXString(ptx, "diffuseTextured");
	Program diffuse_ah = context->createProgramFromPTXString(ptx, "shadow");
	Program diffuse_ah_radiance = context->createProgramFromPTXString(ptx, "any_hit_radiance");

	Material diffuse_light = context->createMaterial();
	Program diffuse_em = context->createProgramFromPTXString(ptx, "diffuseEmitter");
	diffuse_light->setClosestHitProgram(0, diffuse_em);

	// create geometry instances

	// load model
	Aabb model_aabb;
	OptiXMesh mesh;

	mesh.closest_hit = diffuse_ch;
	mesh.any_hit = diffuse_ah;

	mesh.has_any_hit_radiance = true;
	mesh.any_hit_radiance = diffuse_ah_radiance;

	mesh.context = context;

	Matrix4x4 scale_matrix;
	scale_matrix.setRow(0, make_float4(scene.scale.x, 0.0f,			 0.0f,		    0.0f));
	scale_matrix.setRow(1, make_float4(0.0f,		  scene.scale.y, 0.0f,		    0.0f));
	scale_matrix.setRow(2, make_float4(0.0f,		  0.0f,			 scene.scale.z, 0.0f));
	scale_matrix.setRow(3, make_float4(0.0f,		  0.0f,			 0.0f,		    1.0f));

	loadMesh(scene.file_name, mesh, scale_matrix);

	model_aabb.set(mesh.bbox_min, mesh.bbox_max);
	float a = model_aabb.maxExtent();
	context["far_plane"]->setFloat(length(model_aabb.center() - scene.eye) - 0.5f * model_aabb.maxExtent() + model_aabb.maxExtent() * 1.1f);

	GeometryGroup geometry_group = context->createGeometryGroup();
	geometry_group->addChild(mesh.geom_instance);
	geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
	context["top_object"]->set(geometry_group);
	context["top_shadower"]->set(geometry_group);

	const float3 white = make_float3(0.8f, 0.8f, 0.8f);

	const float3 light_em = scene.parallelogram_lights[0].emission;

	ParallelogramLight light = scene.parallelogram_lights[0];
	float3 relativeCorner = light.corner;
	light.corner = make_float3(model_aabb.center().x + relativeCorner.x, model_aabb.center().y + relativeCorner.y, model_aabb.center().z + relativeCorner.z);

	Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
	light_buffer->setFormat(RT_FORMAT_USER);
	light_buffer->setElementSize(sizeof(ParallelogramLight));
	light_buffer->setSize(1u);
	memcpy(light_buffer->map(), &light, sizeof(light));
	light_buffer->unmap();
	context["lights"]->setBuffer(light_buffer);

	GeometryInstance light_parallelogram = createParallelogram(context, 
		light.corner,
		light.v1,
		light.v2);
	setMaterial(light_parallelogram, diffuse_light, "emission_color", light_em);
}

void glutInitialize(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(SAMPLE_NAME);
	glutHideWindow();
}


void glutRun()
{
	// Initialize GL state                                                            
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, width, height);

	glutShowWindow();
	glutReshapeWindow(width, height);

	// register glut callbacks
	glutDisplayFunc(glutDisplay);
	glutIdleFunc(glutDisplay);
	glutReshapeFunc(glutResize);
	glutKeyboardFunc(glutKeyboardPress);
	glutMouseFunc(glutMousePress);
	glutMotionFunc(glutMouseMotion);

	registerExitHandler();

	glutMainLoop();
}

//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
	camera->update(frame_number);
	
	updateCurrentLevelAdaptiveVariables(context, camera->getChanged());

#ifdef TEST_HOELDER
	if (camera->getChanged())
	{
		currentTotalTimeElapsed = 0.0f;
		equalTimeComparisonDone = 0;
	}

	if (EQUAL_TIME_COMPARISON_ACTIVE && !equalTimeComparisonDone)
	{
		auto start = std::chrono::high_resolution_clock::now();
		context->launch(0, width, height);
		auto end = std::chrono::high_resolution_clock::now();

		// Convert from ms to s and add to total elapsed time.
		currentTotalTimeElapsed +=  0.001 * (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
	}
	if (EQUAL_QUANTITY_COMPARISON_ACTIVE && !equalQuantityComparisonDone)
	{
		context->launch(0, width, height);
	}
#else
	context->launch(0, width, height);
#endif // TEST_HOELDER

	//sutil::displayBufferGL(getOutputDepthBuffer());
	//sutil::displayBufferGL(getDepthGradientBuffer());
	//sutil::displayBufferGL(getHoelderAlphaBuffer());
	//sutil::displayBufferGL(getHoelderRefinementBuffer());
	//sutil::displayBufferGL(getTotalSampleCountBuffer());
	//sutil::displayBufferGL(getHoelderAdaptiveSceneDepthBuffer());
	//sutil::displayBufferGL(getPerWindowVarianceBuffer());

	sutil::displayBufferGL(getOutputBuffer());
	camera->setChanged(false);
	{
		static unsigned frame_count = 0;
		sutil::displayFps(frame_count++);
	}

	//initial_render_run = false;

#ifdef TEST_HOELDER
	if (EQUAL_TIME_COMPARISON_ACTIVE && !equalTimeComparisonDone)
	{
		if (currentTotalTimeElapsed >= TIME_IN_SECONDS)
		{
			const std::string outputImage = std::string(SAMPLE_NAME) + "_time_" + std::to_string(currentTotalTimeElapsed) + ".ppm";
			std::cerr << "Saving current frame to '" << outputImage << "'\n";
			sutil::displayBufferPPM(outputImage.c_str(), getOutputBuffer(), false);

			equalTimeComparisonDone = 1;
		}
	}
	if (EQUAL_QUANTITY_COMPARISON_ACTIVE && !equalQuantityComparisonDone)
	{
		if (currentTotalSampleCount == SAMPLE_PER_PIXEL_QUANTITY)
		{
			const std::string outputImage = std::string(SAMPLE_NAME) + "_samples_" + std::to_string(currentTotalSampleCount) + ".ppm";
			std::cerr << "Saving current frame to '" << outputImage << "'\n";
			sutil::displayBufferPPM(outputImage.c_str(), getOutputBuffer(), false);

			equalQuantityComparisonDone = 1;
		}
	}
#endif // TEST_HOELDER


	glutSwapBuffers();
}

void glutKeyboardPress(unsigned char k, int x, int y)
{

	switch (k)
	{
	case('q'):
	case(27): // ESC
	{
		destroyContext();
		exit(0);
	}
	case('s'):
	{
		const std::string outputImage = std::string(SAMPLE_NAME) + "_" + std::to_string(frame_number) + ".ppm";
		std::cerr << "Saving current frame to '" << outputImage << "'\n";
		sutil::displayBufferPPM(outputImage.c_str(), getOutputBuffer(), false);
		break;
	}
	}
}


void glutMousePress(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_button = button;
		mouse_prev_pos = make_int2(x, y);
	}
	else
	{
		// nothing
	}
}

void glutMouseMotion(int x, int y)
{
	if (mouse_button == GLUT_RIGHT_BUTTON)
	{
		const float dx = static_cast<float>(x - mouse_prev_pos.x) /
			static_cast<float>(width);
		const float dy = static_cast<float>(y - mouse_prev_pos.y) /
			static_cast<float>(height);
		const float dmax = fabsf(dx) > fabs(dy) ? dx : dy;
		const float scale = std::min<float>(dmax, 0.9f);
		camera->setEye(camera->getEye() + (camera->getLookat() - camera->getEye()) * scale);
		camera->setChanged(true);
	}
	else if (mouse_button == GLUT_LEFT_BUTTON)
	{
		const float2 from = { static_cast<float>(mouse_prev_pos.x),
			static_cast<float>(mouse_prev_pos.y) };
		const float2 to = { static_cast<float>(x),
			static_cast<float>(y) };

		const float2 a = { from.x / width, from.y / height };
		const float2 b = { to.x / width, to.y / height };

		camera->setRotation(arcball.rotate(b, a));
		camera->setChanged(true);
	}

	mouse_prev_pos = make_int2(x, y);
}

void glutResize(int w, int h)
{
	if (w == (int)width && h == (int)height) return;

	camera->setChanged(true);

	width = w;
	height = h;

	sutil::resizeBuffer(getOutputBuffer(), width, height);

	glViewport(0, 0, width, height);

	glutPostRedisplay();
}

void printUsageAndExit( const char* argv0 );

struct UsageReportLogger
{
	void log(int lvl, const char* tag, const char* msg)
	{
		std::cout << "[" << lvl << "][" << std::left << std::setw(12) << tag << "] " << msg;
	}
};

// Static callback
void usageReportCallback(int lvl, const char* tag, const char* msg, void* cbdata)
{
	// Route messages to a C++ object (the "logger"), as a real app might do.
	// We could have printed them directly in this simple case.

	UsageReportLogger* logger = reinterpret_cast<UsageReportLogger*>(cbdata);
	logger->log(lvl, tag, msg);
}

int main(int argc, char* argv[])
{
	std::string out_file;
	std::string mesh_file = std::string(sutil::samplesDir()) + "/data/cow.obj";
	for (int i = 1; i<argc; ++i)
	{
		const std::string arg(argv[i]);

		if (arg == "-h" || arg == "--help")
		{
			printUsageAndExit(argv[0]);
		}
		else if (arg == "-f" || arg == "--file")
		{
			if (i == argc - 1)
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit(argv[0]);
			}
			out_file = argv[++i];
		}
		else if (arg == "-n" || arg == "--nopbo")
		{
			use_pbo = false;
		}
		else if (arg == "-m" || arg == "--mesh")
		{
			if (i == argc - 1)
			{
				std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
				printUsageAndExit(argv[0]);
			}
			mesh_file = argv[++i];
		}
		else
		{
			std::cerr << "Unknown option '" << arg << "'\n";
			printUsageAndExit(argv[0]);
		}
	}

	AdaptivePathTraceContext* adaptive_PT_context = new AdaptivePathTraceContext();

    try { 
		Scene scene = KilleroosSceneSetupAndGet();

		width = scene.width;
		height = scene.height;

		glutInitialize(&argc, argv);

#ifndef __APPLE__
		glewInit();
#endif
		//setCurrentInitialRenderNumSamples(num_samples);

		adaptive_PT_context->setWidth(width);
		adaptive_PT_context->setHeight(height);

		adaptive_PT_context->setFrameNumber(frame_number);
		adaptive_PT_context->setSqrtNumSample(sqrt_num_samples);
		adaptive_PT_context->setNumSample(num_samples);

		adaptive_PT_context->setWindowSize(windowSize);
		adaptive_PT_context->setMaxAdditionalRaysTotal(maxAdditionalRaysTotal);
		adaptive_PT_context->setMaxAdditionalRaysPerRenderRun(maxAdditionalRaysPerRenderRun);

		adaptive_PT_context->createAdaptiveContext(SAMPLE_NAME, "optixPathTracer.cu", "adaptivePostProcessing.cu", use_pbo, rr_begin_depth);

		float2 radius = make_float2(1.0f, 1.0f);

		setupMitchellFilter(adaptive_PT_context->getContext()->getContext(), radius, sqrt_num_samples);

		camera = new TrackballCamera(adaptive_PT_context->getContext()->getContext(), (int)width, (int)height);
		camera->setup(scene.eye, scene.look_at, scene.up, scene.fov, true);

		loadComplexGeometry(adaptive_PT_context->getContext()->getContext(), scene);

		context = adaptive_PT_context->getContext()->getContext();

		// Testing new implementation
		setCurrentLevelWindowSize(context);

		context->validate();

		if (out_file.empty())
		{
			glutRun();
		}
		else
		{
			context->launch(0, width, height);
			sutil::displayBufferPPM(out_file.c_str(), getOutputBuffer(), false);
		}

		delete adaptive_PT_context;
        return( 0 );

    } SUTIL_CATCH( adaptive_PT_context->getContext()->getContext()->get() )
}


void printUsageAndExit( const char* argv0 )
{
  fprintf( stderr, "Usage  : %s [options]\n", argv0 );
  fprintf( stderr, "Options: --file | -f <filename>      Specify file for image output\n" );
  fprintf( stderr, "         --help | -h                 Print this usage message\n" );
  fprintf( stderr, "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n" );
  exit(1);
}