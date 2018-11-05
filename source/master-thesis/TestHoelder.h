#ifndef TEST_HOELDER

#define TEST_HOELDER
#include <chrono>

const int USE_HOELDER_ADAPTIVE = 1;

const int KILLEROO_SCENE = 0;
const int BARCELONA_PAVILLON_SCENE = 0;

const int EQUAL_TIME_COMPARISON_ACTIVE = 1;
const int EQUAL_QUANTITY_COMPARISON_ACTIVE = 0;

// variables to modify
int equalTimeComparisonDone = 0;
int equalQuantityComparisonDone = 0;

const int TIME_IN_SECONDS = 10;
const int SAMPLE_PER_PIXEL_QUANTITY = 64;

// variables to modify
unsigned int currentTotalSampleCount = 0;
float currentTotalTimeElapsed = 0.0f;

#endif // !DEBUG_HOELDER