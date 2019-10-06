#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cmath>
#include <vector>

namespace ScanMatching {
	void initSimulation(int N1, glm::vec3* src_pc, int N2, glm::vec3* target_pc);
	void stepSimulationNaive(float dt);
	void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);

	void endSimulation();
}