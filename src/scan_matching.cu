#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "scan_matching.h"

#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		if (line >= 0) {
			fprintf(stderr, "Line %d: ", line);
		}
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

/*! Size of the starting area in simulation space. */
#define scene_scale 0.1f


glm::vec3 *dev_src_pc;
glm::vec3 *dev_target_pc;

int N_SRC;
int N_TARGET;


/******************
* initSimulation *
******************/

__global__ void kernResetBuffer(int N, glm::vec3 *buffer, glm::vec3 value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		buffer[index] = value;
	}
}

/**
* Initialize memory, update some globals
*/
void ScanMatching::initSimulation(int N, glm::vec3* src_pc, int M, glm::vec3* target_pc) {
	cudaMalloc((void**)&dev_src_pc, N * sizeof(glm::vec3));
	cudaMalloc((void**)&dev_target_pc, M * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMemcpy(dev_src_pc, src_pc, N * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_target_pc, target_pc, M * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	N_SRC = N;
	N_TARGET = M;


	cudaDeviceSynchronize();
}

/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N_SRC, glm::vec3 *pos_src, int N_TARGET, glm::vec3 *pos_target,  float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale = -1.0f / s_scale;

	if (index < N_SRC + N_TARGET) {
		if (index < N_SRC) {
			vbo[4 * index + 0] = pos_src[index].x * c_scale;
			vbo[4 * index + 1] = pos_src[index].y * c_scale;
			vbo[4 * index + 2] = pos_src[index].z * c_scale;
			vbo[4 * index + 3] = 1.0f;
		}
		else {
			vbo[4 * index + 0] = pos_target[index - N_SRC].x * c_scale;
			vbo[4 * index + 1] = pos_target[index - N_SRC].y * c_scale;
			vbo[4 * index + 2] = pos_target[index - N_SRC].z * c_scale;
			vbo[4 * index + 3] = 1.0f;
		}
	}
}

__global__ void kernCopyVelocitiesToVBO(int N_SRC, int N_TARGET, float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N_SRC + N_TARGET) {
		if (index < N_SRC) {
			vbo[4 * index + 0] = 1.0 + 0.3f;
			vbo[4 * index + 1] = 1.0 + 0.3f;
			vbo[4 * index + 2] = 1.0 + 0.3f;
			vbo[4 * index + 3] = 1.0f;
		}
		else {
			vbo[4 * index + 0] = 1.0 + 0.3f;
			vbo[4 * index + 1] = 0.0 + 0.3f;
			vbo[4 * index + 2] = 0.0 + 0.3f;
			vbo[4 * index + 3] = 1.0f;
		}
	}
}


/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void ScanMatching::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
	dim3 fullBlocksPerGrid((N_SRC + N_TARGET + blockSize - 1) / blockSize);

	kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (N_SRC, dev_src_pc, N_TARGET, dev_target_pc, vbodptr_positions, scene_scale);
	kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> > (N_SRC, N_TARGET, vbodptr_velocities, scene_scale);
	checkCUDAErrorWithLine("copyBoidsToVBO failed!");
	cudaDeviceSynchronize();
}

/******************
* stepSimulation *
******************/


__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
	return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernFindCorrespondences(int numSrc, glm::vec3* dev_src, int numTarget, glm::vec3* dev_target, glm::vec3* dev_corres) {
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);

	if (idx >= numSrc) return;

	glm::vec3 me = dev_src[idx];
	int minIdx = 0;
	float minDistance = glm::distance(me, dev_target[0]);
	for (int i = 1; i < numTarget; i++) {
		float d = glm::distance(me, dev_target[idx]);
		if ( d < minDistance) {
			minIdx = i;
			minDistance = d;
		}
	}

	dev_corres[idx] = dev_target[minIdx];
}



/**
* Step the entire N-body simulation by `dt` seconds.
*/
void ScanMatching::stepSimulationNaive(float dt) {

	//glm::vec3* dev_corres;
	//cudaMalloc((void **)&dev_corres, N_SRC * sizeof(glm::vec3));


	//dim3 numBlocksPerGrid((N_SRC + blockSize - 1) / blockSize);

	//kernFindCorrespondences << < numBlocksPerGrid, blockSize >> > (N_SRC, dev_src_pc, N_TARGET, dev_target_pc, dev_corres);


	////printf("%.4f %.4f %.4f\n", corres[0].x, corres[0].y, corres[0].z);

	//cudaDeviceSynchronize();
}

void ScanMatching::endSimulation() {


	cudaFree(dev_src_pc);
	cudaFree(dev_target_pc);

}