#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "scan_matching.h"
#include "svd3.h"

#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

void printArray2D(glm::mat3& X) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++)
			printf("%f ", X[j][i]);
		printf("\n");
	}
	printf("\n");
}

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
glm::vec3 *dev_src_pc_shift;
glm::vec3 *dev_target_pc;

glm::vec3 *dev_X_mean_sub;
glm::vec3 *dev_Y_mean_sub;

glm::vec3* dev_corres;

glm::mat3* dev_M_yxt;

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
	cudaMalloc((void**)&dev_src_pc_shift, N * sizeof(glm::vec3));
	cudaMalloc((void**)&dev_X_mean_sub, N * sizeof(glm::vec3));
	cudaMalloc((void**)&dev_Y_mean_sub, N * sizeof(glm::vec3));
	cudaMalloc((void **)&dev_corres, N * sizeof(glm::vec3));
	cudaMalloc((void **)&dev_M_yxt, N * sizeof(glm::mat3));

	cudaMalloc((void**)&dev_target_pc, M * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc failed!");

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

//struct sum
//{
//	__host__ __device__
//		glm::vec3 operator()(const glm::vec3& v1, const glm::vec3& v2){
//		return v1 + v2;
//	}
//};


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
		float d = glm::distance(me, dev_target[i]);
		if ( d < minDistance) {
			minIdx = i;
			minDistance = d;
		}
	}

	dev_corres[idx] = dev_target[minIdx];
}

/*
	Calculate Y (target 3 x N) x X.T (source N x 3) result 3 x 3 
*/

__global__ void kernMultiplyYXTranspose(int N, glm::vec3* tars, glm::vec3* srcs, glm::mat3* outM) {
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);

	if (idx >= N) return;

	glm::vec3 s = srcs[idx];
	glm::vec3 t = tars[idx];

	outM[idx] = glm::outerProduct(t, s);

	//outM[idx] = glm::mat3(t.x*s.x, t.x*s.y, t.x*s.z,
	//					t.y*s.x, t.y*s.y, t.y*s.z,
	//					t.z*s.x, t.z*s.y, t.z*s.z);

}

__global__ void kernTranspose(int) {

}

struct mean_center_op
{
	const glm::vec3 mean;

	mean_center_op(glm::vec3 _a) : mean(_a) {}

	__host__ __device__
		glm::vec3 operator()(const glm::vec3& x) const {
		return x - mean;
	}
};

struct transform_src_op
{
	const glm::vec3 t;
	const glm::mat3 R;

	transform_src_op(glm::vec3 _a, glm::mat3 _b) : t(_a), R(_b) {}

	__host__ __device__
		glm::vec3 operator()(const glm::vec3& x) const {
		return R*x + t;
	}
};


void multiply(glm::mat3& mat1,
	glm::mat3& mat2,
	glm::mat3& res)
{
	int i, j, k;
	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{
			res[i][j] = 0;
			for (k = 0; k < 3; k++)
				res[i][j] += mat1[i][k] *
				mat2[k][j];
		}
	}
}



/**
* Step the entire N-body simulation by `dt` seconds.
*/
void ScanMatching::stepSimulationNaive(float dt) {
	dim3 numBlocksPerGrid((N_SRC + blockSize - 1) / blockSize);

	kernFindCorrespondences << < numBlocksPerGrid, blockSize >> > (N_SRC, dev_src_pc, N_TARGET, dev_target_pc, dev_corres);

	glm::vec3 mean_src = thrust::reduce(thrust::device, dev_src_pc, dev_src_pc + N_SRC, glm::vec3(0.0f));
	mean_src /= (float) N_SRC;

	glm::vec3 mean_tar = thrust::reduce(thrust::device, dev_corres, dev_corres + N_SRC, glm::vec3(0.0f));
	mean_tar /= (float)N_SRC;

	//glm::vec3* check = new glm::vec3[N_SRC];
	//cudaMemcpy(check, dev_corres, N_SRC * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < 5; i++) {
	//	//std::cout << check[i].x << " " << check[i].y << " " << check[i].z << '\n';
	//}
	//std::cout << '\n';


	std::cout << mean_src.x << " " << mean_src.y << " " << mean_src.z << '\n';
	std::cout << mean_tar.x << " " << mean_tar.y << " " << mean_tar.z << '\n';

	mean_center_op opX(mean_src);
	mean_center_op opY(mean_tar);

	thrust::transform(thrust::device, dev_src_pc, dev_src_pc + N_SRC, dev_X_mean_sub, opX);
	thrust::transform(thrust::device, dev_corres, dev_corres + N_SRC, dev_Y_mean_sub, opY);

	kernMultiplyYXTranspose<<< numBlocksPerGrid, blockSize >>> (N_SRC, dev_Y_mean_sub, dev_X_mean_sub, dev_M_yxt);

	glm::mat3 host_M = thrust::reduce(thrust::device, dev_M_yxt, dev_M_yxt + N_SRC, glm::mat3(0.0f));
	//glm::mat3 host_M = glm::transpose(host_MM);

	std::cout << "M" << '\n';
	printArray2D(host_M);

	//float IM[3][3] = { 0 };
	//for (int i = 0; i < 3; i++)
	//	for (int j = 0; j < 3; j++)
	//		IM[i][j] = host_M[j][i];

	//float U[3][3] = { 0 };
	//float S[3][3] = { 0 };
	//float V[3][3] = { 0 };

	glm::mat3 U(0.0f);
	glm::mat3 S(0.0f);
	glm::mat3 V(0.0f);
	//float host_M[3][3];
	//float U[3][3];
	//float S[3][3];
	//float V[3][3];

	svd(host_M[0][0], host_M[1][0], host_M[2][0],
		host_M[0][1], host_M[1][1], host_M[2][1],
		host_M[0][2], host_M[1][2], host_M[2][2],
		U[0][0], U[1][0], U[2][0],
		U[0][1], U[1][1], U[2][1],
		U[0][2], U[1][2], U[2][2],
		S[0][0], S[1][0], S[2][0],
		S[0][1], S[1][1], S[2][1],
		S[0][2], S[1][2], S[2][2],
		V[0][0], V[1][0], V[2][0],
		V[0][1], V[1][1], V[2][1],
		V[0][2], V[1][2], V[2][2]);

	//glm::mat3 Uglm = glm::mat3(
	//	U[0][0], U[1][0], U[2][0],
	//	U[0][1], U[1][1], U[2][1],
	//	U[0][2], U[1][2], U[2][2]);

	//glm::mat3 Vglm = glm::mat3(
	//	V[0][0], V[1][0], V[2][0],
	//	V[0][1], V[1][1], V[2][1],
	//	V[0][2], V[1][2], V[2][2]);

	//std::cout << "U" << '\n';
	//printArray2D(Uglm);
	//std::cout << "V" << '\n';
	//printArray2D(Vglm);

	glm::mat3 R(0.0f);
	//multiply(Uglm, glm::transpose(Vglm), R);
	//R = Uglm *glm::transpose(Vglm);
	R = U * glm::transpose(V);

	std::cout << "R" << '\n';
	printArray2D(R);

	glm::vec3 t = mean_tar - R * mean_src;


	thrust::transform(thrust::device, dev_src_pc, dev_src_pc + N_SRC, dev_src_pc_shift, transform_src_op(t, R));

	cudaMemcpy(dev_src_pc, dev_src_pc_shift, N_SRC * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

	cudaDeviceSynchronize();
}

void ScanMatching::endSimulation() {


	cudaFree(dev_src_pc);
	cudaFree(dev_target_pc);

}