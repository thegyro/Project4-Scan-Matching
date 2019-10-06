#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <fstream>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include "main.hpp"


#define VISUALIZE 1

const float DT = 0.2f;

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-0.01, 0.01);

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg)

glm::vec3 *src_pc, *target_pc;
int numSrc, numTarget;

void printArray2D(float *X, int nR, int nC) {
	for (int i = 0; i < nR; i++) {
		for (int j = 0; j < nC; j++)
			printf("%.2f ", X[i*nC + j]);
		printf("\n");
	}
}

void readPointCloud(const std::string& filename, glm::vec3 **points, int* n) {

	std::ifstream myfile(filename);
	std::string line;
	std::cout << filename << '\n';
	std::cout << myfile.is_open() << '\n';
	if (myfile.is_open()) {
		for (int i = 1; i <= 17; i++) {
			std::getline(myfile, line);
			std::cout << line << '\n';
		}

		// read element vertex
		std::getline(myfile, line);
		std::istringstream tokenStream(line);
		std::string token;
		for (int i = 1; i <= 3; i++) {
			std::getline(tokenStream, token, ' ');
		}

		int numVertex = std::stoi(token);
		*n = numVertex;
		std::cout << numVertex << '\n';

		*points = (glm::vec3 *) malloc(numVertex * sizeof(glm::vec3));

		for (int i = 1; i <= 6; i++) {
			std::getline(myfile, line);
			std::cout << line << '\n';
		}

		for (int i = 0; i < numVertex; i++) {
			std::getline(myfile, line);
			std::istringstream ss(line);
			std::string buf;
			ss >> buf;
			float x = std::stof(buf);
			ss >> buf;
			float y = std::stof(buf);
			ss >> buf;
			float z = std::stof(buf);

			(*points)[i] = glm::vec3(x, y, z);

			//std::cout << (*X)[i * 3 + 0] << ' ' << (*X)[i * 3 + 1] << ' ' << (*X)[i * 3 + 2] << '\n';
		}

		myfile.close();
	}
}

int main(int argc, char* argv[]) {

	std::string src_filename = "../data/bunny/data/bun000.ply";
	std::string target_filename = "../data/bunny/data/bun045.ply";

	readPointCloud(src_filename, &src_pc, &numSrc);
	readPointCloud(target_filename, &target_pc, &numTarget);
	std::cout << numSrc << ' ' << numTarget << '\n';

	//for (int i = 0; i < nSrc; i++) {
	//	std::cout << src[i].x << ' ' << src[i].y << ' ' << src[i].z << '\n';
	//}

	if (init(argc, argv)) {
		mainLoop();
		ScanMatching::endSimulation();
		return 0;
	}
	else {
		return 1;
	}
}


std::string deviceName;
GLFWwindow *window;

/**
* Initialization of CUDA and GLFW.
*/
bool init(int argc, char **argv) {
	// Set window title to "Student Name: [SM 2.0] GPU Name"
	cudaDeviceProp deviceProp;
	int gpuDevice = 0;
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	if (gpuDevice > device_count) {
		std::cout
			<< "Error: GPU device number is greater than the number of devices!"
			<< " Perhaps a CUDA-capable GPU is not installed?"
			<< std::endl;
		return false;
	}
	cudaGetDeviceProperties(&deviceProp, gpuDevice);
	int major = deviceProp.major;
	int minor = deviceProp.minor;

	std::ostringstream ss;
	//ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
	//deviceName = ss.str();
	deviceName = "batman";

	// Window setup stuff
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		std::cout
			<< "Error: Could not initialize GLFW!"
			<< " Perhaps OpenGL 3.3 isn't available?"
			<< std::endl;
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
	if (!window) {
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, mousePositionCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}

	// Initialize drawing state
	initVAO();

	// Default to device ID 0. If you have more than one GPU and want to test a non-default one,
	// change the device ID.
	cudaGLSetGLDevice(0);

	cudaGLRegisterBufferObject(boidVBO_positions);
	cudaGLRegisterBufferObject(boidVBO_velocities);

	// Initialize N-body simulation
	ScanMatching::initSimulation(numSrc, src_pc, numTarget, target_pc);

	updateCamera();

	initShaders(program);

	glEnable(GL_DEPTH_TEST);

	return true;
}

void initVAO() {

	int numTotal = numSrc + numTarget;

	std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (numTotal)] };
	std::unique_ptr<GLuint[]> bindices{ new GLuint[numTotal] };

	glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
	glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

	for (int i = 0; i < numTotal; i++) {
		bodies[4 * i + 0] = 0.0f;
		bodies[4 * i + 1] = 0.0f;
		bodies[4 * i + 2] = 0.0f;
		bodies[4 * i + 3] = 1.0f;
		bindices[i] = i;
	}


	glGenVertexArrays(1, &boidVAO); // Attach everything needed to draw a particle to this
	glGenBuffers(1, &boidVBO_positions);
	glGenBuffers(1, &boidVBO_velocities);
	glGenBuffers(1, &boidIBO);

	glBindVertexArray(boidVAO);

	// Bind the positions array to the boidVAO by way of the boidVBO_positions
	glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions); // bind the buffer
	glBufferData(GL_ARRAY_BUFFER, 4 * numTotal * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

	glEnableVertexAttribArray(positionLocation);
	glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	// Bind the velocities array to the boidVAO by way of the boidVBO_velocities
	glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
	glBufferData(GL_ARRAY_BUFFER, 4 * numTotal * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(velocitiesLocation);
	glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, numTotal * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

	glBindVertexArray(0);
}

void initShaders(GLuint * program) {
	GLint location;

	program[PROG_BOID] = glslUtility::createProgram(
		"shaders/boid.vert.glsl",
		"shaders/boid.geom.glsl",
		"shaders/boid.frag.glsl", attributeLocations, 2);
	glUseProgram(program[PROG_BOID]);

	if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
	if ((location = glGetUniformLocation(program[PROG_BOID], "u_cameraPos")) != -1) {
		glUniform3fv(location, 1, &cameraPosition[0]);
	}
}

//====================================
// Main loop
//====================================
void runCUDA() {
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
	// use this buffer

	float4 *dptr = NULL;
	float *dptrVertPositions = NULL;
	float *dptrVertVelocities = NULL;

	cudaGLMapBufferObject((void**)&dptrVertPositions, boidVBO_positions);
	cudaGLMapBufferObject((void**)&dptrVertVelocities, boidVBO_velocities);

	// execute the kernel
	ScanMatching::stepSimulationNaive(DT);

#if VISUALIZE
	ScanMatching::copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);
#endif
	// unmap buffer object
	cudaGLUnmapBufferObject(boidVBO_positions);
	cudaGLUnmapBufferObject(boidVBO_velocities);
}

void mainLoop() {
	double fps = 0;
	double timebase = 0;
	int frame = 0;

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		frame++;
		double time = glfwGetTime();

		if (time - timebase > 1.0) {
			fps = frame / (time - timebase);
			timebase = time;
			frame = 0;
		}

		runCUDA();

		std::ostringstream ss;
		ss << "[";
		ss.precision(1);
		ss << std::fixed << fps;
		ss << " fps] " << deviceName;
		glfwSetWindowTitle(window, ss.str().c_str());

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#if VISUALIZE
		glUseProgram(program[PROG_BOID]);
		glBindVertexArray(boidVAO);
		glPointSize((GLfloat)pointSize);
		glDrawElements(GL_POINTS,  numSrc + numTarget, GL_UNSIGNED_INT, 0);
		glPointSize(1.0f);

		glUseProgram(0);
		glBindVertexArray(0);

		glfwSwapBuffers(window);
#endif
	}
	glfwDestroyWindow(window);
	glfwTerminate();
}


void errorCallback(int error, const char *description) {
	fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (leftMousePressed) {
		// compute new camera parameters
		phi += (xpos - lastX) / width;
		theta -= (ypos - lastY) / height;
		theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
		updateCamera();
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
		updateCamera();
	}

	lastX = xpos;
	lastY = ypos;
}

void updateCamera() {
	cameraPosition.x = zoom * sin(phi) * sin(theta);
	cameraPosition.z = zoom * cos(theta);
	cameraPosition.y = zoom * cos(phi) * sin(theta);
	cameraPosition += lookAt;

	projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
	glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
	projection = projection * view;

	GLint location;

	glUseProgram(program[PROG_BOID]);
	if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
}