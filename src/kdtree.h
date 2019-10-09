#pragma once
#include <glm/glm.hpp>

struct TreeNode {
	glm::vec3 point;
	bool good;
	int depth;

	TreeNode(glm::vec3 p, bool g, int d) {
		point = p;
		good = g;
		depth = d;
	}
};