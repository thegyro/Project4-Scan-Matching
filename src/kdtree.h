#pragma once
#include <glm/glm.hpp>

struct TreeNode {
	glm::vec3 point;
	TreeNode* left;
	TreeNode* right;

	TreeNode(glm::vec3 p) {
		point = p;
		left = NULL;
		right = NULL;
	}
};