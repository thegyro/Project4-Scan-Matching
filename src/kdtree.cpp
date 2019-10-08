#include <algorithm>
#include <iostream>
#include "kdtree.h"

struct compareVec {
	int idx;
	compareVec(int i) {
		idx = i;
	}
	bool operator() (glm::vec3 v1, glm::vec3 v2) {
		return v1[idx] < v2[idx];
	}
};

TreeNode* constructKDTree(glm::vec3* points, int n) {

	glm::vec3** points_sorted = new glm::vec3*[3];
	for (int i = 0; i < 3; i++) {
		points_sorted[i] = new glm::vec3[n];
		memcpy(points_sorted[i], points, n * sizeof(glm::vec3));
		std::sort(points_sorted[i], points_sorted[i] + n, compareVec(i));
	}

	constructKDTreeUtil(points_sorted, 0, n - 1, 0);

	for (int i = 0; i < 3; i++) {
		delete[] points_sorted[i];
	}
	delete[] points_sorted;

	return NULL;
}

TreeNode* constructKDTreeUtil(glm::vec3** points_sorted, int start, int end, int depth) {
	
	if (start > end) return NULL;

	int cd = depth % 3;
	int median = start + (end - start) / 2;

	glm::vec3 point = points_sorted[cd][median];
	TreeNode* node = new TreeNode(point);
	node->left = constructKDTreeUtil(points_sorted, start, median - 1, depth + 1);
	node->right = constructKDTreeUtil(points_sorted, median + 1, end, depth + 1);

	return node;
}

int main(int argc, char* argv[]) {

	glm::vec3* points = new glm::vec3[3];

	points[0] = glm::vec3(3.0f, 2.0f, 4.0f);
	points[1] = glm::vec3(1.0f, 4.0f, 2.0f);
	points[2] = glm::vec3(2.0f, 3.0f, 1.0f);

	constructKDTree(points, 3);


}