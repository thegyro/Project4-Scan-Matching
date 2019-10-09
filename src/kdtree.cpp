#include <algorithm>
#include <iostream>
#include "kdtree.h"

#define NULL_VEC -7879

struct compareVec {
	int idx;
	compareVec(int i) {
		idx = i;
	}
	bool operator() (glm::vec3 v1, glm::vec3 v2) {
		return v1[idx] < v2[idx];
	}
};


void constructKDTreeUtil(glm::vec3* tree, glm::vec3* points, int pos, int start, int end, int depth, int n) {

	if (pos > n || start > end) return;

	int cd = depth % 3;
	int median = start + (end - start) / 2;

	std::sort(points + start, points + end+1, compareVec(cd));
	
	std::cout << "Start: " << start << " " << end << " " << median << " " << pos << '\n';
	for (int i = start; i <= end; i++) {
		std::cout << points[i].x << ' ' << points[i].y << ' ' << points[i].z << '\n';
	}
	std::cout << '\n';

	glm::vec3 point = points[median];
	tree[pos] = point;

	constructKDTreeUtil(tree, points, 2*pos+1, start, median-1, depth + 1, n);
	constructKDTreeUtil(tree, points, 2*pos+2, median+1 , end, depth + 1, n);

	return;
}

void print(glm::vec3* tree, int pos, int n) {
	if (pos > n || tree[pos].z == NULL_VEC) return;
	
	glm::vec3 point = tree[pos];
	std::cout << point.x << " " << point.y << " " << point.z << '\n';
	print(tree, 2*pos+1, n);
	print(tree, 2*pos+2, n);
}


glm::vec3* constructKDTree(glm::vec3* points, int n) {


	int treeSize = 1 << ((int)ceil(log2(n)) + 1);
	glm::vec3* tree = new glm::vec3[treeSize];
	for (int i = 0; i < treeSize; i++) {
		tree[i] = glm::vec3(NULL_VEC, NULL_VEC, NULL_VEC);
	}

	constructKDTreeUtil(tree, points, 0, 0, n-1, 0, treeSize);
	return tree;
}

int main(int argc, char* argv[]) {

	//[(7, 2), (5, 4), (9, 6), (4, 7), (8, 1), (2, 3)]

	glm::vec3* points = new glm::vec3[6];

	//points[0] = glm::vec3(3.0f, 2.0f, 4.0f);
	//points[1] = glm::vec3(1.0f, 4.0f, 2.0f);
	//points[2] = glm::vec3(2.0f, 3.0f, 1.0f);

	points[0] = glm::vec3(7, 2, 1);
	points[1] = glm::vec3(5, 4, 1);
	points[2] = glm::vec3(9, 6, 1);
	points[3] = glm::vec3(4, 7, 1);
	points[4] = glm::vec3(8, 1, 1);
	points[5] = glm::vec3(2, 3, 1);

	glm::vec3* root = constructKDTree(points, 6);
	//std::cout << root->point.x << " " << root->point.y << " " << root->point.z << '\n';
	//std::cout << root->left->point.x << " " << root->left->point.y << " " << root->left->point.z << '\n';
	//std::cout << root->right->point.x << " " << root->right->point.y << " " << root->right->point.z << '\n';

	print(root, 0, 1 << ((int)ceil(log2(6))) + 1);
}