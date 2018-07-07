#include <vector>

struct TreeNode;
struct TreeBranch;

struct TreeNode
{
	char label;		//determines the node's attribute or '+' or '-'
	vector<TreeBranch *> branch;
};

struct TreeBranch
{
	int label;			//determines the branch's value
	TreeNode *child;	//node to which this branch points
};