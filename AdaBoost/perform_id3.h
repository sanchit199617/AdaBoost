#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

TreeNode * id3(vector<vector<string>>& data, vector<Attribute> attributes);
int mostCommonValue(vector<vector<string>>& data);
Attribute bestAttribute(vector<vector<string>>& data, vector<Attribute> attributes);

//performs id3 and returns the root node of the decision tree
TreeNode * id3(vector<vector<string>>& data, vector<Attribute> attributes)
{
	TreeNode *root = new TreeNode;
	int positive = 1, negative = 1;
	for (int i = 0; i < data.size(); i++)
	{
		if (output_map[data[i][14]] != output_map[">50K"])
		{
			positive = 0;
		}
		else if (output_map[data[i][14]] != output_map["<=50K"])
		{
			negative = 0;
		}
	}
	if (positive == 1)
	{
		root->label = '+';
		return root;
	}
	if (negative == 1)
	{
		root->label = '-';
		return root;
	}
	if (attributes.empty())
	{
		int mcv = mostCommonValue(data);   //most common value, 1 for >50K and 0 for <=50K
		if (mcv == 1)
		{
			root->label = '+';
			return root;
		}
		else
		{
			root->label = '-';
			return root;
		}
	}
	Attribute a = bestAttribute(data, attributes);
	root->label = attributes_map[a.index];
	root->branch.resize(a.values);
	for (int i = 0; i < a.values; i++)
	{
		TreeBranch *b = new TreeBranch;
		b->label = i + 1;
		root->branch[i] = b;
		vector<vector<string>> sub = subset(data, i + 1, a);
		if (sub.empty())
		{
			TreeNode *node = new TreeNode;
			int mcv = mostCommonValue(data);
			if (mcv == 1)
			{
				node->label = '+';
			}
			else
			{
				node->label = '-';
			}
			b->child = node;
		}
		else
		{
			vector<Attribute> attributes_left = attributes;
			int remove_pos;
			for (int j = 0; j < attributes_left.size(); j++)
			{
				if (attributes_left[j].index == a.index)
				{
					remove_pos = j;
				}
			}
			attributes_left.erase(attributes_left.begin() + remove_pos);
			b->child = id3(sub, attributes_left);
		}
	}
	return root;
}

//returns 1 if >50K is more in the data, else returns 0
int mostCommonValue(vector<vector<string>>& data)
{
	int p = 0, n = 0;
	for (int i = 0; i < data.size(); i++)
	{
		if (output_map[data[i][14]] == output_map[">50K"])
		{
			p++;
		}
		else if (output_map[data[i][14]] == output_map["<=50K"])
		{
			n++;
		}
	}
	if (p < n)
	{
		return 0;
	}
	else
	{
		return 1;
	}
}

Attribute bestAttribute(vector<vector<string>>& data, vector<Attribute> attributes)
{
	vector<float> info_gain;
	info_gain.resize(attributes.size());
	for (int i = 0; i < attributes.size(); i++)
	{
		info_gain[i] = gain(data, attributes[i]);
	}
	float max_gain = *max_element(info_gain.begin(), info_gain.end());
	for (int i = 0; i < info_gain.size(); i++)
	{
		if (info_gain[i] == max_gain)
		{
			return attributes[i];
		}
	}
}