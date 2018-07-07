#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

vector<vector<string>> subset(vector<vector<string>>& data, int value, Attribute a);
float entropy(vector<vector<string>>& data);
float gain(vector<vector<string>>& data, Attribute attr);

//partition the data into subset containing only those rows for which value of attribute a is equal to int value
vector<vector<string>> subset(vector<vector<string>>& data, int value, Attribute attr)
{
	vector<vector<string>> sub;
	int size = 0;								//size of sub
	for (int j = 0; j < data.size(); j++)
	{
		if (attr.map[data[j][attr.index]] == value)
		{
			size++;
		}
	}
	sub.resize(size);
	for (int m = 0; m < size; m++)
	{
		sub[m].resize(15);
	}
	for (int j = 0, k = 0; j < data.size(); j++)
	{
		if (attr.map[data[j][attr.index]] == value)
		{
			for (int l = 0; l < 15; l++)
			{
				sub[k][l] = data[j][l];
			}
			k++;
		}
	}
	return sub;
}

//returns the entropy of a given set of data
float entropy(vector<vector<string>>& data)
{
	float p = 0.0, n = 0.0;
	if (data.size() == 0)
	{
		return  0;
	}
	else
	{
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
		if (p == 0 || n == 0)
		{
			return 0.0;
		}
		float s = p + n;
		return (-((p / s)*(log2((p / s)))) - ((n / s)*(log2((n / s)))));
	}
}

//calculates the gain of attribute attr over data
float gain(vector<vector<string>>& data, Attribute attr)
{
	float gain = 0;
	float sum = 0;
	for (int i = 1; i <= attr.values; i++)
	{
		vector<vector<string>> partition = subset(data, i, attr);
		sum = sum + ((partition.size()*entropy(partition)) / data.size());
	}
	gain = entropy(data) - sum;
	return gain;
}