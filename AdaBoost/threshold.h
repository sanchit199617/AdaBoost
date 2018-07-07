#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include "tree_structure.h"
#include "attributes_structure.h"
#include "gain.h"

bool sortByData(ContAttr lhs, ContAttr rhs);
float threshold(vector<vector<string>>& data, Attribute attr);
float infoGainOfThresholdCandidate(vector<vector<string>>& data, Attribute attr, vector<ContAttr>& contattr, float candidate);

//helps in sorting the data according to a particular continous attribute (helps in calculating the threshold of that attribute)
bool sortByData(ContAttr lhs, ContAttr rhs)
{
	return lhs.data < rhs.data;
}

//function to find the threshold of continous attributes
float threshold(vector<vector<string>>& data, Attribute attr)
{
	//average method of finding thresholds
	float sum = 0;
	for (int i = 0; i < data.size(); i++)
	{
		sum = sum + stoi(data[i][attr.index]);
	}
	return (sum / data.size());

	/*
	//first sort the data according to that continous attribute attr
	vector<ContAttr> contattr;
	contattr.resize(data.size());
	for (int i = 0; i < contattr.size(); i++)
	{
		contattr[i].data = stoi(data[i][attr.index]);
		contattr[i].output = data[i][14];
	}
	sort(contattr.begin(), contattr.end(), sortByData);

	//then find the candidates for threshold
	vector<float> candidates;
	int size = 0;
	for (int i = 0; i < contattr.size() - 1; i++)
	{
		if (contattr[i].output != contattr[i + 1].output)
		{
			size++;
		}
	}
	candidates.resize(size);
	for (int i = 0, j = 0; i < contattr.size() - 1; i++)
	{
		if (contattr[i].output != contattr[i + 1].output)
		{
			candidates[j] = (contattr[i].data + contattr[i + 1].data) / 2.0;
			j++;
		}
	}
	//remove duplicates from candidates vector
	vector<float>::iterator it;
	it = unique(candidates.begin(), candidates.end());
	candidates.resize(distance(candidates.begin(), it));

	//then find the information gain of all candidates
	vector<float> infogain;									//this vector will store the information gain of each candidate
	infogain.resize(candidates.size());
	for (int i = 0; i < candidates.size(); i++)
	{
		infogain[i] = infoGainOfThresholdCandidate(data, attr, contattr, candidates[i]);
	}

	//find the candidate with most information gain and return its value
	float max_info = *max_element(infogain.begin(), infogain.end());
	for (int i = 0; i < infogain.size(); i++)
	{
		if (infogain[i] == max_info)
		{
			return candidates[i];
		}
	}
	*/
}

//this function finds the information gain of a candidate (helps in finding the threshold of a continous attribute)
float infoGainOfThresholdCandidate(vector<vector<string>>& data, Attribute attr, vector<ContAttr>& contattr, float candidate)
{
	map<string, int> candidates_map;
	//find size of both partitions
	int p1_size = 0, p2_size = 0;
	while (p1_size < contattr.size() && contattr[p1_size].data <= candidate)
	{
		p1_size++;
	}
	p2_size = contattr.size() - p1_size;

	for (int j = 0; j < p1_size; j++)
	{
		candidates_map[to_string(contattr[j].data)] = 1;
	}
	for (int j = p1_size; j < contattr.size(); j++)
	{
		candidates_map[to_string(contattr[j].data)] = 2;
	}
	attr.map = candidates_map;
	float candidate_gain = gain(data, attr);
	return candidate_gain;
}