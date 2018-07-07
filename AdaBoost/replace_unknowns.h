#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

void replaceUnknowns(vector<vector<string>>& data, vector<Attribute> attributes);
string mostCommonValueofAttribute(vector<vector<string>>& data, Attribute attr);

//to replace unknowns in data and test file
void replaceUnknowns(vector<vector<string>>& data, vector<Attribute> attributes)
{
	//only attributes workclass, ocuupation, and native-country consist of unknown values
	string most_common_value_workclass = mostCommonValueofAttribute(data, attributes[1]);
	string most_common_value_occupation = mostCommonValueofAttribute(data, attributes[6]);
	string most_common_value_native_country = mostCommonValueofAttribute(data, attributes[13]);

	for (int i = 0; i < data.size(); i++)
	{
		if (data[i][1] == "?")
		{
			data[i][1] = most_common_value_workclass;
		}
		if (data[i][6] == "?")
		{
			data[i][6] = most_common_value_occupation;
		}
		if (data[i][13] == "?")
		{
			data[i][13] = most_common_value_native_country;
		}
	}
}

//takes the data set and returns the most common value of attribute attr in that set
string mostCommonValueofAttribute(vector<vector<string>>& data, Attribute attr)
{
	vector<int> attr_values;
	attr_values.resize(attr.values);
	for (int i = 0; i < data.size(); i++)
	{
		if (data[i][attr.index] == "?")
		{

		}
		else
		{
			attr_values[(attr.map[data[i][attr.index]]) - 1]++;
		}
	}
	int max_attr;
	vector<int>::iterator max;
	max = max_element(attr_values.begin(), attr_values.end());
	max_attr = distance(attr_values.begin(), max);
	//max_attr+1 is the value occuring most
	map<string, int>::iterator it;
	for (it = attr.map.begin(); it != attr.map.end(); it++)
	{
		if (it->second == max_attr + 1)
		{
			return it->first;
		}
	}

}