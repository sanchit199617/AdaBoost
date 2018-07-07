#include <map>

struct Attribute
{

	map<string, int> map;
	int index;				//determines the attributes index in the table
	int values;				//number of values taken by this attribute
	float threshold;		//threshold of the continous attribute, is zero for all nominal attributes

	Attribute(int i, int v)
	{
		index = i;
		values = v;
	}
};

// this structure helps in finding the threshold of a continous attribute
struct ContAttr
{
	int data;			//data in that continous attribute
	string output;		//corresponding output value
};