#include <string>
#include <map>
#include <vector>

using namespace std;

map<string, int> output_map{ { ">50K",1 },{ "<=50K",-1 } };
map<string, int> country_map
{
	{ "United-States",1 },
	{ "Cambodia",2 },
	{ "England",3 },
	{ "Puerto-Rico",4 },
	{ "Canada",5 },
	{ "Germany",6 },
	{ "Outlying-US(Guam-USVI-etc)",7 },
	{ "India",8 },
	{ "Japan",9 },
	{ "Greece",10 },
	{ "South",11 },
	{ "China",12 },
	{ "Cuba",13 },
	{ "Iran",14 },
	{ "Honduras",15 },
	{ "Philippines",16 },
	{ "Italy",17 },
	{ "Poland",18 },
	{ "Jamaica",19 },
	{ "Vietnam",20 },
	{ "Mexico",21 },
	{ "Portugal",22 },
	{ "Ireland",23 },
	{ "France",24 },
	{ "Dominican-Republic",25 },
	{ "Laos",26 },
	{ "Ecuador",27 },
	{ "Taiwan",28 },
	{ "Haiti",29 },
	{ "Columbia",30 },
	{ "Hungary",31 },
	{ "Guatemala",32 },
	{ "Nicaragua",33 },
	{ "Scotland",34 },
	{ "Thailand",35 },
	{ "Yugoslavia",36 },
	{ "El-Salvador",37 },
	{ "Trinadad&Tobago",38 },
	{ "Peru",39 },
	{ "Hong",40 },
	{ "Holand-Netherlands",41 }
};
map<string, int> sex_map{ { "Male",1 },{ "Female",2 } };
map<string, int> race_map
{
	{ "White",1 },
	{ "Asian-Pac-Islander",2 },
	{ "Amer-Indian-Eskimo",3 },
	{ "Other",4 },
	{ "Black",5 }
};
map<string, int> relationship_map
{
	{ "Wife",1 },
	{ "Own-child",2 },
	{ "Husband",3 },
	{ "Not-in-family",4 },
	{ "Other-relative",5 },
	{ "Unmarried",6 }
};
map<string, int> occupation_map
{
	{ "Tech-support",1 },
	{ "Craft-repair",2 },
	{ "Other-service",3 },
	{ "Sales",4 },
	{ "Exec-managerial",5 },
	{ "Prof-specialty",6 },
	{ "Handlers-cleaners",7 },
	{ "Machine-op-inspct",8 },
	{ "Adm-clerical",9 },
	{ "Farming-fishing",10 },
	{ "Transport-moving",11 },
	{ "Priv-house-serv",12 },
	{ "Protective-serv",13 },
	{ "Armed-Forces",14 }
};
map<string, int> marital_map
{
	{ "Married-civ-spouse",1 },
	{ "Divorced",2 },
	{ "Never-married",3 },
	{ "Separated",4 },
	{ "Widowed",5 },
	{ "Married-spouse-absent",6 },
	{ "Married-AF-spouse",7 }
};
map<string, int> education_map
{
	{ "Bachelors",1 },
	{ "Some-college",2 },
	{ "11th",3 },
	{ "HS-grad",4 },
	{ "Prof-school",5 },
	{ "Assoc-acdm",6 },
	{ "Assoc-voc",7 },
	{ "9th",8 },
	{ "7th-8th",9 },
	{ "12th",10 },
	{ "Masters",11 },
	{ "1st-4th",12 },
	{ "10th",13 },
	{ "Doctorate",14 },
	{ "5th-6th",15 },
	{ "Preschool",16 }
};
map<string, int> workclass_map
{
	{ "Private",1 },
	{ "Self-emp-not-inc",2 },
	{ "Self-emp-inc",3 },
	{ "Federal-gov",4 },
	{ "Local-gov",5 },
	{ "State-gov",6 },
	{ "Without-pay",7 },
	{ "Never-worked",8 }
};
vector < map<string, int >> cont_attr_map;  //maps for age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week in the same order
map<int, char> attributes_map
{
	{ 0, 'a' },		//age
	{ 1, 'b' },		//workclass
	{ 2, 'c' },		//fnlwgt
	{ 3, 'd' },		//education
	{ 4, 'e' },		//education_num
	{ 5, 'f' },		//marital_status
	{ 6, 'g' },		//occupation
	{ 7, 'h' },		//relationship
	{ 8, 'i' },		//race
	{ 9, 'j' },		//sex
	{ 10, 'k' },	//capital_gain
	{ 11, 'l' },	//capital_loss
	{ 12, 'm' },	//hours_per_week
	{ 13, 'n' }		//native_country
};
map<char, int> attributes_map_ctoi
{
	{ 'a', 0 },		//age
	{ 'b', 1 },		//workclass
	{ 'c', 2 },		//fnlwgt
	{ 'd', 3 },		//education
	{ 'e', 4 },		//education_num
	{ 'f', 5 },		//marital_status
	{ 'g', 6 },		//occupation
	{ 'h', 7 },		//relationship
	{ 'i', 8 },		//race
	{ 'j', 9 },		//sex
	{ 'k', 10 },	//capital_gain
	{ 'l', 11 },	//capital_loss
	{ 'm', 12 },	//hours_per_week
	{ 'n', 13 }		//native_country
};
map<char, string> output_map_ctos
{
	{ '+', ">50K" },
	{ '-', "<=50K" }
};