/****************************************************************
*****************************************************************
    _/        _/        Numerical Simulation Laboratory
   _/        _/        Physics Department
  _/        _/        Universita' degli Studi di Milano
 _/        _/        Student Lorenzo Lazzarino
_/_/_/_/  _/_/_/_/  email: lorenzo.lazzarino98@gmail.com
*****************************************************************
*****************************************************************/

//this library contains miscellaneous useful code

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <armadillo>
#include <iomanip>
#include <complex>

using namespace std;
using namespace arma;

//write a result file in a table format
void write_data_file(string, vector<string>, vector<vec>);

//calculate the average and error for a cumulative number of data blocks
void blockAverage(vec data, int Nblocks, vec& average, vec& error);

//calculate the average and error of the reciprocals for a cumulative number of data blocks
void blockAverageReciprocal(vec data, int Nblocks, vec& average, vec& error);

//calculate the average and error from a dataset by dividing it in blocks
vec blockAverage(vec data, int Nblocks);


/****************************************************************
*****************************************************************
    _/        _/        Numerical Simulation Laboratory
   _/        _/        Physics Department
  _/        _/        Universita' degli Studi di Milano
 _/        _/        Student Lorenzo Lazzarino
_/_/_/_/  _/_/_/_/  email: lorenzo.lazzarino98@gmail.com
*****************************************************************
*****************************************************************/
