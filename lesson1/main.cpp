/****************************************************************
*****************************************************************
  Ex 1 _____  Numerical Simulation Lab _____  Lorenzo Lazzarino
*****************************************************************
*****************************************************************/
//for this exercise the seed used was 0000 0000 0000 0007

#include <iostream>
#include <fstream>
#include <string>
#include "include/random.h"
#include "include/utilities.h"
#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char *argv[]){

  //***** ex01 *****//
  //*** pseudo-random number generator (prng) ***//

  int Nthrows = 1e6;
  int Nblocks = 1e2;

  //* ex01.11 *//
  vec throws(Nthrows, fill::zeros);
  vec result(Nblocks, fill::zeros);
  vec error(Nblocks, fill::zeros);

  //define and construct the random generator
  Random rnd = Random("src/generator/Primes","src/generator/seed.in");

  //generate uniform random data that will be used among exercises
  for(int i=0; i<Nthrows; i++){
    throws(i) = rnd.Rannyu();
  }

  //evaluate average and uncertainty of uniform prng
  blockAverage(throws, Nblocks, result, error);

  //output data results
  string title = "ex01.11";
  vector<string> column_names = {"averages", "errors"};
  vector<vec> data_vectors = {result, error};
  write_data_file(title, column_names, data_vectors);

  //* ex01.12 *//
  //predispose data structures to evaluate the variance of the uniform prng
  vec secondmoment(Nblocks, fill::zeros);
  vec sigmas(Nblocks, fill::zeros);
  vec sigmaerror(Nblocks, fill::zeros);

  secondmoment = (throws - .5) % (throws - .5);

  //evaluate average and uncertainty of the uniform prng variance
  blockAverage(secondmoment, Nblocks, sigmas, sigmaerror);

  //output data results
  title = "ex01.12";
  column_names = {"sigmas", "errors"};
  data_vectors = {sigmas, sigmaerror};
  write_data_file(title, column_names, data_vectors);

  //* ex01.13 *//
  //divide [0,1) in "Nbins" bins
  int Nbins = 1e2;
  double binSize = 1./Nbins;
  vec occurrencies(Nbins, fill::zeros);
  //chi squared will be evaluated "chiEvaluations" times with "realizations" random extractions each time
  int realizations = 1e4;
  int chiEvaluations = 100;
  vec chiSquared(chiEvaluations, fill::zeros);

  //calculate the expected events given the number of generated random numbers
  double expEvents = realizations / Nbins;

  for(int j = 0; j < chiEvaluations; j++ ){
    //increment the occurrencies vector element with the index corresponding to the respective bin
    for(int i = j* realizations ; i < (j+1) * realizations; i++){
      //map the generated random numbers to the bin index via multiplication by 1/binSize
      int binIndex = int(throws(i)/binSize);
      occurrencies(binIndex)+=1;
    }
    //calculate chi squared for each cumulative sample
    chiSquared(j) = dot((occurrencies - expEvents),(occurrencies - expEvents)) / expEvents;
    occurrencies = zeros<vec>(Nbins);
  }

  //output data results
  title = "ex01.13";
  column_names = {"Chi Squared"};
  data_vectors = {chiSquared};
  write_data_file(title, column_names, data_vectors);

  //***** ex02 *****//
  //*** inverse of the cumulative distribution and CLT ***//

  vec xi(realizations, fill::zeros);
  data_vectors.clear();

  //* ex01.21 *//

  //generate 100 samples of extractions in number "realizations" from an uniform distribution
  for(int j=0;j<100;j++){
    for(int i=0;i<realizations;i++){
      //cumulatively sum the samples to test the central limit theorem
      xi(i)+=rnd.Rannyu();
    }
    //save some significant cumulative samples
    if(j==0 || j==1 || j==9 || j==99){
      data_vectors.push_back(xi/(j+1));
    }
  }
  //output saved data
  title = "ex01.21";
  column_names = {"unif1","unif2","unif10","unif100"};
  write_data_file(title, column_names, data_vectors);

  //* ex01.22 *//
  xi = zeros<vec>(realizations);
  data_vectors.clear();
  double gamma = 1.;

  //generate 100 samples of extractions in number "realizations" from an exponential distribution
  for(int j=0;j<100;j++){
    for(int i=0;i<realizations;i++){
      //cumulatively sum the samples to test the central limit theorem
      xi(i)+=rnd.Exp(gamma);
    }
    //save some significant cumulative samples
    if(j==0 || j==1 || j==9 || j==99){
      data_vectors.push_back(xi/(j+1));
    }
  }
  //output saved data
  title = "ex01.22";
  column_names = {"exp1","exp2","exp10","exp100"};
  write_data_file(title, column_names, data_vectors);

  //* ex01.23 *//
  xi = zeros<vec>(realizations);
  data_vectors.clear();
  double mean = 0.;
  double Gamma = 1.;

  //generate 100 samples of extractions in number "realizations" from a Cauchy-Lorentz distribution
  for(int j=0;j<100;j++){
    for(int i=0;i<realizations;i++){
      xi(i)+=rnd.Lorentz(mean,Gamma);
    }
    if(j==0 || j==1 || j==9 || j==99){
      data_vectors.push_back(xi/(j+1));
    }
  }
  //output saved data
  title = "ex01.23";
  column_names = {"Lore1","Lore2","Lore10","Lore100"};
  write_data_file(title, column_names, data_vectors);

  ///***exercise 1.3***///
  //*** Buffon experiment: estimate pi via random throws of a needle ***//

  int NBuffonThrows = 1e8;
  Nblocks=1e2;
  vec BuffonHits(NBuffonThrows, fill::zeros);
  vec pi(Nblocks, fill::zeros);
  vec pierror(Nblocks, fill::zeros);
  //the NeedleLenght shall be lesser than the Distance
  double Distance = 1.;
  double NeedleLenght = 0.6;

  //to simulate the needle throw, extract the center of mass position along a segment and the needle orientation
  for(int i = 0; i< NBuffonThrows; i++){
    double CMpos = rnd.Rannyu(0.,Distance);
    double x;
    double y;
    double normSquared;
    do{
      //the needle orientation is generated through an accept/reject method in order not to use pi and trigonometric functions
      x = rnd.Rannyu();
      y = rnd.Rannyu();
      normSquared = x*x+y*y;
      //reject the orientation vector if it doesn't fall in the unitary (quarter-)circle
    }while(normSquared > 1.);
    double sinTheta = y/sqrt(normSquared);

    if( (CMpos + sinTheta*NeedleLenght/2 >= Distance) or (CMpos - sinTheta*NeedleLenght/2 <= 0.)){
      //the vector BuffonHits contains 1 at the indices where the needle touches the grid
      BuffonHits(i) = 1.;
    }
  }

  BuffonHits = BuffonHits *Distance / (2*NeedleLenght);
  //calculate pi and its error through a block average applied to the reciprocals of each block average
  //i.e.  you extract pi from each block and work from there
  blockAverageReciprocal(BuffonHits, Nblocks, pi, pierror);

  //output the results
  title = "ex01.31";
  column_names = {"averages", "errors"};
  data_vectors = {pi, pierror};
  write_data_file(title, column_names, data_vectors);

  //let us draw a comparison with the propagation of errors theory
  //calculate 1/pi and its error through a block average
  blockAverage(BuffonHits, Nblocks, pi, pierror);

  //estimate pi and its error via propagation of errors
  //---> in this case the results are equivalent, however propagation of errors is never the way
  pierror = pierror/(pi % pi);
  pi = 1/pi;

  //output the results
  title = "ex01.31.poe";
  column_names = {"averages", "errors"};
  data_vectors = {pi, pierror};
  write_data_file(title, column_names, data_vectors);

  //return the arrival seed
  rnd.SaveSeed();
  return 0;
}

/****************************************************************
*****************************************************************
  Ex 1 _____  Numerical Simulation Lab _____  Lorenzo Lazzarino
*****************************************************************
*****************************************************************/
