/****************************************************************
*****************************************************************
  Ex 3 _____  Numerical Simulation Lab _____  Lorenzo Lazzarino
*****************************************************************
*****************************************************************/
//for this exercise the seed used was 0000 0000 0000 0001

#include <iostream>
#include <fstream>
#include <string>
#include "include/random.h"
#include "include/utilities.h"
#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char *argv[]){

  //***** ex03 *****//
  //*** plain vanilla option pricing ***//

  //* ex03.1 *//
  //* by sampling directly the final asset price *//

  //define and construct the random generator
  Random rnd = Random("src/generator/Primes","src/generator/seed.in");

  int sampleSize = 1e5; //number of random extractions
  int Nblocks = 1e2;

  //market parameters
  double s0=100;                  //initial asset value
  double delivery_time = 1.;      //time between option signature and option call
  double strike_price = 100;      //prescribed price to pay at delivery
  double interest = 0.1;          //rate of asset growth guaranteed by the Bank
  double volatility = 0.25;       //strength of the statistical price fluctuations

  //initialize vectors to hold temporary and important values
  vec gaussSample(sampleSize, fill::zeros);
  vec finalAssetPrices(sampleSize, fill::zeros);
  vec callOptionPricings(sampleSize, fill::zeros);
  vec putOptionPricings(sampleSize, fill::zeros);
  vec callPrice(Nblocks, fill::zeros);
  vec callError(Nblocks, fill::zeros);
  vec putPrice(Nblocks, fill::zeros);
  vec putError(Nblocks, fill::zeros);

  //sample final asset prices at delivery time
  for(int i=0; i<sampleSize; i++){
    gaussSample(i) = rnd.Gauss(0, sqrt(delivery_time));
  }
  finalAssetPrices = s0 * exp( (interest - 0.5 * volatility*volatility) * delivery_time + volatility * gaussSample);

  //evaluate call[put] options prices as the max of (0,final asset price - strike price) [(0,strike price - final asset price)]
  vec SminusK = finalAssetPrices - strike_price;
  vec KminusS = -1 * SminusK;

  //uword is a data type that is part of the Armadillo C++ linear algebra library. It is an alias for unsigned long long and is used
  //to provide a consistent data type for indices and dimensions in its linear algebra operations
  uword i = 0;
  //the for loop applies the "max" algorithm as a lambda function to each element in the vecs
  //The first two arguments to std::for_each define the range to be operated on. first and last are iterators that specify the beginning and end of the range, respectively.
  //The third argument is a unary function object that will be applied to each element in the range.
  for_each(SminusK.begin(), SminusK.end(), [&](double x) { callOptionPricings(i++) = x > 0 ? x : 0.0; });
  i = 0;
  for_each(KminusS.begin(), KminusS.end(), [&](double x) { putOptionPricings(i++) = x > 0 ? x : 0.0; });

  //apply discount to pricings based on Bank interest
  callOptionPricings = callOptionPricings * exp(-interest * delivery_time);
  putOptionPricings = putOptionPricings * exp(-interest * delivery_time);

  //use the blocking average method to evaluate average and error
  blockAverage(callOptionPricings, Nblocks, callPrice, callError);
  blockAverage(putOptionPricings, Nblocks, putPrice, putError);

  //output the results
  string title = "ex03.1";
  vector<string> column_names = {"callPrice", "callError", "putPrice", "putError"};
  vector<vec> data_vectors = {callPrice, callError, putPrice, putError};
  write_data_file(title, column_names, data_vectors);

  //* ex03.2 *//
  //* by sampling the discretized Geometric Brownian Motion(GBM) path of the asset price *//

  //divide the delivery time into equal steps
  int Nsteps= 100;
  double timeStep= delivery_time / Nsteps;

  //restore the useful vectors
  finalAssetPrices = zeros<vec>(sampleSize);
  callOptionPricings = zeros<vec>(sampleSize);
  putOptionPricings = zeros<vec>(sampleSize);
  callPrice = zeros<vec>(Nblocks);
  callError = zeros<vec>(Nblocks);
  putPrice = zeros<vec>(Nblocks);
  putError = zeros<vec>(Nblocks);

  //sample final asset prices at delivery time with the Geometric Brownian Motion
  finalAssetPrices += s0;
  for(int j=0; j<sampleSize; j++){
    for(int i=0; i<Nsteps; i++){
      finalAssetPrices(j) = finalAssetPrices(j) * exp((interest - 0.5*volatility*volatility)*timeStep + volatility * rnd.Gauss(0, 1) * sqrt(timeStep));
    }
  }

  //evaluate call[put] options prices as the max of (0,final asset price - strike price) [(0,strike price - final asset price)]
  SminusK = finalAssetPrices - strike_price;
  KminusS = -1 * SminusK;
  i = 0;
  for_each(SminusK.begin(), SminusK.end(), [&](double x) { callOptionPricings(i++) = x > 0 ? x : 0.0; });
  i = 0;
  for_each(KminusS.begin(), KminusS.end(), [&](double x) { putOptionPricings(i++) = x > 0 ? x : 0.0; });

  //apply discount to pricings based on Bank interest
  callOptionPricings = callOptionPricings * exp(-interest * delivery_time);
  putOptionPricings = putOptionPricings * exp(-interest * delivery_time);

  //use the blocking average method to evaluate average and error
  blockAverage(callOptionPricings, Nblocks, callPrice, callError);
  blockAverage(putOptionPricings, Nblocks, putPrice, putError);

  //output the results
  title = "ex03.2";
  data_vectors = {callPrice, callError, putPrice, putError};
  write_data_file(title, column_names, data_vectors);

  //return the arrival seed
  rnd.SaveSeed();
  return 0;
}

/****************************************************************
*****************************************************************
  Ex 3 _____  Numerical Simulation Lab _____  Lorenzo Lazzarino
*****************************************************************
*****************************************************************/
