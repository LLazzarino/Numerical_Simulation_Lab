/****************************************************************
*****************************************************************
  Ex 2 _____  Numerical Simulation Lab _____  Lorenzo Lazzarino
*****************************************************************
*****************************************************************/
//for this exercise the seed used was 0000 0000 0000 0008

#include <iostream>
#include <fstream>
#include <string>
#include "include/random.h"
#include "include/utilities.h"
#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char *argv[]){

  //***** ex02.1 *****//
  //*** monte carlo integration ***//

  //define and construct the random generator
  Random rnd = Random("src/generator/Primes","src/generator/seed.in");

  //initialize useful data structures and parameters
  int sampleSize = 1e6;   //total number of times that the integral estimator is evaluated
  int Nblocks = 1e2;      // number of blocks in blocking method

  //the integral will be evaluated with 4 different sampling strategies

  // (1) uniform sampling
  vec uniformSample(sampleSize, fill::zeros);
  vec g_x(sampleSize, fill::zeros);         //integral estimator
  vec uniform(Nblocks, fill::zeros);        //contains the integral value calculated via blocking method
  vec unif_error(Nblocks, fill::zeros);     //contains the integral error calculated via blocking method

  for(int i=0; i<sampleSize; i++){
    //sample uniform numbers between 0,1 and evaluate the integral estimator
    uniformSample(i) = rnd.Rannyu();
    g_x(i)= M_PI_2 * cos(M_PI_2 * uniformSample(i) );
  }

  blockAverage(g_x, Nblocks, uniform, unif_error);

  // (2) importance sampling: second order polynomial pdf with accept/reject technique
  vec parabSample(sampleSize, fill::zeros);
  vec parable(Nblocks, fill::zeros);        //contains the integral value calculated via blocking method
  vec parab_error(Nblocks, fill::zeros);    //contains the integral error calculated via blocking method

  for(int i=0; i<sampleSize; i++){
    //sample numbers between 0,1 with pdf: (pi/2)*(1-x^2) and evaluate the integral estimator
    double x;
    double y;
    do{
      //accept reject technique
      x = rnd.Rannyu();
      y = rnd.Rannyu(0,M_PI_2);
    }while(y>M_PI_2*(1-x*x));
    parabSample(i) = x;
    g_x(i) =  M_PI/3. * cos(M_PI_2 * parabSample(i)) / (1-parabSample(i)*parabSample(i));
  }

  blockAverage(g_x, Nblocks, parable, parab_error);

  // (3) importance sampling: exponential pdf with inversion technique
  vec expSample(sampleSize, fill::zeros);
  vec exponential(Nblocks, fill::zeros);    //contains the integral value calculated via blocking method
  vec exp_error(Nblocks, fill::zeros);      //contains the integral error calculated via blocking method

  for(int i=0; i<sampleSize; i++){
    //sample numbers between 0,1 with a truncated exponential pdf and evaluate the integral estimator
    double s=rnd.Rannyu();
    expSample(i) = -log(1 - s*(1-exp(-1)) );
    g_x(i) = M_PI_2 * cos(M_PI_2 * expSample(i)) / exp(-expSample(i)) *(1-exp(-1));
  }

  blockAverage(g_x, Nblocks, exponential, exp_error);

  // (4) antithetic variates from uniform sample
  vec uniformSample2(sampleSize, fill::zeros);
  vec antithetic(Nblocks, fill::zeros);   //contains the integral value calculated via blocking method
  vec anti_error(Nblocks, fill::zeros);   //contains the integral error calculated via blocking method

  for(int i=0; i<sampleSize; i++){
    //sample uniform numbers between 0,1 and evaluate the integral estimator via antithetic variates method
    uniformSample2(i) = rnd.Rannyu();
    g_x(i)= 0.5* ( M_PI_2 * cos(M_PI_2 * uniformSample(i)) + M_PI_2 * cos(M_PI_2 * (1-uniformSample(i))) );
  }

  blockAverage(g_x, Nblocks, antithetic, anti_error);

  //output the results
  string title = "ex02.1";
  vector<string> column_names = {"uniform", "errors", "parable","errors","exponential", "errors","antithetic","errors"};
  vector<vec> data_vectors = {uniform, unif_error, parable, parab_error, exponential, exp_error, antithetic, anti_error};
  write_data_file(title, column_names, data_vectors);


  //***** ex02.2 *****//
  //*** random walk(RW) sampling ***//

  //define useful parameters
  int Nsteps = 1e2;
  int realizations = 1e5;
  int ndim = 3;

  // (1) discrete RW
  mat discrete_RW(realizations, ndim, fill::zeros);   //last step in the RW
  vec RMSD(Nsteps, fill::zeros);                      //root mean square distance for each step
  vec RMSDerror(Nsteps, fill::zeros);                 //root mean square distance error for each step
  vec moduliSquared(realizations, fill::zeros);       //contains the distance from the origin of the last step
  vec RMSDstep(2, fill::zeros);                       //rmsd and its error at each step calculated via blocking method

  //perform the discrete random walk
  for(int j = 0; j< Nsteps; j++){ // loop over the number of time steps
    for(int i = 0; i< realizations; i++){ // loop over the number of realizations
      double randomUnif = rnd.Rannyu(0.,6.); // generate a random uniform number between 0 and 6
      int direction = int(randomUnif)/2; // determine the direction to move in based on the random number
      int step = (int(randomUnif)%2)*2 - 1; // determine the size of the step to take based on the random number

      // update the position of the random walker in the specified direction
      discrete_RW(i,direction) += step;

      // calculate the distance from the origin of the random walker's position vector
      for(int k=0; k < ndim; k++){
        moduliSquared(i) += discrete_RW(i,k)*discrete_RW(i,k); // sum up the squares of the components of the random walker's position vector
        //alternative version: moduliSquared(i) = dot(continuous_RW.row(i), continuous_RW.row(i));
      }
    }

    // calculate the root mean squared displacement of the random walker over the current time step
    RMSDstep = blockAverageRMS(moduliSquared, Nblocks);

    // store the RMSD and its error in arrays for later analysis
    RMSD(j) = RMSDstep(0);
    RMSDerror(j) = RMSDstep(1);

    // reset the moduliSquared vector to zero for the next time step
    moduliSquared = zeros<vec>(realizations);
  }

  // (2) continuous RW
  mat continuous_RW(realizations, ndim, fill::zeros);   //last step in the RW
  vec cRMSD(Nsteps, fill::zeros);                       //root mean square distance for each step
  vec cRMSDerror(Nsteps, fill::zeros);                  //root mean square distance error for each step
  moduliSquared = zeros<vec>(realizations);

  // perform the continuous random walk
  for(int j = 0; j < Nsteps; j++){
      for(int i = 0; i < realizations; i++){
          // generate a random direction for the step using spherical coordinates
          double theta = rnd.Rannyu(0.,M_PI);
          double phi = rnd.Rannyu(0.,2*M_PI);
          vec step = {sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)};

          // move the particle in the selected direction
          continuous_RW.row(i) += step.t();

          // compute the moduli squared of the particle position vector
          for(int k = 0; k < ndim; k++){
              moduliSquared(i) += continuous_RW(i,k)*continuous_RW(i,k);    //moduliSquared(i) = dot(continuous_RW.row(i), continuous_RW.row(i));
          }
      }

      // compute the RMSD and the standard deviation of the RMSD for the current step
      RMSDstep = blockAverageRMS(moduliSquared, Nblocks);
      cRMSD(j) = RMSDstep(0);
      cRMSDerror(j) = RMSDstep(1);

      // reset the vector storing the moduli squared of the particle position vector
      moduliSquared = zeros<vec>(realizations);
  }

  //output the results
  title = "ex02.2";
  column_names = {"discrete_RW","error", "continuous_RW", "error"};
  data_vectors = {RMSD, RMSDerror, cRMSD, cRMSDerror};
  write_data_file(title, column_names, data_vectors);

  //return the arrival seed
  rnd.SaveSeed();
  return 0;
}

/****************************************************************
*****************************************************************
  Ex 2 _____  Numerical Simulation Lab _____  Lorenzo Lazzarino
*****************************************************************
*****************************************************************/
