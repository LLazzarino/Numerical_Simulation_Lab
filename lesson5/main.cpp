/****************************************************************
*****************************************************************
  Ex 5 _____  Numerical Simulation Lab _____  Lorenzo Lazzarino
*****************************************************************
*****************************************************************/
//for this exercise the seed used was 0000 0000 0000 0001

#include <iostream>
#include <fstream>
#include <string>
#include "include/random.h"
#include "include/utilities.h"
#include "include/metropolis.h"
#include <armadillo>

using namespace std;
using namespace arma;

int main(){

  //***** ex05 *****//
  //***** Metropolis sampling of hydrogen orbitals *****//

  // Define and construct the random generator
  Random rnd = Random("src/generator/Primes","src/generator/seed.in");

  // Initialize useful parameters for the simulation
  int Nsteps = 1e6;
  int NequilSteps = 0;    // Thermalization steps are not needed for the first simulations thanks to good starting position

  // Define lambda functions to feed to the Metropolis class. \
  These are random number generator linked to the same instance of Random
  auto unifStep = [&rnd](double x) { return rnd.UnifStep(x); };       // Generate a random uniform number between -x and x
  auto gaussStep = [&rnd](double x) { return rnd.GaussStep(x); };     // Generate a random Gaussian number with mean 0 and sigma x
  auto unif = [&rnd]() { return rnd.Rannyu(); };                      // Generate a random uniform number between 0 and 1

  // Define lambda functions to feed to the Metropolis class. \
  these two are the analytic functions used as target tridimensional probability distribution functions \
  x is a 3D cartesian position vector while the function returns a double, the probability
  auto density100 = [](const vec& x) {
    double r = sqrt( x(0)*x(0) + x(1)*x(1) + x(2)*x(2) );
    return exp(-2*r)/M_PI;
  };
  auto density210 = [](const vec& x) {
    double r = sqrt( x(0)*x(0) + x(1)*x(1) + x(2)*x(2) );
    return exp(-r) * x(2)*x(2) / 32. / M_PI;
  };

  //*** Sample 1s and 2p orbitals with uniform steps ***//

  // Define important parameters and initialize Metropolis class instances for the sampling procedure
  vec start = {0.,0.,0.};
  double width = 1.3;
  Metropolis orb1s_unif(start, Nsteps, NequilSteps, width, unifStep, density100, unif);

  start = {0.,0.,5.};
  width = 3.2;
  Metropolis orb2p_unif(start, Nsteps, NequilSteps, width, unifStep, density210, unif);

  mat sampled_1s_unif(3, Nsteps, fill::zeros);        // Create a matrix to hold the sampled PDF
  string fileName = "acceptance1s.txt";               // Choose a file name to hold the accepted and attempted counters
  sampled_1s_unif = orb1s_unif.Sample(fileName);      // Sample the 1s orbital with uniform steps

  mat sampled_2p_unif(3, Nsteps, fill::zeros);        // Create a matrix to hold the sampled PDF
  fileName = "acceptance2p.txt";                      // Choose a file name to hold the accepted and attempted counters
  sampled_2p_unif = orb2p_unif.Sample(fileName);      // Sample the 2p orbital with uniform steps

  // Output the results
  string title = "1s_unif_sample";
  vector<string> column_names = {"x", "y", "z"};
  vector<vec> data_vectors = {sampled_1s_unif.row(0).t(), sampled_1s_unif.row(1).t(), sampled_1s_unif.row(2).t()};
  write_data_file(title, column_names, data_vectors);
  title = "2p_unif_sample";
  data_vectors = {sampled_2p_unif.row(0).t(), sampled_2p_unif.row(1).t(), sampled_2p_unif.row(2).t()};
  write_data_file(title, column_names, data_vectors);

  // Calculate the average and error on the radius of the orbital via blocking average
  int Nblocks=100;

  vec sampled_r_1s_unif(Nsteps, fill::zeros);         // Structure to hold the radius estimator
  for(int i=0;i<Nsteps;i++){
    // Populate the radius estimator with the distance of the walker from the origin
    sampled_r_1s_unif(i) = sqrt( sampled_1s_unif(0,i)*sampled_1s_unif(0,i) +
                                 sampled_1s_unif(1,i)*sampled_1s_unif(1,i) +
                                 sampled_1s_unif(2,i)*sampled_1s_unif(2,i) );
  }

  vec r_1s_unif(Nblocks, fill::zeros);           // Structure to hold the average value calculated via blocking method
  vec r_1s_unif_error(Nblocks, fill::zeros);     // Structure to hold the error calculated via blocking method
  blockAverage(sampled_r_1s_unif, Nblocks, r_1s_unif, r_1s_unif_error);

  vec sampled_r_2p_unif(Nsteps, fill::zeros);         // Structure to hold the radius estimator
  for(int i=0;i<Nsteps;i++){
    // Populate the radius estimator with the distance of the walker from the origin
    sampled_r_2p_unif(i) = sqrt( sampled_2p_unif(0,i)*sampled_2p_unif(0,i) +
                                 sampled_2p_unif(1,i)*sampled_2p_unif(1,i) +
                                 sampled_2p_unif(2,i)*sampled_2p_unif(2,i) );
  }

  vec r_2p_unif(Nblocks, fill::zeros);           // Structure to hold th average value calculated via blocking method
  vec r_2p_unif_error(Nblocks, fill::zeros);     // Structure to hold the error calculated via blocking method
  blockAverage(sampled_r_2p_unif, Nblocks, r_2p_unif, r_2p_unif_error);

  // Output the results
  title = "radii_1s_2p_unif";
  column_names = {"1s","error","2p","error"};
  data_vectors = {r_1s_unif, r_1s_unif_error, r_2p_unif, r_2p_unif_error};
  write_data_file(title, column_names, data_vectors);

  //*** Autocorrelation of the simulations ***//
  /* Via FFT */

  // In order to evaluate power spectrum density and autocorrelation via discrete fourier transform one needs to have a process with zero mean and variance one
  vec sampled_r2_1s_unif(Nsteps, fill::zeros);         // Structure to hold the variance estimator
  for(int i=0;i<Nsteps;i++){
    // Populate the variance estimator
    sampled_r2_1s_unif(i) = (sampled_r_1s_unif(i) - r_1s_unif(Nblocks-1)) * (sampled_r_1s_unif(i) - r_1s_unif(Nblocks-1));
  }
  vec variance_1s_unif = blockAverage(sampled_r2_1s_unif, Nblocks);     // Evaluate variance average and error \
  Now the stochastic process associated with the radius can be "normalized" (=> mean 0 and variance 1)
  vec sampled_r_1s_unif_normalized = ( sampled_r_1s_unif - r_1s_unif(Nblocks-1) ) / sqrt(variance_1s_unif(0));

  vec sampled_r2_2p_unif(Nsteps, fill::zeros);         // Structure to hold the variance estimator
  for(int i=0;i<Nsteps;i++){
    // Populate the variance estimator
    sampled_r2_2p_unif(i) = (sampled_r_2p_unif(i) - r_2p_unif(Nblocks-1)) * (sampled_r_2p_unif(i) - r_2p_unif(Nblocks-1));
  }
  vec variance_2p_unif = blockAverage(sampled_r2_2p_unif, Nblocks);     // Evaluate variance average and error \
  Now the stochastic process associated with the radius can be "normalized" (=> mean 0 and variance 1)
  vec sampled_r_2p_unif_normalized = ( sampled_r_2p_unif - r_2p_unif(Nblocks-1) ) / sqrt(variance_2p_unif(0));

  // Perform FFT with Armadillo built-in functions \
  1) Evaluate the DFT of the stochastic process. It is in general complex, thus we use complex vectors cx_vec
  cx_vec r_1s_unif_fft = fft(sampled_r_1s_unif_normalized);
  cx_vec r_2p_unif_fft = fft(sampled_r_2p_unif_normalized); //\
  2) Evaluate the power spectrum density (PSD) as the modulus square of the process in reciprocal space divided by the number of elements
  cx_vec PSD_1s_unif = r_1s_unif_fft % conj(r_1s_unif_fft) / r_1s_unif_fft.n_elem;
  cx_vec PSD_2p_unif = r_2p_unif_fft % conj(r_2p_unif_fft) / r_2p_unif_fft.n_elem; //\
  3) The autocorrelation is the inverse Fourier transform of the PSD. The imaginary part is zero or negligible
  vec autocorrelation_1s_unif_FFT = real( ifft(PSD_1s_unif) );
  vec autocorrelation_2p_unif_FFT = real( ifft(PSD_2p_unif) );

  // Output the results
  title = "autocorrelationsViaFFT";
  column_names = {"1s","2p"};
  data_vectors = {autocorrelation_1s_unif_FFT, autocorrelation_2p_unif_FFT};
  write_data_file(title, column_names, data_vectors);

  /* Via definition of autocorrelation (brute force) */

  // Since calculation the autocorrelation with a brute force algorithm has a computational cost of n^2, n being the data size \
  We evaluate it just for $\tau \in [0,100]$
  int lagmax = 100;   // Maximum value of the lag tau

  // Structures to hold the autocorrelation
  vec autocorrelation_1s_unif_definition(lagmax, fill::zeros);
  vec autocorrelation_2p_unif_definition(lagmax, fill::zeros);

  for(int lag = 0; lag < lagmax ;lag++){
    // Evaluate the autocorrelation "lagmax" times
    for(int i=0; i< Nsteps - lag; i++){
      // Use all the points in the dataset to average the correlation of the normalized process with itself at a shifted time
      autocorrelation_1s_unif_definition(lag) +=  sampled_r_1s_unif_normalized(i) * sampled_r_1s_unif_normalized(i+lag);
      autocorrelation_2p_unif_definition(lag) +=  sampled_r_2p_unif_normalized(i) * sampled_r_2p_unif_normalized(i+lag);
    }
    autocorrelation_1s_unif_definition(lag) = autocorrelation_1s_unif_definition(lag) / (Nsteps - lag);
    autocorrelation_2p_unif_definition(lag) = autocorrelation_2p_unif_definition(lag) / (Nsteps - lag);
  }

  // Output the results
  title = "autocorrelationsViaDefinition";
  column_names = {"1s","2p"};
  data_vectors = {autocorrelation_1s_unif_definition, autocorrelation_2p_unif_definition};
  write_data_file(title, column_names, data_vectors);

  //*** Simulations with walker starting out of equilibrium ***//
  // We investigate what happens when the walker starts out of equilibrium (in an extremely unlikely position for the target PDF)
  // We limit the investigation to the 1s orbital

  // 1) Start away from the origin
  // Set a new starting point
  start = {0.,0.,20.};
  orb1s_unif.SetStart(start);
  // Set a new step width (in case it is needed)
  width = 1.3;
  orb1s_unif.SetWidth(width);
  // Set a number of thermalization steps to discard
  NequilSteps = 0;
  orb1s_unif.SetThermalization(NequilSteps);        // No steps will be discarded

  fileName = "acceptance1s_offstart.txt";
  sampled_1s_unif = orb1s_unif.Sample(fileName);

  // Output the results
  title = "1s_unif_sample_offstart";
  column_names = {"x", "y", "z"};
  data_vectors = {sampled_1s_unif.row(0).t(), sampled_1s_unif.row(1).t(), sampled_1s_unif.row(2).t()};
  write_data_file(title, column_names, data_vectors);

  // 2) Start much further off 
  // Set a new starting point far off the origin
  start = {0.,0.,20000.};
  orb1s_unif.SetStart(start);
  // Set a new step width (in case it is needed)
  width = 130;
  orb1s_unif.SetWidth(width);
  // Set a number of thermalization steps to discard
  NequilSteps = 0;
  orb1s_unif.SetThermalization(NequilSteps);        // No steps will be discarded

  fileName = "acceptance1s_offstart2.txt";
  sampled_1s_unif = orb1s_unif.Sample(fileName);

  // Output the results
  title = "1s_unif_sample_offstart2";
  column_names = {"x", "y", "z"};
  data_vectors = {sampled_1s_unif.row(0).t(), sampled_1s_unif.row(1).t(), sampled_1s_unif.row(2).t()};
  write_data_file(title, column_names, data_vectors);

  // 3) Thermalization 
  //Set a new starting point
  start = {0.,0.,20.};
  orb1s_unif.SetStart(start);
  // Set a new step width (in case it is needed)
  width = 1.3;
  orb1s_unif.SetWidth(width);
  // Set a number of thermalization steps to discard
  NequilSteps = Nsteps / 2;
  orb1s_unif.SetThermalization(NequilSteps);       // Many steps will be discarded in order to assure equilibrium and forget the starting point

  fileName = "acceptance1s_postTherm.txt";
  sampled_1s_unif = orb1s_unif.Sample(fileName);

  // Output the results
  title = "1s_unif_sample_postTherm";
  column_names = {"x", "y", "z"};
  data_vectors = {sampled_1s_unif.row(0).t(), sampled_1s_unif.row(1).t(), sampled_1s_unif.row(2).t()};
  write_data_file(title, column_names, data_vectors);

  //*** Sample 1s and 2p orbitals with Gaussian steps ***//
  // Repeat the simulation with a different metropolis move to see whether the results are the same

  // Define important parameters and initialize Metropolis class instances for the sampling procedure
  start = {0.,0.,0.};
  width = 0.65;
  NequilSteps = Nsteps / 2;
  Metropolis orb1s_gauss(start, Nsteps, NequilSteps, width, gaussStep, density100, unif);

  start = {0.,0.,5.};
  width = 1.6;
  NequilSteps = Nsteps / 2;
  Metropolis orb2p_gauss(start, Nsteps, NequilSteps, width, gaussStep, density210, unif);

  mat sampled_1s_gauss(3, Nsteps, fill::zeros);           // Create a matrix to hold the sampled PDF
  fileName = "acceptance1s_gauss.txt";                    // Choose a file name to hold the accepted and attempted counters
  sampled_1s_gauss = orb1s_gauss.Sample(fileName);        // Sample the 1s orbital with Gaussian steps

  mat sampled_2p_gauss(3, Nsteps, fill::zeros);           // Create a matrix to hold the sampled PDF
  fileName = "acceptance2p_gauss.txt";                    // Choose a file name to hold the accepted and attempted counters
  sampled_2p_gauss = orb2p_gauss.Sample(fileName);        // Sample the 2p orbital with Gaussian steps

  // Calculate the average and error on the radius of the orbital via blocking average
  vec sampled_r_1s_gauss(Nsteps, fill::zeros);         // Structure to hold the radius estimator
  for(int i=0;i<Nsteps;i++){
    // Populate the radius estimator with the distance of the walker from the origin
    sampled_r_1s_gauss(i) = sqrt( sampled_1s_gauss(0,i)*sampled_1s_gauss(0,i) +
                                 sampled_1s_gauss(1,i)*sampled_1s_gauss(1,i) +
                                 sampled_1s_gauss(2,i)*sampled_1s_gauss(2,i) );
  }

  vec r_1s_gauss(Nblocks, fill::zeros);           // Structure to hold the average value calculated via blocking method
  vec r_1s_gauss_error(Nblocks, fill::zeros);     // Structure to hold the error calculated via blocking method
  blockAverage(sampled_r_1s_gauss, Nblocks, r_1s_gauss, r_1s_gauss_error);

  vec sampled_r_2p_gauss(Nsteps, fill::zeros);         // Structure to hold the radius estimator
  for(int i=0;i<Nsteps;i++){
    // Populate the radius estimator with the distance of the walker from the origin
    sampled_r_2p_gauss(i) = sqrt( sampled_2p_gauss(0,i)*sampled_2p_gauss(0,i) +
                                 sampled_2p_gauss(1,i)*sampled_2p_gauss(1,i) +
                                 sampled_2p_gauss(2,i)*sampled_2p_gauss(2,i) );
  }

  vec r_2p_gauss(Nblocks, fill::zeros);           // Structure to hold the average value calculated via blocking method
  vec r_2p_gauss_error(Nblocks, fill::zeros);     // Structure to hold the error calculated via blocking method
  blockAverage(sampled_r_2p_gauss, Nblocks, r_2p_gauss, r_2p_gauss_error);

  // Output the results
  title = "radii_1s_2p_gauss";
  column_names = {"1s","error","2p","error"};
  data_vectors = {r_1s_gauss, r_1s_gauss_error, r_2p_gauss, r_2p_gauss_error};
  write_data_file(title, column_names, data_vectors);

  // Return arrival seed
  rnd.SaveSeed();
  return 0;
}

/****************************************************************
*****************************************************************
  Ex 5 _____  Numerical Simulation Lab _____  Lorenzo Lazzarino
*****************************************************************
*****************************************************************/
