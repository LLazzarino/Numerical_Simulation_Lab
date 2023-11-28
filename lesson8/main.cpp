/****************************************************************
*****************************************************************
  Ex 8 _____  Numerical Simulation Lab _____  Lorenzo Lazzarino
*****************************************************************
*****************************************************************/
//for this exercise the seed used was 0000 0000 0000 0001

#include <iostream>
#include <fstream>
#include <string>
#include "include/random.h"
#include "include/utilities.h"
#include "include/metropolis.h"
#include "include/VMC.h"
#include <armadillo>

using namespace std;
using namespace arma;

int main(){

  //***** ex08 *****//
  //***** Metropolis sampling of QM Hamiltonians  *****//

  // Define and construct the random generator
  Random rnd = Random("src/generator/Primes","src/generator/seed.in");

  // Initialize useful parameters for the simulation
  int Nsteps = 1e7;
  int NequilSteps = 0;    // Thermalization steps are not needed for the first simulations thanks to good starting position

  // Define lambda functions to feed to the Metropolis class. \
  These are random number generator linked to the same instance of Random
  auto unifStep = [&rnd](double x) { return rnd.UnifStep(x); };       // Generate a random uniform number between -x and x
  auto gaussStep = [&rnd](double x) { return rnd.GaussStep(x); };     // Generate a random Gaussian number with mean 0 and sigma x
  auto unif = [&rnd]() { return rnd.Rannyu(); };                      // Generate a random uniform number between 0 and 1

  //the wavefunction is parameterized by two parameters:
  double mu =  0.817378;       
  double sigma = 0.619481;
  // Define a lambda function to feed to the Metropolis class. \
  this is the wave function squared with fixed mu and sigma we wish to sample to calculate Eloc \
  x is a 1D cartesian position vector while the function returns a double, the probability
  auto density = [&mu, &sigma](const vec& x) {
    double a = (x(0)*x(0) + mu*mu)/(sigma*sigma);
    double b = x(0)*mu/sigma/sigma ;
    return exp(-a) * (1. + cosh(2*b));
  };

  //*** Sample the trial wavefunction with uniform steps ***//

  // Define important parameters and initialize Metropolis class instances for the sampling procedure
  vec start = {0.};
  double width = 1.;
  Metropolis psi2(start, Nsteps, NequilSteps, width, unifStep, density, unif, 0); //last zero is the mode: we feed the actual pdf

  mat sampled_psi2(1, Nsteps, fill::zeros);        // Create a matrix to hold the sampled PDF
  string fileName = "acceptance.txt";               // Choose a file name to hold the accepted and attempted counters
  sampled_psi2 = psi2.Sample(fileName);      // Sample the trial density function with uniform steps

  // Output the results
  string title = "psi2_sample";
  vector<string> column_names = {"x"};
  vector<vec> data_vectors = {sampled_psi2.row(0).t()};
  write_data_file(title, column_names, data_vectors);

  // Calculate the average and error of the local energy sampled as psi2 via blocking average
  int Nblocks=100;

  vec estimated_Eloc(Nsteps, fill::zeros);         // Structure to hold the Eloc
  for(int i=0;i<Nsteps;i++){
    // Calculate the Eloc estimates
    estimated_Eloc(i) = Eloc_calc(sampled_psi2(0,i), mu, sigma);
  }

  vec Eloc(Nblocks, fill::zeros);           // Structure to hold the average value calculated via blocking method
  vec Eloc_error(Nblocks, fill::zeros);     // Structure to hold the error calculated via blocking method
  blockAverage(estimated_Eloc, Nblocks, Eloc, Eloc_error);


  // Output the results
  title = "Energy_trial_wavefunction";
  column_names = {"energy","error"};
  data_vectors = {Eloc, Eloc_error};
  write_data_file(title, column_names, data_vectors);

  //***** Simulated Annealing *****//

  // Define important parameters and initialize Metropolis class instances for the sampling procedure
  
  // (Re)initialization of the metropolis object to evaluate H as loss function for S.A.
  Nsteps = 1e5;
  NequilSteps = 0; 
  Nblocks=100;
  start = {0.};
  width = 1.;
  Metropolis trialWF(start, Nsteps, NequilSteps, width, unifStep, density, unif, 0); //last zero is the mode: we feed the actual pdf
  
  // Create a matrix to hold the sampled wave function density
  mat sampled_trialWF(1, Nsteps, fill::zeros);              

  // Setup a file to contain the errors on the energy evaluation in the simulated annealing \
  They will later need analysis since they don't account for rejected steps \
  though I do not think this error should be fed to the metropolis class, since it's irrelevant for the algorithm and also would steer the class away from its generality
  ofstream out;
  const int wd=14;
  out.open("output_Annealing_energy_errors.txt");
  out << setw(wd) << "energy_errors" << endl;
  
  // Define the starting inverse temperature for the Simulated Annealing
  double beta = 1e10;

  // Define a lambda function to feed to the Metropolis class. \
  this is loss function to perform S.A. -the energy of the system- \
  it is fed as a logpdf to the Metropolis class in _mode == 1 \
  x is a 2D vector in the hyperspace of variational parameters, while the function returns a double, the Boltzmann beta*loss (logpdf)
  auto Boltzmann = [&beta, &mu, &sigma, &trialWF, &sampled_trialWF, &Nsteps, &Nblocks, &out](const vec& x) {
    // Update mu and sigma as they will be proposed by Metropolis WFspace
    mu = x(0);
    sigma = x(1);
    // Estimate the energy for this wavefunction
    sampled_trialWF = trialWF.Sample("acceptance2.txt"); 
    vec estimated_Eloc(Nsteps, fill::zeros);
    for(int i=0;i<Nsteps;i++){
      // Calculate the Eloc estimates
      estimated_Eloc(i)  = Eloc_calc(sampled_trialWF(0,i), mu, sigma);
    }
    vec loss;
    //
    loss = blockAverage(estimated_Eloc, Nblocks);
    out << setw(wd) << loss(1) << endl;
    return loss(0) * beta;
  };
  
  // Define useful parameters for the simulated annealing
  vec starting_var_param = {1.,1.};     
  double hyperWidth = 0.1;
  // Create an instance of Metropolis that will perform SA \
  Nsteps is set to zero since it will be modified at each stage of the annealing schedule \
  mode is set to 1 since we feed a logpdf
  Metropolis WFspace(starting_var_param, 0, 0, hyperWidth, unifStep, Boltzmann, unif, 1); 

  // Create a matrix to hold the Annealing Scheme \
  Create a 2x(number of stages) matrix. First row is Nsteps, second row is betas
  mat AnnealingScheme = {{100, 100, 100, 100}, {1e10,1e11,1e12,1e13}}; 
  mat sampled_var_param;                            // Create a matrix to hold the annealing output
  for(int i=0; i< AnnealingScheme.n_cols;i++){
    // The internal variable of the WFspace class containing the probability of the previous step has to be rescaled beacuse beta is changed
    WFspace.ScaleCurrentProb(double(AnnealingScheme(1,i))/double(beta));
    // Set new beta
    beta = AnnealingScheme(1,i);
    // Set new Nsteps
    WFspace.SetNsteps(AnnealingScheme(0,i));
    // Perform the current stage of SA
    mat stage = WFspace.Optimize();            
    cout << "stage " << i+1 << " completed" << endl;
    // Accumulate measurements
    sampled_var_param = join_rows(sampled_var_param, stage);
  }
 
  // Output the results
  title = "output_Annealing";
  column_names = {"mu","sigma","energy","acceptance"};
  data_vectors = {sampled_var_param.row(0).t(), sampled_var_param.row(1).t(),sampled_var_param.row(2).t(),sampled_var_param.row(3).t()};
  write_data_file(title, column_names, data_vectors);

  // Return arrival seed
  rnd.SaveSeed();
  return 0;
}

/****************************************************************
*****************************************************************
  Ex 8 _____  Numerical Simulation Lab _____  Lorenzo Lazzarino
*****************************************************************
*****************************************************************/