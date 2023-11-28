/****************************************************************
*****************************************************************
  Ex 9 _____  Numerical Simulation Lab _____  Lorenzo Lazzarino
*****************************************************************
*****************************************************************/
//for this exercise the seed used was 0000 0000 0000 0001

#include <iostream>
#include <fstream>
#include <string>
#include "include/random.h"
#include "include/utilities.h"
#include "include/metropolis.h"
#include "include/GA.h"
#include <armadillo>

using namespace std;
using namespace arma;

int main(){

  //***** ex09 *****//
  //***** TSP w/ Genetic Algorithm  *****//

  // Define and construct the random generator
  Random rnd = Random("src/generator/Primes","src/generator/seed.in");
  auto unif = [&rnd]() { return rnd.Rannyu(); };    // Generate a random uniform number between 0 and 1

  // Define the fitness function for the Traveling Salesman Problem (TSP)
  auto fitnessFunction = [](const mat& data, const rowvec& chromosome) { // L2 norm
    uvec indices = conv_to<uvec>::from(chromosome.t());
    size_t shiftAmount = 1;
    uvec shiftedIndices = join_cols(indices.tail(indices.n_elem - shiftAmount), indices.head(shiftAmount));
    vec xcities = data.col(0);
    vec ycities = data.col(1);
    return (dot(xcities.elem(indices) - xcities.elem(shiftedIndices), xcities.elem(indices) - xcities.elem(shiftedIndices)) 
          + dot(ycities.elem(indices) - ycities.elem(shiftedIndices), ycities.elem(indices) - ycities.elem(shiftedIndices)));
  };

  // Open an output file for fitness values
  ofstream out; 
  out.open("out_fitness.txt");
  const int wd=16;
  out << setw(wd) << "BestFitness" << setw(wd) << "AverageFitnessOnBestHalf" << endl;

  // Initialize the GeneticLab instance with the defined random generator and fitness function
  GeneticLab Glab(unif, fitnessFunction);
  Glab.SetupLab(); // Setup the genetic algorithm

  // Write initial fitness values to the output file
  out << setw(wd) << Glab.GetBestFitness();
  out << setw(wd) << Glab.GetSemiGlobalFitness() << endl;

  int Ngen = Glab.GetNGen(); // Get the number of generations
  // Run the genetic algorithm for Ngen generations
  for(int i=0; i<Ngen; i++){
    Glab.NextGenAndStudy(i); // Perform the next generation and study
    // Write fitness values to the output file after each generation
    out << setw(wd) << Glab.GetBestFitness();
    out << setw(wd) << Glab.GetSemiGlobalFitness() << endl;;
  }
  
  rowvec result = Glab.GetBestChromosome(); // Retrieve the best chromosome

  // Open an output file for the best path obtained
  ofstream out1; 
  out1.open("out_path.txt");
  out1 << result; // Write the best path to the output file

  out.close(); // Close the fitness output file
  out1.close(); // Close the path output file

  // Return arrival seed
  rnd.SaveSeed();
  return 0;
}

/****************************************************************
*****************************************************************
  Ex 9 _____  Numerical Simulation Lab _____  Lorenzo Lazzarino
*****************************************************************
*****************************************************************/