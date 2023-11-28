/****************************************************************
*****************************************************************
  Ex 9 _____  Numerical Simulation Lab _____  Lorenzo Lazzarino
*****************************************************************
*****************************************************************/

#include <iostream>
#include <fstream>
#include <string>
#include "include/random.h"
#include "include/utilities.h"
#include "include/metropolis.h"
#include "include/GA.h"
#include "mpi.h"
#include <armadillo>


using namespace std;
using namespace arma;

int main(int argc, char* argv[]){

  //***** ex09 *****//
  //***** TSP w/ Genetic Algorithm  *****//

  //initialize MPI communication
  int size, rank;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  

  // Define and construct the random generator
  Random rnd = Random("src/generator/Primes","src/generator/seed"+to_string(rank)+".in");
  auto unif = [&rnd]() { return rnd.Rannyu(); };    // Generate a random uniform number between 0 and 1

  // Define the fitness function for the Traveling Salesman Problem (TSP)
  /*auto fitnessFunction = [](const mat& data, const rowvec& chromosome) { // L1 norm with haversine formula for longitude and latitude
    double earthRadius = 6371.0; // Earth's radius in kilometers
    uvec indices = conv_to<uvec>::from(chromosome.t());
    size_t shiftAmount = 1;
    uvec shiftedIndices = join_cols(indices.tail(indices.n_elem - shiftAmount), indices.head(shiftAmount));
    vec loncities = data.col(0) / 180. *M_PI;
    vec latcities = data.col(1) / 180. *M_PI;

    vec dlon = loncities.elem(indices) - loncities.elem(shiftedIndices);
    vec dlat = latcities.elem(indices) - latcities.elem(shiftedIndices);
    vec a = arma::sin(dlat.elem(indices) / 2)  % arma::sin(dlat.elem(indices) / 2) + arma::cos(latcities.elem(indices)) 
    % arma::cos(latcities.elem(shiftedIndices)) % arma::sin(dlon / 2) % arma::sin(dlon / 2);
    vec distances = 2 * arma::atan2(sqrt(a), sqrt(1 - a)) * earthRadius;
    return accu(distances);
  };*/

  auto fitnessFunction = [](const mat& data, const rowvec& chromosome) { // L2 norm
    uvec indices = conv_to<uvec>::from(chromosome.t());
    size_t shiftAmount = 1;
    uvec shiftedIndices = join_cols(indices.tail(indices.n_elem - shiftAmount), indices.head(shiftAmount));
    vec xcities = data.col(0);
    vec ycities = data.col(1);
    return (dot(xcities.elem(indices) - xcities.elem(shiftedIndices), xcities.elem(indices) - xcities.elem(shiftedIndices)) 
          + dot(ycities.elem(indices) - ycities.elem(shiftedIndices), ycities.elem(indices) - ycities.elem(shiftedIndices)));
  };

  int Nmigrations = 100; // time interval between migrations
  int Ntravellers = 4; // number of exchanged chromosomes every migration

  // Open an output file for fitness values
  ofstream out; 
  out.open("out_fitness_"+to_string(rank)+".txt");
  const int wd=16;
  out << setw(wd) << "BestFitness" << setw(wd) << "AverageFitnessOnBestHalf" << endl;

  // Initialize the GeneticLab instance with the defined random generator and fitness function
  GeneticLab Glab(unif, fitnessFunction);
  Glab.SetupLab(); // Setup the genetic algorithm
  int NAlleles = Glab.GetNAlleles(); // useful for MPI communication

  // Write initial fitness values to the output file
  out << setw(wd) << Glab.GetBestFitness();
  out << setw(wd) << Glab.GetSemiGlobalFitness() << endl;


  int Ngen = Glab.GetNGen(); // Get the number of generations
  // Run the genetic algorithm for Ngen generations
  for(int i=0; i<Ngen; i++){
    Glab.NextGenAndStudy(i); // Perform the next generation and study
      
      /*
      if (i%Nmigrations==0){
          ///////////////////////// Migration Routine
          
          if(rank==0){
            // print only once that migration is work in progress
            cout << "migration in progres..."<<endl;
          }
          
          // loop among different pairs of processes
          for(int communicator1 = 0; communicator1 < size-1; communicator1++){
            for(int communicator2 = communicator1 + 1; communicator2 < size; communicator2++){
              // for each process extract a number of chromosomes to be exchanged based on fitness
              mat pop = Glab.GetPop(); // Retrieve the population matrix
              uvec leaving_indices = Glab.Migrate_Leave(Ntravellers); // Identify leaving chromosomes

              // cicle on the leaving chromosomes
              for(int i = 0; i<Ntravellers; i++){
                MPI_Status stat1, stat2;
                MPI_Request req;
                int* imesg = new int[NAlleles]; 
                int* imesg2 = new int[NAlleles];
                int itag=1; 
                int itag2=2;

                // bidirectional communication routine
                if(rank==communicator1){
                  rowvec leaving_c = pop.row(leaving_indices(i));

                  // Fill imesg buffer with leaving chromosome data
                  for(int k = 0; k< NAlleles;k++){
                    imesg[k] = int(leaving_c(k));
                  }
                  
                  // Send data from communicator1 to communicator2
                  MPI_Isend(imesg,NAlleles, MPI_INT,communicator2,itag, MPI_COMM_WORLD,&req);  //non blocking to avoid deadlocks
                  // Receive data from communicator2
                  MPI_Recv(imesg2,NAlleles,MPI_INT,communicator2,itag2,MPI_COMM_WORLD,&stat2);
                  
                  // Perform the migration
                  rowvec arrived_c(NAlleles); 
                  for(int k = 0; k< NAlleles;k++){
                    arrived_c(k) = imesg2[k];
                  }

                  Glab.Migrate_Welcome(arrived_c, leaving_indices(i));
                }
                else if(rank==communicator2){
                  rowvec leaving_c = pop.row(leaving_indices(i));

                  // Fill imesg2 buffer with leaving chromosome data
                  for(int k = 0; k< NAlleles;k++){
                    imesg2[k] = int(leaving_c(k));
                  }

                  // Send data from communicator2 to communicator1
                  MPI_Send(imesg2,NAlleles, MPI_INT,communicator1,itag2,MPI_COMM_WORLD);
                  // Receive data from communicator1
                  MPI_Recv(imesg,NAlleles,MPI_INT,communicator1,itag,MPI_COMM_WORLD, &stat1);

                  rowvec arrived_c(NAlleles); 
                  for(int k = 0; k< NAlleles;k++){
                    arrived_c(k) = imesg[k];
                  }

                  Glab.Migrate_Welcome(arrived_c, leaving_indices(i));
                }

                // Additional operations after migration
                Glab.CheckPopulation(); // Check the generated population
                Glab.EvalFitness(); // Evaluate fitness for each chromosome
                Glab.OrderByFitness(); // Order the population by fitness
              }
            }
          }

          Glab.SaveBest(); // Save the best fitness and chromosome  

          // update the files
          out << setw(wd) << Glab.GetBestFitness();
          out << setw(wd) << Glab.GetSemiGlobalFitness() << endl;

          if(rank==0){
            // print only once that the migration is done
            cout << "migration done!"<<endl;
            cout << "------------------------------"<<endl;
            cout << "------------------------------"<<endl<<endl;
          }
          
          //////////////////////////////// end migration
          
    } */
    //
    // Write fitness values to the output file after each generation
    out << setw(wd) << Glab.GetBestFitness();
    out << setw(wd) << Glab.GetSemiGlobalFitness() << endl;
  }
  
  rowvec result = Glab.GetBestChromosome(); // Retrieve the best chromosome

  // Open an output file for the best path obtained
  ofstream out1; 
  out1.open("out_path_"+to_string(rank)+".txt");
  out1 << result; // Write the best path to the output file

  out.close(); // Close the fitness output file
  out1.close(); // Close the path output file

  // Return arrival seed
  rnd.SaveSeed();

  MPI_Finalize(); //end MPI communication

  return 0;
}

/****************************************************************
*****************************************************************
  Ex 9 _____  Numerical Simulation Lab _____  Lorenzo Lazzarino
*****************************************************************
*****************************************************************/