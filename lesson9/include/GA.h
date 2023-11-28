#ifndef __GA__
#define __GA__

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <armadillo>
#include <iomanip>
#include <cassert>
#include "../include/random.h"

using namespace std;
using namespace arma;


class GeneticLab{

  public:
    // Default constructor, it sets up all the parameters at initialization
    GeneticLab(function<double(void)> randUnif,
               function<double(mat,rowvec)> fitnessFunction) : 
               _randUnif(randUnif),
               _fitnessFunction(fitnessFunction){ ; };

    // Destructor
    ~GeneticLab(){;};

    void ReadParameters();                                                              // Reads important parameters to set up the lab
    void ReadInput();                                                                   // Reads the data associated with the alleles
    void GenerateStartingPop();                                                         // Generates the first starting population randomly with shuffle algorithm
    void CheckChromosome(rowvec, int);                                                  // Checks whether the chromosome respects basic necessary properties
    void CheckPopulation();                                                             // Checks all the chromosomes at current generation
    void Shuffle(rowvec&, int);                                                         // Shuffles a vector with Fisher-Yates
    void EvalFitness();                                                                 // Evaluates the fitness of the population using _data and _fitnessFunction
    mat GetPop() {return _pop;};                                                        // Returns the current population
    mat GetData() {return _data;};                                                      // Returns the input data
    vec GetFitnessLog() {return _fitnessLog;};                                          // Returns the current evaluated fitnesses for each chromosome
    void OrderByFitness();                                                              // Sorts all chromosomes by fitness and stores the indices
    void NewGeneration();                                                               // Creates a new generation starting from the current
    uvec SelectPair();                                                                  // Extracts with the selection bias two chromosomes and returns the indices
    void Crossover(mat&, unsigned int, unsigned int);                                   // Crossover operator
    void Mutate_PairPermutation(mat&, unsigned int);                                    // Mutation operator that permutes a random pair of alleles
    void Mutate_ClusterPermutation(mat&, unsigned int);                                 // Mutation operator that permutes two clusters of alleles of random length
    void Mutate_Shift(mat&, unsigned int);                                              // Mutation operator that shifts randomly a cluster of random length
    void Mutate_Inversion(mat&, unsigned int);                                          // Mutation operator that inverts the order of the alleles in a random sequence 
    void NextGenAndStudy(int genNumber);                                                // Creates a new generation, checks it and evaluates important metrics
    void SetupLab();                                                                    // Initializes the first generation, checks it and evaluates important metrics
    void SaveBest();                                                                    // Stores the best chromosome, lowest fitness, and fitness average on the best half population
    rowvec GetBestChromosome() {return _bestChromosome;};                               // Returns best chromosome
    double GetBestFitness() {return _bestFitness;};                                     // Returns lowest fitness
    double GetSemiGlobalFitness() {return _avHalfBestFitness;};                         // Returns fitness average on the best half population
    int GetNGen() {return _Ngenerations;};                                              // Returns number of generation as prescribed in the parameter file


  private:
    mat _pop;                                                       // all the chromosomes are collected in a matrix
    vec _fitnessLog;                                                // each entry is the fitness of a chromosome
    uvec _sortedIndicesFitness;                                     // indices of fitness log ordered by ascending fitness

    mat _data;                                                      // holds the data associated with the alleles
    rowvec _bestChromosome;                                         // holds the best chromosome
    double _bestFitness;                                            // stores the best fitness
    double _avHalfBestFitness;                                      // stores fitness average on the best half population


    function<double(mat,rowvec)> _fitnessFunction;                  // function to evaluate fitness. In this implementation lower is better
    function<double(void)> _randUnif;                               // Uniform random number generator to use across the algorithm

    int _Nalleles;                                                  // Number of alleles per chromosome
    int _popSize;                                                   // Size of the population
    double _selectionBias;                                          // Bias factor for selection
    int _elitism;                                                   // Number of elite chromosomes
    double _pcrossover;                                             // Crossover probability
    double _pmutation_pairPermutation;                              // Probability of pair permutation mutation
    double _pmutation_ClusterPermutation;                           // Probability of cluster permutation mutation
    double _pmutation_Shift;                                        // Probability of shift mutation
    double _pmutation_Inversion;                                    // Probability of inversion mutation
    int _Ngenerations;                                              // Number of generations in a run (Number of iterations)
};

#endif // __Genetic Algorithm__