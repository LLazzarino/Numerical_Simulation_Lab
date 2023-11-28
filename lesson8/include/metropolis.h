#ifndef __Metropolis__
#define __Metropolis__

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <armadillo>
#include <iomanip>
#include "../include/random.h"

using namespace std;
using namespace arma;

class Metropolis {

  public:
    // Default constructor, it sets up all the parameters at initialization
    Metropolis(vec start, int Nsteps, int NequilSteps,
							 double width, function<double(double)> stepProposal,
					 	 	 function<double(vec)> pdf, function<double()> acceptRejectPdf,
							 int mode ) :
							 _pdf(pdf), _startingPoint(start), _Nsteps(Nsteps),
							 _NequilSteps(NequilSteps), _stepWidth(width),
							 _step(stepProposal), _currentWalker(start) ,
							 _configDimension(start.n_elem), _acceptRejectPdf(acceptRejectPdf),
							 _mode(mode) { _currentProb = _pdf(_currentWalker); }

    // Destructor
    ~Metropolis(){;};

	void Move();                                                                  // Perform a single Metropolis update
	mat Sample(string acceptanceFile);                                            // Sample the desired PDF with subsequent moves
	mat Optimize();                                            					  // As Metropolis.Sample(...) but returns PDF evaluation at each move
    void SetStart(vec start) {_startingPoint = start; _currentWalker = start;};   // Set the starting walker position
	vec GetWalker() {return _currentWalker;};									  // Returns the current walker position
    void SetWidth(double width) {_stepWidth = width;};                            // Set the proposal step width
    void SetThermalization(int NequilSteps) {_NequilSteps = NequilSteps;};        // Set the number of initial steps to be discarded
	void SetNsteps(int Nsteps) {_Nsteps = Nsteps;};								  // Set the number of steps to perform in a sample
	void ScaleCurrentProb(double scale) {_currentProb = _currentProb * scale;};   // Rescale or set to zero the current pdf value

  private:
	function<double(vec)> _pdf;                                                   // The target PDF
	vec _startingPoint;                                                           // Starting position of the walker
	int _Nsteps;                                                                  // Number of configurations to sample
	int _NequilSteps;                                                             // Number of initial steps to discard in order to reach thermalization
	double _stepWidth;                                                            // Width (usually standard deviation) of the transition kernel
	function<double(double)> _step;                                               // Random number generator used as transition kernel
	vec _currentWalker;                                                           // Current position of the walker in the configuration space
	int _configDimension;                                                         // Number of dimensions of the configuration space
	function<double(void)> _acceptRejectPdf;                                      // Uniform random number generator to use in the acceptance of the metropolis move
	int _mode;																	  // If 0 -> classic acceptance; if 1 -> uses logpdfs, useful for boltzmann probabilities
	double _currentProb;														  // Stores the target pdf value in the current walker, to avoid recalculation

	int _accepted;                                                                // Number of accepted metropolis moves
	int _attempted;                                                               // Number of attempted metropolis moves

};

#endif // __Metropolis__
