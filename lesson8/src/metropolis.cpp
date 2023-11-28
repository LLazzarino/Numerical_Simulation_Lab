#include "../include/metropolis.h"


void Metropolis::Move(){
	// This method updates the current walker's position in configuration space according to the metropolis algorithm

	// Create a vector to hold the new walker's position
	vec newWalker(_configDimension, fill::zeros);

	for(int i=0;i<_configDimension; i++){
		// For each direction in the configuration space generate a proposal step
		newWalker[i] = _currentWalker[i] + _step(_stepWidth);
	}
	// Evaluate the acceptance rate of the proposed step
	double acceptance;
	double newprob = _pdf(newWalker);
	if(_mode == 0)  acceptance = newprob / _currentProb;
	if(_mode == 1)  acceptance = exp( - newprob + _currentProb );

	// Accept the step with probability = "acceptance"
	double r = _acceptRejectPdf();
	if(r<=acceptance){
		_currentWalker = newWalker;
		_accepted += 1; 			//update the counters
		_currentProb = newprob; 	//update the storage to avoid recalculations
	}
	_attempted +=1;					//update the counters
};


mat Metropolis::Sample(string acceptanceFile){
	// This method combines subsequent Moves to sample the target PDF

	// Open output file to hold the number of attempted and accepted trials
	ofstream out;
	out.open(acceptanceFile);
	const int wd=16;
	out<< setw(wd) <<"attempted" << setw(wd) << "accepted" << endl;

	// Thermalize the system
	for(int i=0;i<_NequilSteps; i++){
		// Perform some steps to be discarded
		Move();
	}

	// Reset counters after thermalization
	_accepted = 0;
	_attempted = 0;

	// Create a matrix to hold the sampled PDF
	mat sampledPDF(_configDimension, _Nsteps, fill::zeros);

	// Sampling
	for(int i=0;i<_Nsteps; i++){
		// Generate a move for the walker
		Move();
		// Save the walker path in the matrix
		sampledPDF.col(i) = _currentWalker;
		// Print out current counters
		out<< setw(wd) << _attempted << setw(wd) << _accepted << endl;
	}

	// Close the file and return the matrix
	out.close();
	return sampledPDF;
};

mat Metropolis::Optimize(){
	// This method combines subsequent Moves to sample the target PDF and returns the estimated PDF value

	const int wd=16;

	// Thermalize the system
	for(int i=0;i<_NequilSteps; i++){
		// Perform some steps to be discarded
		Move();
	}

	// Reset counters after thermalization
	_accepted = 0;
	_attempted = 0;

	// Create a matrix to hold the sampled PDF
	mat sampledPDF(_configDimension+2, _Nsteps, fill::zeros);

	// Sampling
	for(int i=0;i<_Nsteps; i++){
		// Generate a move for the walker
		Move();
		// Save the walker path, the acceptance and the pdf value in the matrix
		sampledPDF.col(i) = join_cols(_currentWalker, vec{_currentProb}, vec{double(_accepted)/double(_attempted)} );
		
	}

	// Return the matrix
	return sampledPDF;

};   