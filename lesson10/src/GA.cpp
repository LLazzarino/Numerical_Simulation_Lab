#include "../include/GA.h"

void GeneticLab::Inject_New(int num){
    uvec indices = Select_Worst(num);

    int rows = _popSize;
    int cols = _Nalleles;

    for(int i = 0; i<num; i++){
        

       
        for (int j = 0; j < cols; j++) {
            _pop(indices(i), j) = j; // Assign increasing numbers starting from 1 in each column
        }
    

        // Shuffle elements in each chromosome, excluding the first element
        
        rowvec rowbutfirstelement = _pop.row(indices(i)).subvec(1, cols - 1);
        Shuffle( rowbutfirstelement, cols-1 );
        _pop.row(indices(i)).subvec(1, cols - 1) = rowbutfirstelement;
        
    }
};

// Sorts num different chromosomes randomly among the best: useful for migration purposes
uvec GeneticLab::Select_Worst(int num){
    // Extract first chromosome
    uvec indices = {};
    int c1 = (int)(_popSize*pow(_randUnif(),1./_selectionBias)); //_selectionBias should be greater than 1
    uvec newindex = {_sortedIndicesFitness(c1)};
    indices = join_cols(indices, newindex );
    // Etract further chromosomes: they need to be different
    int nextc;
    uvec newc;
    for(int i = 1; i<num; i++){
        do{
            nextc = _sortedIndicesFitness( (int)(_popSize*pow(_randUnif(),_selectionBias)) );
            newc = find(indices == nextc);
        } while( newc.n_elem > 0); // Check if the proposed chromosome is new
        newindex = {nextc};
        indices = join_cols(indices, newindex );
    }
    return indices;
};

// Sorts num different chromosomes randomly among the best: useful for migration purposes
uvec GeneticLab::Migrate_Leave(int num){
    // Extract first chromosome
    uvec indices = {};
    int c1 = (int)(_popSize*pow(_randUnif(),_selectionBias)); //_selectionBias should be greater than 1
    uvec newindex = {_sortedIndicesFitness(c1)};
    indices = join_cols(indices, newindex );
    // Etract further chromosomes: they need to be different
    int nextc;
    uvec newc;
    for(int i = 1; i<num; i++){
        do{
            nextc = _sortedIndicesFitness( (int)(_popSize*pow(_randUnif(),_selectionBias)) );
            newc = find(indices == nextc);
        } while( newc.n_elem > 0); // Check if the proposed chromosome is new
        newindex = {nextc};
        indices = join_cols(indices, newindex );
    }
    return indices;
};

// Modifies the population with a given chromosome: useful for migration among continents
void GeneticLab::Migrate_Welcome(rowvec newc, int pos){
    // Saves the chromosome in the private matrix _pop
    _pop.row(pos) = newc;
}; 

// Mutation operation: Inversion mutation on a chromosome in the population
void GeneticLab::Mutate_Inversion(mat& pop, unsigned int index){
    // Select random segment size 'm' for inversion
    // _Nalleles represents the number of alleles in the chromosome
    int m = (int)(_randUnif()*(_Nalleles-1) + 1);

    // Select starting position 'pos' for inversion
    int pos = (int)(_randUnif()*(_Nalleles - m) + 1);

    // Create a subview of length 'm' starting at position 'j' in row 'i'
    arma::subview_row<double> sub = pop.row(index).subvec(pos, pos + m - 1);

    // Copy the subview elements to a temporary vector
    vector<double> temp(sub.begin(), sub.end());

    // Reverse the elements in the temporary vector
    reverse(temp.begin(), temp.end());

    // Copy the reversed elements back to the subview
    for (int k = 0; k < m; ++k) {
        sub(k) = temp[k];
    }

    
};

// Mutation operation: Pair permutation mutation on a chromosome
void GeneticLab::Mutate_PairPermutation(mat& pop, unsigned int index){
    // Select two distinct positions 'pos1' and 'pos2' for pair permutation
    int pos1 = (int)(_randUnif()*(_Nalleles-1) + 1);
    int pos2;
    do{
        pos2 = (int)(_randUnif()*(_Nalleles-1) + 1);
    }   while(pos2 == pos1);

    // Perform pair permutation by swapping elements at 'pos1' and 'pos2'
    int temp = pop(index, pos1);
    pop(index, pos1) = pop(index, pos2);
    pop(index, pos2) = temp;
};

// Mutation operation: Cluster permutation mutation on a chromosome
void GeneticLab::Mutate_ClusterPermutation(mat& pop, unsigned int index){
    // Select random cluster size 'm' for permutation
    int m = (int)(_randUnif()*(_Nalleles-1)/4 + 1);

    // Select starting positions 'pos1' and 'pos2' for cluster permutation
    int pos1 = (int)(_randUnif()*(_Nalleles-m) + 1);
    int pos2;
    do{
        pos2 = (int)(_randUnif()*(_Nalleles-m) + 1);
    }   while((pos1 >= pos2 && pos1 <= pos2+m-1) || (pos2 >= pos1 && pos2 <= pos1+m-1));

    // Perform cluster permutation by swapping elements in clusters of size 'm'
    for(int i =0; i<m;i++){
        int temp = pop(index, pos1+i);
        pop(index, pos1+i) = pop(index, pos2+i);
        pop(index, pos2+i) = temp;
    }
};

// Mutation operation: Shift mutation on a chromosome
void GeneticLab::Mutate_Shift(mat& pop, unsigned int index){
    // Select random segment size 'm' for shift
    int m = (int)(_randUnif()*(_Nalleles-1)/4 + 1) ;

    // Select random shift amount
    int shift = (int)(_randUnif()*(_Nalleles-1)/4 + 1);

    // Select starting position 'pos' for shift
    int pos = (int)(_randUnif()*(_Nalleles - m - shift) + 1);

    // Perform shift mutation on the chromosome
    for(int i =m-1; i>=0;i--){
        int temp = pop(index, pos+i);
        for(int j =0; j<shift;j++){
            pop(index, pos+i+j) = pop(index, pos+i+j+1);   
        }
        pop(index, pos+shift+i) = temp;
    }
};

// Crossover operation: Combines genetic information from two chromosomes
void GeneticLab::Crossover(mat& pop, unsigned int index1, unsigned int index2){
    // Select a crossover position 'pos'
    int pos = (int)(_randUnif()*(_Nalleles - 2) + 1);

    // Get chromosomes to perform crossover
    rowvec c1 = pop.row(index1);
    rowvec c2 = pop.row(index2);

    // Perform crossover by exchanging genetic information between chromosomes
    // based on the crossover position
    // Select matching elements between the two chromosomes and swap them
    uvec indices(_Nalleles-pos);
    for(int i=pos; i<_Nalleles;i++){
        double searchTarget = pop(index1,i);
        uvec searchResults = arma::find(pop.row(index2) == searchTarget);
        indices(i-pos) = searchResults(0);
    }
    uvec sortedIndicesCrossover1 = arma::sort_index(indices);

    for(int i=pos; i<_Nalleles;i++){
        double searchTarget = pop(index2,i);
        uvec searchResults = arma::find(pop.row(index1) == searchTarget);
        indices(i-pos) = searchResults(0);
    }
    uvec sortedIndicesCrossover2 = arma::sort_index(indices);

    
    // Perform the crossover operation
    for(int i=pos; i<_Nalleles;i++){

        pop(index1,i) = c1(pos+sortedIndicesCrossover1(i-pos));
    }

    for(int i=pos; i<_Nalleles;i++){
        pop(index2,i) = c2(pos+sortedIndicesCrossover2(i-pos));
    }


};

// Function to generate the next generation of the population
void GeneticLab::NewGeneration(){
    // Create a new population matrix
    int rows = _popSize;
    int cols = _Nalleles;
    mat newPop(rows, cols, fill::zeros);

    // Elitism: save the best chromosomes
    for(int i=0; i<_elitism; i=i+2){
        uvec pair = SelectPair();
        newPop.row(i) = _pop.row(pair(0));
        newPop.row(i+1) = _pop.row(pair(1));
    }
    
    // For the remaining population, perform crossover, mutation
    for(int i=_elitism; i<_popSize; i=i+2){
        //selects a pair
        uvec pair = SelectPair();
        //copies in the new matrix
        newPop.row(i) = _pop.row(pair(0));
        newPop.row(i+1) = _pop.row(pair(1));

        /* straight copies to test mutations
        newPop.row(i) = _pop.row(i);
        newPop.row(i+1) = _pop.row(i+1);
        */

        //crossover on the pair
        if(_randUnif() < _pcrossover ) Crossover(newPop, i,i+1);

        //mutate on the singles
        if(_randUnif() < _pmutation_pairPermutation ) Mutate_PairPermutation(newPop, i);
        if(_randUnif() < _pmutation_ClusterPermutation ) Mutate_ClusterPermutation(newPop, i);
        if(_randUnif() < _pmutation_Shift ) Mutate_Shift(newPop, i);
        if(_randUnif() < _pmutation_Inversion ) Mutate_Inversion(newPop, i);

        if(_randUnif() < _pmutation_pairPermutation ) Mutate_PairPermutation(newPop, i+1);
        if(_randUnif() < _pmutation_ClusterPermutation ) Mutate_ClusterPermutation(newPop, i+1);
        if(_randUnif() < _pmutation_Shift ) Mutate_Shift(newPop, i+1);
        if(_randUnif() < _pmutation_Inversion ) Mutate_Inversion(newPop, i+1);
    }
        
    // Replace the old population with the new one
    _pop = newPop;    
}; 

// Function to carry out a generation and perform analysis/study
void GeneticLab::NextGenAndStudy(int genNumber){
    cout << "gen number: " << genNumber << endl;
    Inject_New(_popSize/10);
    EvalFitness(); // Evaluate fitness for each chromosome
    OrderByFitness(); // Order the population by fitness

    NewGeneration(); // Generate the next generation
    CheckPopulation(); // Check the generated population
    EvalFitness(); // Evaluate fitness for each chromosome
    OrderByFitness(); // Order the population by fitness
    SaveBest(); // Save the best fitness and chromosome
    //_selectionBias = _selectionBias / sqrt(1./genNumber + 1) + 1;
    //_pmutation_ClusterPermutation = _pmutation_ClusterPermutation * pow( genNumber/1000 +1 , 2) ;
    //_pmutation_Inversion = _pmutation_Inversion * pow( genNumber/1000 +1 , 2);
    //_pmutation_pairPermutation = _pmutation_pairPermutation * pow( genNumber/1000 +1 , 2);
    //_pmutation_Shift = _pmutation_Shift * pow( genNumber/1000 +1 , 2);
    cout << "gen "<< genNumber << " done." <<endl;
    cout << "Best Fitness is: "<< _bestFitness << endl;
    cout << "Best Global Fitness average on best half population is: "<< _avHalfBestFitness << endl;
    cout << "------------------------------"<<endl;
    cout << "------------------------------"<<endl<<endl;
};

// Function to save the best fitness and chromosome in the current generation
void GeneticLab::SaveBest(){
    _bestFitness = _fitnessLog(_sortedIndicesFitness(0));
    _bestChromosome = _pop.row(_sortedIndicesFitness(0));
    
    int halfSize = _fitnessLog.n_elem / 2;
    _avHalfBestFitness = 0.0;
    for (int i = 0; i < halfSize; i++) {
        _avHalfBestFitness += _fitnessLog(_sortedIndicesFitness(i));
    }
    _avHalfBestFitness = _avHalfBestFitness / halfSize;
};

// Function to set up the parameters and data for the genetic algorithm
void GeneticLab::SetupLab(){ 
    cout << "Setting up the lab..."<<endl;
    ReadParameters(); // Read simulation parameters
    ReadInput(); // Read location data of the cities
    GenerateStartingPop(); // Generate the initial population
    CheckPopulation(); // Check if the generated population meets requirements
    EvalFitness(); // Evaluate fitness for each chromosome
    OrderByFitness(); // Order the population by fitness
    SaveBest(); // Save the best fitness and chromosome
    cout << "Setup done."<<endl;
    cout << "Best Fitness is: "<< _bestFitness << endl;
    cout << "Best Global Fitness average on best half population is: "<< _avHalfBestFitness << endl;
    cout << "------------------------------"<<endl;
    cout << "------------------------------"<<endl<<endl;
};

// Function to select a pair of chromosomes based on a biased selection strategy
uvec GeneticLab::SelectPair(){
    // Perform selection bias based on population size and random selection
    int c1 = (int)(_popSize*pow(_randUnif(),_selectionBias)); //_selectionBias should be greater than 1
    int c2 = (int)(_popSize*pow(_randUnif(),_selectionBias));

    // Second possible scheme
    //int c1 = (int)(_popSize* (exp(_randUnif()*_selectionBias)-1)/(exp(_selectionBias)-1)); //_selectionBias should be greater than 1
    //int c2 = (int)(_popSize* (exp(_randUnif()*_selectionBias)-1)/(exp(_selectionBias)-1));

    uvec indices = {(_sortedIndicesFitness(c1)), (_sortedIndicesFitness(c2))}; 
    return indices;
};

// Function to order the population based on fitness values
void GeneticLab::OrderByFitness(){
    // Get the indices that would sort the vector in ascending order (first elements are closer to the solution)
    _sortedIndicesFitness = arma::sort_index(_fitnessLog);
};

// Function to evaluate fitness for each chromosome in the population
void GeneticLab::EvalFitness(){
    // Initialize fitness log for the population
    _fitnessLog.zeros(_popSize);

    // Evaluate fitness for each chromosome using the fitness function
    for(int i =0; i<_popSize;i++){
        _fitnessLog(i) = _fitnessFunction(_data, _pop.row(i));
    }
};

// Function to shuffle a vector used in chromosome operations
void GeneticLab::Shuffle(rowvec& chromosome, int length){
    // Shuffle the vector elements for chromosome manipulation
    for(int i=length-1; i >= 1; i--){
        int o = (int)(_randUnif()*(i+1));
        int t = chromosome[o];
        chromosome[o] = chromosome[i];
        chromosome[i] = t; 
    }
}; 

// Function to check if a chromosome meets certain requirements
void GeneticLab::CheckChromosome(rowvec chromosome, int length){
    // Check specific necessary conditions for the chromosome
    bool OK = true;
    if (chromosome[0] != 0){
        OK = false;
        assert(("The chromosome does not satisty the requirements: it does not start with one", OK));
    }
    if(int(accu(chromosome)) != int( (length) * (length-1)/ 2.) ){
        OK = false;
        assert(("The chromosome does not satisty the requirements: sum of the elements is not as expected", OK));
    }
    assert(("The chromosome does not satisty the requirements", OK));
};

// Function to check if the entire population meets certain requirements
void GeneticLab::CheckPopulation(){
    // Check each chromosome in the population
    for(int i=0; i<_popSize; i++){
        CheckChromosome(_pop.row(i), _Nalleles);
    }
};

// Function to generate the initial population
void GeneticLab::GenerateStartingPop(){
    // Initialize the population matrix with ascending numbers
    int rows = _popSize;
    int cols = _Nalleles;

    // Assign increasing numbers starting from 0 in each chromosome
    _pop.zeros(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            _pop(i, j) = j; // Assign increasing numbers starting from 1 in each column
        }
    }

    // Shuffle elements in each chromosome, excluding the first element
    for(int i = 0; i<rows; i++){
        rowvec rowbutfirstelement = _pop.row(i).subvec(1, cols - 1);
        Shuffle( rowbutfirstelement, cols-1 );
        _pop.row(i).subvec(1, cols - 1) = rowbutfirstelement;
    }
};

// Function to read simulation parameters from an input file
void GeneticLab::ReadParameters(){
    // Read parameters from an input file
    cout << "Reading simulation parameters ..." << endl;

    ifstream ReadInput;
    ReadInput.open("parameters.in");
    ReadInput >> _Nalleles;
    cout << "Number of alleles" << " = " << _Nalleles << endl;
    ReadInput >> _popSize;
    cout << "Population size" << " = " << _popSize << endl;
    ReadInput >> _selectionBias;
    cout << "Selection bias" << " = " << _selectionBias << endl;
    ReadInput >> _elitism;
    cout << "Number elite chromosomes" << " = " << _elitism << endl;
    ReadInput >> _pcrossover;
    cout << "Crossover probability" << " = " << _pcrossover << endl;
    
    ReadInput >> _pmutation_pairPermutation;
    ReadInput >> _pmutation_ClusterPermutation;
    ReadInput >> _pmutation_Shift;
    ReadInput >> _pmutation_Inversion;

    cout << "Mutation probabilities:" << endl;
    cout << "Pair permutation probability" << " = " << _pmutation_pairPermutation << endl;
    cout << "Cluster permutation probability" << " = " << _pmutation_ClusterPermutation << endl;
    cout << "Shift probability" << " = " << _pmutation_Shift << endl;
    cout << "Inversion probability" << " = " << _pmutation_Inversion << endl;
    
    ReadInput >> _Ngenerations;
    cout << "Number of generations in a run" << " = " << _Ngenerations << endl;

    ReadInput.close();

    cout << "------------------------------" << endl;	
};

// Function to read input data from a file
void GeneticLab::ReadInput(){
    // Read the location data of the cities from an input file
    // Initialize _data matrix with city coordinates
    cout << "Reading location of the cities ..." << endl;

    int rows = _Nalleles;
    int cols = 2;
    _data.zeros(rows, cols);
    ifstream ReadInput;
    ReadInput.open("input.in");
    for(int i=0;i<rows;i++){
        ReadInput >> _data(i,0);
        ReadInput >> _data(i,1);
    }
    ReadInput.close();


    cout << "------------------------------" << endl;
};
