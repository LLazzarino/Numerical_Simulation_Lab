#include "../include/utilities.h"

void write_data_file(string title, vector<string> column_names, vector<vec> data_vectors) {
    //write a result file in a table format
    int column_width = 16;

    // Open the output file
    ofstream outfile(title + ".txt");

    if (outfile.is_open()){
      // Write the column names to the first line of the output file
      for (auto& name : column_names) {
          outfile << setw(column_width) << name << "\t";
      }
      outfile << endl;

      // Write the data from each vector to the corresponding column in the output file
      int num_rows = data_vectors[0].size();
      int num_cols = data_vectors.size();
      for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
          outfile << setw(column_width) << data_vectors[j](i) << "\t";
        }
        outfile << endl;
      }
    } else cerr << "PROBLEM: Unable to open the output file" << endl;

    // Close the output file
    outfile.close();
}

void blockAverage(vec data, int Nblocks, vec& average, vec& error){
  //calculate the average and error for a cumulative number of data blocks
  int Ndata=data.size();
  int blocksize = int(Ndata/Nblocks);

  vec sums(Nblocks, fill::zeros);
  vec averages(Nblocks, fill::zeros);
  vec averages2(Nblocks, fill::zeros);

  for(int i=0; i<Nblocks; i++){
    for(int j=0; j<blocksize; j++){
      //sum all data in a block
      sums(i)+=data(i*blocksize+j);
    }
    averages(i) = sums(i) / blocksize;
  }
  averages2 = averages % averages;
  //each value of the average and error vector will be the average and error considering increasing blocks
  average = cumsum(averages);
  error = cumsum(averages2);
  //set the error of the first block average to 0
  error(0)=0.;

  for(int i = 1; i<Nblocks; i++){
    //calculate the average and standard deviation
    average(i) = average(i) / (i+1);
    error(i) = sqrt((error(i)/(i+1) - average(i) * average(i))/i);
  }
}

void blockAverageReciprocal(vec data, int Nblocks, vec& average, vec& error){
  //calculate the average and error of the reciprocals for a cumulative number of data blocks
  int Ndata=data.size();
  int blocksize = int(Ndata/Nblocks);

  vec sums(Nblocks, fill::zeros);
  vec reciprocalAverages(Nblocks, fill::zeros);
  vec averages2(Nblocks, fill::zeros);

  for(int i=0; i<Nblocks; i++){
    for(int j=0; j<blocksize; j++){
      //sum all data in a block: this assures no block has value zero as long as blocks are big enough
      sums(i)+=data(i*blocksize+j);
    }
    //take the reciporcal of every average
    reciprocalAverages(i) = blocksize / sums(i);
  }
  averages2 = reciprocalAverages % reciprocalAverages;
  //each value of the average and error vector will be the average and error considering increasing blocks
  average = cumsum(reciprocalAverages);
  error = cumsum(averages2);
  //set the error of the first block average to 0
  error(0)=0.;

  for(int i = 1; i<Nblocks; i++){
    //calculate the average and standard deviation
    average(i) = average(i) / (i+1);
    error(i) = sqrt((error(i)/(i+1) - average(i) * average(i))/i);
  }
}

vec blockAverage(vec data, int Nblocks){
  //calculate the average and error from a dataset by dividing it in blocks
  int Ndata=data.size();
  int blocksize = int(Ndata/Nblocks);

  vec sums(Nblocks, fill::zeros);
  vec averages(Nblocks, fill::zeros);
  vec averages2(Nblocks, fill::zeros);

  for(int i=0; i<Nblocks; i++){
    //sum all the elements in a block and average
    for(int j=0; j<blocksize; j++){
      sums(i)+=data(i*blocksize+j);
    }
    averages(i) = sums(i) / blocksize;
  }

  //calculate the average and error using all blocks
  averages2 = averages % averages;
  double result = mean(averages);
  double error = sqrt( (mean(averages2) - result*result) / (Nblocks-1) );

  //compresses the output before return
  vec results = {result,error};
  return results;
}
