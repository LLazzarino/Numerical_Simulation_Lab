
#include <iostream>
#include <fstream>
#include <ostream>
#include <cmath>
#include <iomanip>
#include <random>

using namespace std;

// Create a random number generator engine
std::random_device rd;   // Obtain a random seed from hardware
std::mt19937 gen(rd());  // Seed the generator

// Define a uniform distribution from 0.0 to 1.0
std::uniform_real_distribution<> dis(0.0, 1.0);


void Shuffle(int array[], int length){
  for(int i=length-1; i >= 1; i--){
    //Select randomly a particle (for C++ syntax, 0 <= o <= nspin-1)
    int o = (int)(dis(gen)*(i+1));
    int t = array[o];
    array[o] = array[i];
    array[i] = t;
  }
} 

int main(){
    int nspin=50;
    int* flip_list = new int[nspin];

  for(int i=0; i<nspin; i++){
    flip_list[i]=i;
  }

  //shuffle the positions

  Shuffle(flip_list, nspin);

  for(int i = 0; i<nspin; i++){
    cout<< flip_list[i]<< ", ";
  }
}