#include "../include/VMC.h"

//calculate Local Energy for ex8 trial function and Hamiltonian
double Eloc_calc(double x, double mu, double sigma){
    double sig2 = sigma*sigma;
    double sig4 = sig2*sig2;
    double x2 = x*x;
    double a = x * mu / sig2;
    double kin = (sig2 - mu*mu - x2)/(2.*sig4) + a /sig2 * tanh(a);
    double pot = x2*x2 - 5./2. * x2;
    return kin + pot;
}