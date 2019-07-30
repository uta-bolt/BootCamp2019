/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *
 * This file contains routines to serially compute the call and
 * put price of an European option.
 *
 * Simon Scheidegger
 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#include <algorithm>    // Needed for the "max" function
#include <cmath>
#include <iostream>


/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 A simple implementation of the Box-Muller algorithm, used to
generate gaussian random numbers; necessary for the Monte Carlo
method below. */

double gaussian_box_muller() {
  double x = 0.0;
  double y = 0.0;
  double euclid_sq = 0.0;

  // Continue generating two uniform random variables
  // until the square of their "euclidean distance"
  // is less than unity
  do {
    x = 2.0 * rand() / static_cast<double>(RAND_MAX)-1;
    y = 2.0 * rand() / static_cast<double>(RAND_MAX)-1;
    euclid_sq = x*x + y*y;
  } while (euclid_sq >= 1.0);

  return x*sqrt(-2*log(euclid_sq)/euclid_sq);
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Pricing a European vanilla call option with a Monte Carlo method

double monte_carlo_call_price(const int& num_sims, const double& S, const double& K, const double& r, const double& v, const double& T) {
  double S_adjust = S * exp(T*(r-0.5*v*v)); //this is what he calls S(T)
  double S_cur = 0.0;
  double payoff_sum = 0.0;

  for (int i=0; i<num_sims; i++) {
    double gauss_bm = gaussian_box_muller();
    S_cur = S_adjust * exp(sqrt(v*v*T)*gauss_bm); //gauss_bm is Z_I
    payoff_sum += std::max(S_cur - K, 0.0); //we add these C_is
  }

  return (payoff_sum / static_cast<double>(num_sims)) * exp(-r*T); // we exponentiate and then sum them this is basically Cn
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Pricing a European vanilla put option with a Monte Carlo method

double monte_carlo_put_price(const int& num_sims, const double& S, const double& K, const double& r, const double& v, const double& T) {
  double S_adjust = S * exp(T*(r-0.5*v*v));
  double S_cur = 0.0;
  double payoff_sum = 0.0;

  for (int i=0; i<num_sims; i++) {
    double gauss_bm = gaussian_box_muller();
    S_cur = S_adjust * exp(sqrt(v*v*T)*gauss_bm);
    std::cout << "S_cur";
    std::cout << S_cur;
    payoff_sum += std::max(K - S_cur, 0.0);
  }

  return (payoff_sum / static_cast<double>(num_sims)) * exp(-r*T);
}
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Pricing an Asian  option with a Monte Carlo method

double asian_monte_carlo_call_price(const int& num_trans,const int& num_sims, const double& S, const double& K, const double& r, const double& v, const double& T) {
  double payoff_sum = 0.0;
  for (int i=0; i<num_sims; i++) {
    double S_old=S;
    double S_cur=0.0;
    double S_new=0.0;
    double told=0.0;
    double tnew=0.0;
    double S_bar=0.0;
    int x;
    for (int j=0; j<num_trans;j++){
      double tnew=j+1.0;
      double gauss_bm= gaussian_box_muller();
      S_new = S_old * exp((tnew-told)*(r-0.5*v*v) +sqrt(v*v*(tnew-told))*gauss_bm); //gauss_bm is Z_I
      S_cur += S_new;
      //std::cout << "  scur";
      //std::cout <<  S_cur;
      //std::cin >> x;
      S_old=S_new;
      told=tnew;
      }
      S_bar=(double)S_cur/num_trans;
      payoff_sum += std::max(S_bar - K, 0.0); //we add these C_is
  }

  return (payoff_sum / static_cast<double>(num_sims)) * exp(-r*T); // we exponentiate and then sum them this is basically Cn
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Pricing an Asian  option with a Monte Carlo method

double asian_monte_carlo_put_price(const int& num_trans,const int& num_sims, const double& S, const double& K, const double& r, const double& v, const double& T) {
  double payoff_sum = 0.0;
  for (int i=0; i<num_sims; i++) {
    double S_old=S;
    double S_cur=0.0;
    double S_new=0.0;
    double told=0.0;
    double tnew=0.0;
    double S_bar=0.0;
    int x;
    for (int j=0; j<num_trans;j++){
      double tnew=j+1.0;
      double gauss_bm= gaussian_box_muller();
      S_new = S_old * exp((tnew-told)*(r-0.5*v*v) +sqrt(v*v*(tnew-told))*gauss_bm); //gauss_bm is Z_I
      S_cur += S_new;
      //std::cout << "  scur";
      //std::cout <<  S_cur;
      //std::cin >> x;
      S_old=S_new;
      told=tnew;
      }
      S_bar=(double)S_cur/num_trans;
      payoff_sum += std::max( K-S_bar, 0.0); //we add these C_is
  }

  return (payoff_sum / static_cast<double>(num_sims)) * exp(-r*T); // we exponentiate and then sum them this is basically Cn
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int main(int argc, char **argv) {

  // Parameters
  int num_sims = 300;   // Number of simulated asset paths
  int num_trans=300; //Number of transition paths
  double S = 100.0;  // Option price
  double K = 100.0;  // Strike price
  double r = 0.05;   // Risk-free rate (5%)
  double v = 0.2;    // Volatility of the underlying (20%)
  double T = 1.0;    // One year until expiry

  // Then we calculate the call/put values via Monte Carlo
  double call = monte_carlo_call_price(num_sims, S, K, r, v, T);
  double put = monte_carlo_put_price(num_sims, S, K, r, v, T);
  double call_asian= asian_monte_carlo_call_price(num_trans,num_sims, S, K, r, v, T);
  double put_asian= asian_monte_carlo_put_price(num_trans,num_sims, S, K, r, v, T);
  // Finally we output the parameters and prices
  std::cout << "Number of Paths: " << num_sims << std::endl;
  std::cout << "Underlying:      " << S << std::endl;
  std::cout << "Strike:          " << K << std::endl;
  std::cout << "Risk-Free Rate:  " << r << std::endl;
  std::cout << "Volatility:      " << v << std::endl;
  std::cout << "Maturity:        " << T << std::endl;

  std::cout << "Call Price:      " << call << std::endl;
  std::cout << "Put Price:       " << put << std::endl;
  std::cout << "Asian call Price:      " << call_asian << std::endl;
  std::cout << "Asian put Price:      " << put_asian << std::endl;

  return 0;
}
