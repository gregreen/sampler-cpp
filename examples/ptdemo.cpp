/* The MIT License (MIT)
 *
 * Copyright (c) 2017 Gregory M. Green <gregorymgreen@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <functional>
#include <math.h>
#include <memory>
#include <map>
#include <iomanip>

#include "../src/custom_types.h"
#include "../src/ptsampler.h"
#include "../src/mhsampler.h"
#include "../src/linalg.h"
#include "../src/rand_helpers.h"

using namespace sampler;


// Returns a log probability density function with peaks at x = +- 1.
// Higher gamma -> peaks are sharper (and more isolated).
//
//   \ln p(x) = -\gamma * (x^2 - 1)^2 + const.
//
// In the vicinity of x = +- 1, the probability density function is a skewed
// Gaussian with
//
//   \sigma \approx (8 \gamma)^{-1/2} .
pdensity double_potential_well(double gamma) {
    return [gamma](double* x) {
        return -gamma * (x[0]*x[0] - 1.) * (x[0]*x[0] - 1.);
    };
}


/*
 * Pretty-print various objects
 */

void print_float(double x, std::ostream& o,
                 int width=10, int precision=5,
                 std::string pm="") {
    std::stringstream s;
    s << std::fixed << std::setw(width) << std::setprecision(precision) << x;
    o << s.str();
}


void print_vector(shared_const_vector v, std::ostream& s) {
    for(auto el : *v) {
        print_float(el, s);
    }
    s << std::endl;
}


void print_matrix(shared_const_matrix mat, std::ostream& s) {
    for(int j=0; j<mat->rows(); j++) {
        for(int k=0; k<mat->cols(); k++) {
            print_float((*mat)(j,k), s);
        }
        s << std::endl;
    }
}


void print_correlation_matrix(shared_const_matrix mat, std::ostream& s) {
    double tmp;
    for(int j=0; j<mat->rows(); j++) {
        for(int k=0; k<mat->cols(); k++) {
            tmp = (*mat)(j,k);
            if(j == k) {
                tmp = sqrt(tmp);
            } else {
                tmp /= sqrt((*mat)(j,j) * (*mat)(k,k));
            }
            print_float(tmp, s);
        }
        s << std::endl;
    }
}


// Sample a multi-modal distribution using a Metropolis-Hastings sampler
void example_MH() {
    int n_dim = 3;
    pdensity ln_prior = [](double* x) { return 0.; };

    for(double gamma=1.; gamma<16.01; gamma*=2.0) {
        pdensity lnL = double_potential_well(gamma);
        pdensity lnL_nd  = [lnL, n_dim](double* x) {
            double ret = 0.;
            for(int i=0; i<n_dim; i++) { ret += lnL(x+i); }
            return ret;
        };

        BasicRandGenerator r;
        // std::random_device rd;
        // std::default_random_engine rand_gen(rd());
        // std::uniform_real_distribution<> p(-1.5, 1.5);

        MHSampler sampler(lnL_nd, ln_prior, n_dim);

        std::vector<double> x0;
        for(int i=0; i<n_dim; i++) {
            x0.push_back(r.uniform(-1.5, 1.5));
        }
        sampler.set_state(x0);

        int n_steps = 1000000;
        for(int i=0; i<n_steps; i++) {
            sampler.step();
        }
        sampler.clear_chain();
        for(int i=0; i<n_steps; i++) {
            sampler.step();
        }

        shared_const_vector elem = sampler.get_chain()->get_elements();
        shared_const_vector weight = sampler.get_chain()->get_weights();
        //
        // for(int i=0; i<elem->size(); i+=100) {
        //     std::cerr << i << "  "
        //               << elem->at(i) << "  "
        //               << weight->at(i) << std::endl;
        // }
        //
        std::map<int, double> hist;
        for(int i=0; i<weight->size(); i++) {
            int bin = int(round(4.*elem->at(n_dim*i)));
            hist[bin] += weight->at(i);
        }

        double dn = (double)n_steps / 100.;
        std::cerr << std::endl;

        for(int i=-12; i<=12; i++) {
            std::cerr << std::setw(5) << (double)i / 4. << " ";
            for(double n=0; n<hist[i]; n+=dn) {
                std::cerr << "*";
            }
            std::cerr << std::endl;
        }

        std::cerr << std::endl
                  << "acceptance: "
                  << 100. * sampler.accept_frac()
                  << "%"
                  << std::endl << std::endl;

        shared_vector x_mean = weighted_mean(elem, weight, 2);
        shared_matrix cov = weighted_covariance(elem, weight, 2);

        std::cerr << "mean = " << x_mean->at(0) << std::endl;
        std::cerr << std::endl
                  << "cov:" << std::endl;
        print_matrix(cov, std::cerr);
        std::cerr << std::endl
                  << "cov^-1:" << std::endl;
        // shared_matrix inv_cov = cov->inverse();
        print_matrix(std::make_shared<MatrixNxM>(cov->inverse()), std::cerr);
        std::cout << std::endl;

        std::stringstream out_fname;
        out_fname << "output/res_gamma_"
                  << std::setprecision(1) << gamma
                  << ".txt";
        std::cerr << "Writing output to: " << out_fname.str() << std::endl;
        std::ofstream f_out(out_fname.str());
        for(int k=0; k<n_dim; k++) {
            f_out << "x" << k << " ";
        }
        f_out << "w" << std::endl;
        for(int i=0; i<weight->size(); i++) {
            for(int k=0; k<n_dim; k++) {
                f_out << elem->at(n_dim*i+k) << " ";
            }
            f_out << weight->at(i) << std::endl;
        }
        f_out.close();

        std::cerr << std::endl;
    }
}


// Sample a multi-modal distribution using the Parallel Tempering sampler
void example_PT() {
    // Basic parameters of the distribution
    int n_dim = 2;
    double gamma = 100.; // Higher -> modes more isolated

    // PT sampler parameters
    int n_temperatures = 5;
    double temperature_spacing = 4.; // Separate temperatures by this factor

    // Create the ln(prior) and ln(likelihood) functions
    pdensity ln_prior = [](double* x) { return 0.; };
    pdensity lnL = double_potential_well(gamma);

    // Double well along each axis
    pdensity lnL_nd  = [lnL, n_dim](double* x) {
        double ret = 0.;
        for(int i=0; i<n_dim; i++) { ret += lnL(x+i); }
        return ret;
    };

    // Set up the PT sampler
    PTSampler sampler(lnL_nd, ln_prior, n_dim,
                      n_temperatures, temperature_spacing);

    // Function that generates random states
    vector_generator rand_state = [n_dim]() {
        BasicRandGenerator r;
        shared_vector v = std::make_shared<std::vector<double> >(n_dim);
        for(int i=0; i<n_dim; i++) {
            v->push_back(r.uniform(-1.5, 1.5));
        }
        return v;
    };

    // Initialize the PT sampler with random-state generator
    sampler.set_state(rand_state);

    // Burn in while tuning the single-temperature proposal distributions
    // and the temperature spacing. The chain is cleared at the end of tuning.
    PTTuningParameters p;
    p.n_rounds = 100;
    p.n_swaps_per_round = 100;
    p.n_steps_per_swap = 50;

    sampler.tune_all(p);

    shared_const_vector beta_tuned = sampler.get_beta();
    std::cerr << std::endl
              << "beta = ";
    print_vector(beta_tuned, std::cerr);
    std::cerr << std::endl;

    // Sample the distribution
    int n_swaps = 100000;
    int n_steps_per_swap = 10;  // # of Metropolis-Hastings steps between swaps
    sampler.step_multiple(n_swaps, n_steps_per_swap);

    // Report results at each temperature
    for(int beta_idx=0; beta_idx<n_temperatures; beta_idx++) {
        std::cerr << std::endl
                  << "beta = "<< beta_tuned->at(beta_idx)
                  << std::endl << std::endl;

        std::shared_ptr<const Chain> chain = sampler.get_chain(beta_idx);

        // Histogram along one axis
        uint n_bins = 25;
        std::pair<shared_vector, shared_vector> hist = chain->calc_histogram(
            0,      // zeroeth axis
            -3.125, // minimum
            3.125,  // maximum
            n_bins, // # of bins (spacing = 0.25)
            true    // Normalize to unity
        );

        double dn = 1. / (4. * n_bins);
        for(int i=0; i<hist.second->size(); i++) {
            double bin_center = 0.5*(hist.first->at(i) + hist.first->at(i+1));
            double bin_value = hist.second->at(i);

            std::cerr << std::setw(6) << bin_center << " ";
            for(double n=0; n<bin_value; n+=dn) {
                std::cerr << "*";
            }
            std::cerr << std::endl;
        }

        // Acceptance rate for Metropolis-Hastings steps in this sub-sampler
        std::cerr << std::endl
                  << "acceptance: "
                  << 100. * sampler.get_sampler(beta_idx)->accept_frac()
                  << "%"
                  << std::endl << std::endl;

        // Mean and covariance
        shared_vector x_mean = chain->calc_mean();
        shared_matrix cov = chain->calc_covariance();

        std::cerr << "mean = ";
        for(auto v : *x_mean) {
            std::cerr << v << " ";
        }
        std::cerr << std::endl;
        std::cerr << std::endl
                  << "correlation:" << std::endl;
        print_correlation_matrix(cov, std::cerr);
        std::cerr << std::endl
                  << "cov^-1:" << std::endl;
        print_matrix(std::make_shared<MatrixNxM>(cov->inverse()), std::cerr);
        std::cout << std::endl;

        // Write output to file
        int n_samples = 5000;

        shared_const_vector elem = chain->get_elements();
        shared_const_vector weight = chain->get_weights();

        // Thin Markov Chain?
        int thinning = 1;
        if(elem->size() > n_samples) {
            thinning = round((double)elem->size() / (double)n_samples);
            if(thinning == 0) { thinning = 1; }
        }

        std::stringstream out_fname;
        out_fname << "output/res_beta_"
                  << beta_idx
                  << ".txt";
        std::cerr << "Writing output to: " << out_fname.str() << std::endl;
        std::ofstream f_out(out_fname.str());
        f_out << "# beta = " << beta_tuned->at(beta_idx) << std::endl;
        f_out << "# ";
        for(int k=0; k<n_dim; k++) {
            f_out << "x" << k << " ";
        }
        f_out << "w" << std::endl;
        for(int i=0; i<weight->size(); i+=thinning) {
            for(int k=0; k<n_dim; k++) {
                f_out << elem->at(n_dim*i+k) << " ";
            }
            f_out << weight->at(i) << std::endl;
        }
        f_out.close();

        std::cerr << std::endl
                  << "proposal covariance:"
                  << std::endl;

        print_correlation_matrix(
            sampler.get_sampler(beta_idx)->get_proposal_cov(),
            std::cerr
        );

        std::cerr << std::endl;
    }

    // Acceptance rate for swap steps
    std::cerr << std::endl
              << "swap acceptance: "
              << 100. * sampler.swap_accept_frac()
              << "%"
              << std::endl << std::endl;
}


void example_multivariate_normal() {
    shared_matrix sigma = std::make_shared<MatrixNxM>(2,2);
    *sigma << 1.0, -0.9,
              -0.9, 1.0;

    std::cout << "covariance:" << std::endl
              << *sigma << std::endl << std::endl;

    MultivariateNormalGenerator gen(sigma);
    std::cout << "random draw: " << std::endl
              << gen.draw() << std::endl;

}


int main(int argc, char **argv) {
    example_PT();
    // example_multivariate_normal();

    return 0;
}
