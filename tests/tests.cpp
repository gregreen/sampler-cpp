/* The MIT License (MIT)
 *
 * tests.cpp
 * Contains unit and integrated tests of the Markov Chain Monte Carlo samplers.
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


/*
 * Guidelines:
 * ===========
 *   1. Make insides of "REQUIRE" statements as explicit as possible, so
 *      that failure messages are meaningful.
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

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/*
 * Check basic statistics routines
 */

TEST_CASE( "Generate multivariate normals and check statistics.", "[random][chain]" ) {
    // Draw multivariate normals, and add them to a chain
    shared_matrix sigma = std::make_shared<MatrixNxM>(2,2);
    *sigma << 1.0, -0.9,
              -0.9, 1.0;

    MultivariateNormalGenerator gen(sigma);
    Chain chain(2);

    shared_vector x = std::make_shared< std::vector<double> >(2, 0.);

    int n_samples = 100000;
    double sample_factor = 5.0 / sqrt((double)n_samples);

    for(int k=0; k<n_samples; k++) {
        gen.draw(x);
        chain.add(x->data(), 1.0, 0.0, 0.0);
    }

    // Check the mean of the chain
    shared_vector mu_measured = chain.calc_mean();
    for(auto m : *mu_measured) {
        // std::cerr << fabs(m) << " < " << 5.0 / sqrt((double)n_samples) << std::endl;
        REQUIRE( fabs(m) < sample_factor );
    }

    // Check the covariance of the chain
    shared_matrix sigma_measured = chain.calc_covariance();

    for(int j=0; j<2; j++) {
        for(int k=0; k<2; k++) {
            double tolerance = sqrt(fabs((*sigma)(j,j) * (*sigma)(k,k))) * sample_factor;// / sqrt(2.0);
            double delta = (*sigma_measured)(j,k) - (*sigma)(j,k);
            REQUIRE( fabs(delta) < tolerance );
        }
    }
}


TEST_CASE( "Generate uniform variates and check statistics.", "[random][chain]" ) {
    // Draw uniform variates, and add them to a chain
    BasicRandGenerator gen;
    Chain chain(3);

    std::vector<double> mu;
    for(int i=0; i<3; i++) { mu.push_back(gen.uniform()); }

    std::vector<double> x(3, 0.);

    int n_samples = 1000000;
    double sample_factor = 5.0 / sqrt(12. * (double)n_samples);

    for(int k=0; k<n_samples; k++) {
        for(int j=0; j<3; j++) {
            x.at(j) = gen.uniform() - 0.5 + mu.at(j);
        }
        chain.add(x.data(), 1.0, 0.0, 0.0);
    }

    // Check the mean of the chain
    shared_vector mu_measured = chain.calc_mean();
    double delta;

    for(int i=0; i<3; i++) {
        delta = mu_measured->at(i) - mu.at(i);
        // std::cerr << mu_measured->at(i) << " - " << mu.at(i) << " = "
        //           << delta << " < " << sample_factor << std::endl;
        REQUIRE( fabs(delta) < sample_factor );
    }

    // Check the covariance of the chain
    shared_matrix sigma_measured = chain.calc_covariance();
    double tolerance = 5.0 / (12. * sqrt((double)n_samples)); // / sqrt(2.0);

    for(int j=0; j<3; j++) {
        delta = 1./12. - (*sigma_measured)(j,j);
        REQUIRE ( fabs(delta) < tolerance );

        for(int k=j+1; k<3; k++) {
            delta = (*sigma_measured)(j,k);
            REQUIRE( fabs(delta) < tolerance );

            delta = (*sigma_measured)(k,j);
            REQUIRE( fabs(delta) < tolerance );
        }
    }
}


/*
 * End-to-end validation of samplers on test problems.
 */

TEST_CASE( "Sample from a multivariate Gaussian with the Metropolis-Hastings sampler.", "[sampler][mh]") {
    // Set up the probability density function
    const int n_dim = 3;
    shared_matrix sigma = std::make_shared<MatrixNxM>(n_dim, n_dim);
    *sigma <<  1.5, -0.8, 0.2,
              -0.8,  2.5, 0.3,
               0.2,  0.3, 0.5;
    shared_matrix inv_sigma = std::make_shared<MatrixNxM>(sigma->inverse());

    pdensity lnL = [inv_sigma, n_dim](double* x) {
        Eigen::VectorXd v = Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned>(x, n_dim);
        double res = -0.5 * v.transpose() * (*inv_sigma) * v;
        // std::cerr << "ln(L) = " << res << std::endl;
        return res;
    };

    pdensity ln_prior = [](double *x) {
        return 0.0;
    };

    // Set up sampler
    MHSampler sampler(lnL, ln_prior, n_dim);

    // Seed sampler position
    BasicRandGenerator r;
    std::vector<double> x0;
    for(int i=0; i<n_dim; i++) {
        x0.push_back(r.uniform() - 0.5);
    }
    sampler.set_state(x0);

    // Tune proposal distribution
    MHTuningParameters p;
    p.accept = 0.35;
    sampler.tune_all(p);

    // Sample the distribution
    const int n_samples = 100000;
    for(int i=0; i<n_samples; i++) {
        sampler.step();
    }

    // Check acceptance rate
    double delta = sampler.accept_frac() - p.accept;
    double tolerance = 0.25 * p.accept + 0.02;

    REQUIRE( fabs(delta) < tolerance );

    // Check covariance of chain
    std::shared_ptr<const Chain> chain = sampler.get_chain();
    shared_matrix sigma_measured = chain->calc_covariance();

    double n_samples_eff = p.accept * (double)n_samples;
    double sample_factor = 5.0 / sqrt((double)n_samples_eff);

    for(int j=0; j<n_dim; j++) {
        for(int k=0; k<n_dim; k++) {
            double tolerance = 0.05 + sqrt(fabs((*sigma)(j,j) * (*sigma)(k,k))) * sample_factor;
            double delta = (*sigma_measured)(j,k) - (*sigma)(j,k);
            REQUIRE( fabs(delta) < tolerance );
        }
    }

    // Check mean of chain
    shared_vector mu_measured = chain->calc_mean();
    for(auto m : *mu_measured) {
        // std::cerr << fabs(m) << " < " << 5.0 / sqrt((double)n_samples) << std::endl;
        REQUIRE( fabs(m) < sample_factor + 0.05 );
    }
}


TEST_CASE( "Sample from a multi-well potential using the Parallel Tempering sampler.", "[sampler][pt]") {
    // Basic parameters of the distribution
    const int n_dim = 2;
    const double gamma = 25.; // Higher -> modes more isolated

    // Create the ln(prior) and ln(likelihood) functions
    pdensity ln_prior = [](double* x) { return 0.; };

    // Double well along each axis
    BasicRandGenerator r;
    std::vector<double> mu;
    for(int i=0; i<n_dim; i++) {
        mu.push_back(10. * (r.uniform()-0.5));
    }

    pdensity lnL  = [lnL, n_dim, gamma, mu](double* x) {
        double ret = 0.;
        for(int i=0; i<n_dim; i++) {
            double dx = x[i] - mu.at(i);
            ret -= gamma * (dx*dx - 1.) * (dx*dx - 1.);
        }
        return ret;
    };

    // Set up the PT sampler
    const int n_temperatures = 5;
    double temperature_spacing = 4.; // Initially separate temperatures by this factor

    PTSampler sampler(lnL, ln_prior, n_dim,
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

    // Sample the distribution
    int n_swaps = 500000;
    int n_steps_per_swap = 10;  // # of Metropolis-Hastings steps between swaps
    sampler.step_multiple(n_swaps, n_steps_per_swap);

    // Check acceptance rate
    for(int temp_idx=0; temp_idx<n_temperatures; temp_idx++) {
        std::shared_ptr<const MHSampler> s = sampler.get_sampler(temp_idx);

        REQUIRE( fabs(s->accept_frac() - p.step_accept) < 0.25 * p.step_accept + 0.02 );
    }

    REQUIRE( fabs( sampler.swap_accept_frac() - p.swap_accept ) < 0.25 * p.swap_accept + 0.02 );
    
    // Checks on lowest-temperature sampler
    std::shared_ptr<const Chain> chain = sampler.get_chain(0);

    // Check mean of chain
    shared_vector mu_measured = chain->calc_mean();

    for(int i=0; i<mu.size(); i++) {
        REQUIRE( fabs(mu.at(i) - mu_measured->at(i)) < 0.10 );
    }

    // Check weight in each quadrant
    shared_const_vector x_sample = chain->get_elements();
    shared_const_vector w_sample = chain->get_weights();

    std::vector<double> W_quadrant(1 << n_dim, 0);

    for(auto x=x_sample->begin(), w=w_sample->begin(), w_end=w_sample->end();
        w!=w_end; ++w) {
        // Calculate quadrant index
        int idx = 0;
        for(int n=0; n<n_dim; n++, ++x) {
            idx += (1 << n) * (*x > mu.at(n) ? 0 : 1);
        }
        W_quadrant.at(idx) += *w;
    }

    double W_total = std::accumulate(W_quadrant.begin(), W_quadrant.end(), 0.);

    for(auto W : W_quadrant) {
        double quadrant_weight = (double)(1 << n_dim) * (W / W_total);
        REQUIRE( fabs(1. - quadrant_weight) < 0.10 );
    }
}
