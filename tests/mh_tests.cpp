/* The MIT License (MIT)
 *
 * mh_tests.cpp
 * Contains unit and integrated tests of the Metropolis-Hastings sampler.
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

#include "catch.hpp"


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
