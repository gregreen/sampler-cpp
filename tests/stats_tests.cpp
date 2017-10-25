/* The MIT License (MIT)
 *
 * stats_tests.cpp
 * Contains unit and integrated tests of the basic statistics routines.
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
