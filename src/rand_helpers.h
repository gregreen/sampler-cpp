#ifndef _RAND_HELPERS_H__
#define _RAND_HELPERS_H__

#include <random>
#include <stdexcept>

#include "custom_types.h"


namespace sampler {


class BasicRandGenerator {
public:
    BasicRandGenerator();

    // Generate numbers from different distributions
    double uniform();
    double uniform(double low, double high);

    int uniform_int(int min, int max);

    double normal();
    double normal(double mu, double sigma);

private:
    std::default_random_engine _rand_gen;
    std::normal_distribution<> _dist_normal;
    std::uniform_real_distribution<> _dist_uniform;
};


class MultivariateNormalGenerator {
public:
    MultivariateNormalGenerator(uint n_dim);
    MultivariateNormalGenerator(shared_const_matrix sigma);//, shared_const_vector mu);

    void set_covariance(shared_const_matrix sigma);
    // void set_mean(shared_const_vector mu);

    void draw(shared_vector x);
    void draw(std::vector<double>& x);
    Eigen::VectorXd draw();
    // shared_vector draw();

    shared_const_matrix get_covariance() const;

private:
    BasicRandGenerator _r;
    uint _n_dim;

    shared_matrix _sigma;
    shared_matrix _sqrt_sigma;
    // shared_vector_eig _mu;
    shared_vector_eig _vec_scratch;
};


} // namespace sampler
#endif // _RAND_HELPERS_H__
