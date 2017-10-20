
#include "rand_helpers.h"


BasicRandGenerator::BasicRandGenerator() {
    // Seed the pseudorandom number generator with real entropy
    std::random_device _rd;
    _rand_gen.seed(_rd());
}


double BasicRandGenerator::uniform() {
    return _dist_uniform(_rand_gen);
}


double BasicRandGenerator::uniform(double low, double high) {
    return low + (high-low) * uniform();
}


int BasicRandGenerator::uniform_int(int min, int max) {
    std::uniform_int_distribution<> _dist_uniform_int(min, max);
    return _dist_uniform_int(_rand_gen);
}


double BasicRandGenerator::normal() {
    return _dist_normal(_rand_gen);
}


double BasicRandGenerator::normal(double mu, double sigma) {
    return mu + sigma * normal();
}


/*
 *  MultivariateNormalGenerator
 */


MultivariateNormalGenerator::MultivariateNormalGenerator(uint n_dim)
    : _n_dim(n_dim)
{
    // Initialize to unit variance along all axes,
    // no correlations, and zero mean.
    _sigma = std::make_shared<MatrixNxM>(n_dim, n_dim);
    _sigma->setIdentity();
    _sqrt_sigma = std::make_shared<MatrixNxM>(n_dim, n_dim);
    _sqrt_sigma->setIdentity();
    // _mu = std::make_shared<Eigen::VectorXd>(n_dim);
    // _mu->setZero();

    _vec_scratch = std::make_shared<Eigen::VectorXd>(n_dim);
    // _mu = std::make_shared<std::vector<double> >(n_dim, 0.);
}


MultivariateNormalGenerator::MultivariateNormalGenerator(shared_const_matrix sigma) { //, shared_const_vector mu) {
    _n_dim = sigma->rows();

    // Check shapes of sigma and mu
    if(sigma->cols() != _n_dim) {
        throw std::invalid_argument("The matrix <sigma> must be square.");
    } //else if(mu->size() != _n_dim) {
    //     throw std::invalid_argument("The vector <mu> and matrix <sigma> must have compatible shapes.");
    // }

    // Copy sigma and mu into memory
    _sigma = std::make_shared<MatrixNxM>(_n_dim, _n_dim);
    // _mu = std::make_shared<Eigen::VectorXd>(_n_dim);
    // _sigma = std::make_shared<MatrixNxM>(*sigma);
    // _mu = std::make_shared<Eigen::VectorXd>(
    //     Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned>(mu->data(), mu->size())
    // );

    _sqrt_sigma = std::make_shared<MatrixNxM>(_n_dim, _n_dim);
    _vec_scratch = std::make_shared<Eigen::VectorXd>(_n_dim);

    set_covariance(sigma);
    // set_mean(mu);
}


// void MultivariateNormalGenerator::set_mean(shared_const_vector mu) {
//     // Check size of mu
//     if(mu->size() != _n_dim) {
//         throw std::invalid_argument("The vector <mu> has the wrong size.");
//     }
//     // Copy mu
//     *_mu = Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned>(mu->data(), mu->size());
// }


void MultivariateNormalGenerator::set_covariance(shared_const_matrix sigma) {
    // Check shape of sigma
    if((sigma->rows() != _n_dim) || (sigma->cols() != _n_dim)) {
        throw std::invalid_argument("The matrix <sigma> has the wrong shape.");
    }
    // Copy sigma
    *_sigma = *sigma;

    // Find the square-root of sigma
    Eigen::SelfAdjointEigenSolver<MatrixNxM> eigen_solver(*_sigma);
    *_sqrt_sigma = eigen_solver.eigenvectors() * eigen_solver.eigenvalues().cwiseSqrt().asDiagonal();
}


void MultivariateNormalGenerator::draw(shared_vector x) {
    // Generate a multivariate normal vector
    *_vec_scratch = *_sqrt_sigma * _vec_scratch->unaryExpr([&](auto z) { return _r.normal(); });

    // Copy the data into x
    x->resize(_n_dim);
    Eigen::VectorXd::Map(x->data(), _vec_scratch->size()) = *_vec_scratch;
}


void MultivariateNormalGenerator::draw(std::vector<double>& x) {
    // Generate a multivariate normal vector
    *_vec_scratch = *_sqrt_sigma * _vec_scratch->unaryExpr([&](auto z) { return _r.normal(); });

    // Copy the data into x
    x.resize(_n_dim);
    Eigen::VectorXd::Map(x.data(), _vec_scratch->size()) = *_vec_scratch;
}


Eigen::VectorXd MultivariateNormalGenerator::draw() {
    // Generate a multivariate normal vector
    return *_sqrt_sigma * _vec_scratch->unaryExpr([&](auto z) { return _r.normal(); });
}


shared_const_matrix MultivariateNormalGenerator::get_covariance() const {
    return _sigma;
}
