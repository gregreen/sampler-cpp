#ifndef _CUSTOM_TYPES_H__
#define _CUSTOM_TYPES_H__

#include <vector>
#include <memory>
#include <Eigen/Dense>


namespace sampler {


// Probability density function. f : R^N -> R.
typedef std::function<double(double*)> pdensity;

// Vector of doubles (using std::vector): R^N.
typedef std::shared_ptr<std::vector<double> > shared_vector;
typedef std::shared_ptr<const std::vector<double> > shared_const_vector;

// Vector of doubles using the Eigen library
typedef std::shared_ptr<Eigen::VectorXd> shared_vector_eig;
typedef std::shared_ptr<const Eigen::VectorXd> shared_const_vector_eig;

// NxM matrix (dense)
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixNxM;
typedef std::shared_ptr<MatrixNxM> shared_matrix;
typedef std::shared_ptr<const MatrixNxM> shared_const_matrix;

// Function that outputs a shared vector: f : -> R^N
typedef std::function<shared_vector()> vector_generator;


} // namespace sampler
#endif /* end of include guard: _CUSTOM_TYPES_H__ */
