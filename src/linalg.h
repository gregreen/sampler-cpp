
#ifndef _LINALG_H__
#define _LINALG_H__

#include <vector>
#include <memory>
#include <Eigen/Dense>

#include "custom_types.h"


shared_vector weighted_mean(shared_const_vector x,
                            shared_const_vector w,
                            int n_dim);

shared_matrix weighted_covariance(shared_const_vector x,
                                  shared_const_vector w,
                                  int n_dim);

shared_matrix matrix_squareroot(shared_const_matrix M);

std::pair<shared_vector, shared_vector> weighted_histogram(
        shared_const_vector x, shared_const_vector w, int n_dim,
        uint axis, double x0, double x1, uint n_bins,
        bool normalize=false);


#endif /* end of include guard: _LINALG_H__ */
