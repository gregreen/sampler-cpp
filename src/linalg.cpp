
#include "linalg.h"


shared_vector weighted_mean(shared_const_vector x,
                            shared_const_vector w,
                            int n_dim) {
    shared_vector x_mean = std::make_shared<std::vector<double> >(n_dim, 0.);
    double w_tot = 0.;

    for(int nw=0, nx=0; nw<w->size(); nw++, nx+=n_dim) {
        w_tot += (*w)[nw];

        for(int j=0; j<n_dim; j++) {
            // Shifted mean calculation, using first element
            (*x_mean)[j] += (*w)[nw] * ((*x)[nx+j] - (*x)[j]);
        }
    }

    for(int j=0; j<n_dim; j++) {
        (*x_mean)[j] /= w_tot;
        (*x_mean)[j] += (*x)[j];
    }

    return x_mean;
}

shared_matrix weighted_covariance(shared_const_vector x,
                                  shared_const_vector w,
                                  int n_dim) {
    // Calculate the mean
    shared_vector x_mean = weighted_mean(x, w, n_dim);

    // Allocate covariance matrix
    shared_matrix cov = std::make_shared<MatrixNxM>(n_dim, n_dim);
    cov->setZero();

    // Calculate weighted sums
    double w_tot;
    for(int j=0; j<n_dim; j++) {
        for(int k=0; k<=j; k++) {
            w_tot = 0.;
            for(int nw=0, nx=0; nw<w->size(); nw++, nx+=n_dim) {
                w_tot += (*w)[nw];
                (*cov)(j,k) += (*w)[nw]
                            * ((*x)[nx+j] - (*x_mean)[j])
                            * ((*x)[nx+k] - (*x_mean)[k]);
            }
        }
    }

    // Normalize and symmetrize matrix
    for(int j=0; j<n_dim; j++) {
        for(int k=0; k<j; k++) {
            (*cov)(j,k) /= w_tot - 1.;
            (*cov)(k,j) = (*cov)(j,k);
        }
        (*cov)(j,j) /= w_tot - 1.;
    }

    return cov;
}


std::pair<shared_vector, shared_vector> weighted_histogram(
        shared_const_vector x, shared_const_vector w, int n_dim,
        uint axis, double x0, double x1, uint n_bins,
        bool normalize) {
    // Create return arrays
    shared_vector edges = std::make_shared< std::vector<double> >();
    shared_vector hist = std::make_shared< std::vector<double> >(n_bins, 0.);

    // Calculate bin edges
    double dx = (x1 - x0) / (double)n_bins;
    for(int i=0; i<n_bins+1; i++) {
        edges->push_back(x0 + dx*i);
    }

    // Assign weights to bins
    for(int i=0; i<w->size(); i++) {
        double x_axis = x->at(n_dim*i+axis);
        int bin = int((x_axis-x0)/dx);
        if((bin > 0) && (bin < n_bins)) {
            hist->at(bin) += w->at(i);
        }
    }

    // Normalize sum of histogram to unity
    if(normalize) {
        double W = std::accumulate(hist->begin(), hist->end(), 0.);
        for(auto && h : *hist) { h /= W; }
    }

    // Return pair: (bin edges, histogram values)
    return std::make_pair(edges, hist);
}
