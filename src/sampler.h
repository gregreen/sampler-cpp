
#ifndef _SAMPLER_H__
#define _SAMPLER_H__


#include <functional>
#include <vector>
#include <memory>

#include "custom_types.h"
#include "linalg.h"


class State {
public:
    State(uint n_dim);
    State(std::shared_ptr<const State> s);
    // ~State();

    std::vector<double> x;
    double lnL, ln_prior, lnp;

    void set(std::vector<double>& _x, double _lnL, double _ln_prior);
    void set(shared_vector _x, double _lnL, double _ln_prior);

    void copy(std::shared_ptr<const State> s);
};


class Chain {
public:
    Chain(uint n_dim);
    // ~Chain();

    // Mutators
    void add(double* x, double w, double lnL, double ln_prior);
    void add(std::shared_ptr<const State> s, double w);
    void increment_last(double dw);
    void clear();

    // Setters
    void set_capacity(size_t n);

    // Getters
    size_t get_n_dim() const;
    size_t get_length() const;
    size_t get_capacity() const;
    // const std::vector<double>& get_elements() const;
    // const std::vector<double>& get_weights() const;
    shared_const_vector get_elements() const;
    shared_const_vector get_weights() const;

    shared_matrix calc_covariance() const;
    shared_vector calc_mean() const;
    std::pair<shared_vector, shared_vector> calc_histogram(
        uint axis, double x0, double x1, uint n_bins, bool normalize=false) const;

private:
    uint _n_dim;
    size_t _length, _capacity;

    // std::vector<double> _x, _w, _lnL, _ln_prior;
    shared_vector _x, _w, _lnL, _ln_prior;
};


#endif /* end of include guard: _SAMPLER_H__ */
