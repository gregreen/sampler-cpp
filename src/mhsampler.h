
#ifndef _MHSAMPLER_H__
#define _MHSAMPLER_H__


#include <iostream>

#include <vector>
#include <random>
#include <functional>
#include <memory>
#include <math.h>
#include <cmath>
#include <cstdint>

#include "sampler.h"
#include "linalg.h"
#include "rand_helpers.h"
#include "custom_types.h"


struct MHTuningParameters {
    int n_rounds=100;
    int n_steps_per_round=100;
    double accept=0.25;
    double granularity=0.05;
    double prop_persist=0.95;
    double prop_epsilon=1.e-5;
};


class MHSampler {
public:
    MHSampler(pdensity lnL, pdensity ln_prior, uint n_dim);
    // ~MHSampler();

    // Mutators
    void step();

    void set_state(shared_vector x0);
    void set_state(std::vector<double>& x0);
    void set_state(double* x0);
    void set_state(vector_generator v);

    void clear_chain();

    void update_proposal(double persistence=0.95,
                         double epsilon=1.e-5);
    void tune_step_size(double accept_target=0.25,
                        double granularity=0.1);

    void tune_all(const MHTuningParameters& p);

    void force_step(std::shared_ptr<const State> s);
    void null_step();

    // Getters
    std::shared_ptr<const Chain> get_chain() const;
    std::shared_ptr<const State> get_state() const;

    shared_const_matrix get_proposal_cov() const;
    double get_step_size() const;

    double accept_frac() const;

private:
    // Probability density functions
    pdensity _lnL, _ln_prior;

    // Dimensionality
    uint _n_dim;

    // Markov chain
    std::shared_ptr<Chain> _chain;

    // Current/proposed state
    std::shared_ptr<State> _current, _proposed;

    // Step size
    double _step_size;

    // Statistics
    std::uint64_t _n_accepted, _n_rejected;

    // Random number generators
    BasicRandGenerator _r;
    MultivariateNormalGenerator _normal;
    // std::default_random_engine _rand_gen;
    // std::normal_distribution<> _rand_normal;
    // std::uniform_real_distribution<> _rand_uniform;

    // Accept/reject proposed new state
    void _accept_proposal();
    void _reject_proposal();
};



#endif /* end of include guard: _MHSAMPLER_H__ */
