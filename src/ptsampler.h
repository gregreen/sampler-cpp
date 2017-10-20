
#ifndef _PTSAMPLER_H__
#define _PTSAMPLER_H__


#include <vector>
#include <functional>

#include "sampler.h"
#include "mhsampler.h"


pdensity apply_temperature(pdensity lnp, double beta);


struct PTTuningParameters {
    int n_rounds=50;
    int n_swaps_per_round=100;
    int n_steps_per_swap=50;
    double step_accept=0.25;
    double swap_accept=0.25;
    double step_granularity=0.05;
    double swap_granularity=0.05;
    double prop_persist=0.95;
    double prop_epsilon=1.e-5;
};


class PTSampler {
public:
    PTSampler(pdensity lnL, pdensity ln_prior,
              uint n_dim, std::vector<double>& beta);
    PTSampler(pdensity lnL, pdensity ln_prior,
              uint n_dim, uint n_temperatures,
              double temperature_spacing);
    ~PTSampler();

    // Mutators
    void step();    // Step in each single-temperature sampler
    void swap();    // Swap step between two random temperatures

    // Interleaved swaps and Metropolis-Hastings steps
    void step_multiple(int n_swaps, int n_steps_per_swap);

    void set_state(vector_generator v);
    void set_state(uint idx, shared_vector x0);
    void set_state(uint idx, std::vector<double>& x0);

    // Clear Markov Chain, and clear acceptance statistics
    void clear_chain(bool clear_swap_stats=true);

    // Update proposal covariance of each single-temperature sampler
    void update_proposal(double persistence=0.95,
                         double epsilon=1.e-5);

    // Single tuning of step size of each single-temperature sampler
    void tune_step_size(double accept_target=0.25,
                        double granularity=0.05);

    // Single tuning of temperature ladder
    void tune_beta(double accept_target=0.25,
                   double granularity=0.05);

    // Tune both single-temperature samplers and temperature ladder,
    // using an iterative scheme.
    void tune_all(const PTTuningParameters& p);
    void tune_all();    // Use default tuning parameters


    // Getters
    std::shared_ptr<const Chain> get_chain(uint idx) const;
    std::shared_ptr<const MHSampler> get_sampler(uint idx) const;

    double swap_accept_frac() const;
    shared_const_vector get_beta() const;

private:
    shared_vector _beta;

    // std::vector<double> _ln_beta_prior;
    pdensity _lnL, _ln_prior, _lnp;
    uint _n_dim, _n_temps;

    // Statistics
    std::uint64_t _n_swaps_accepted, _n_swaps_rejected;

    // Random number generator
    BasicRandGenerator _r;

    // Ensemble of samplers at different temperatures
    std::vector<std::shared_ptr<MHSampler> > _sampler;

    // Private methods
    void _init_samplers();
    void _alter_beta_spacing(double factor);
};



#endif /* end of include guard: _PTSAMPLER_H__ */
