
#include "ptsampler.h"
namespace sampler {


pdensity apply_temperature(pdensity lnp, double beta) {
    /*
     * Takes a log probability density function, and returns a log probability
     * density function with an inverse temperature beta multiplied in. This
     * is equivalent to raising the probability density to the beta power.
     */
    return [lnp, beta](double* x) {
        return beta * lnp(x);
    };
}


PTSampler::PTSampler(pdensity lnL, pdensity ln_prior,
                     uint n_dim, std::vector<double>& beta)
    : _lnL(lnL), _ln_prior(ln_prior), _n_dim(n_dim),
      _n_swaps_accepted(0.), _n_swaps_rejected(0.)
{
    _beta = std::make_shared< std::vector<double> >(beta);
    _n_temps = beta.size();
    _init_samplers();
}


PTSampler::PTSampler(pdensity lnL, pdensity ln_prior,
                     uint n_dim, uint n_temperatures,
                     double temperature_spacing)
    : _lnL(lnL), _ln_prior(ln_prior), _n_dim(n_dim),
      _n_swaps_accepted(0.), _n_swaps_rejected(0.),
      _n_temps(n_temperatures)
{
    _beta = std::make_shared< std::vector<double> >();
    double beta_spacing = 1. / temperature_spacing;
    double b = 1.;
    for(int i=0; i<n_temperatures; i++, b*=beta_spacing) {
        _beta->push_back(b);
    }
    _init_samplers();
}


PTSampler::~PTSampler() {}


void PTSampler::_init_samplers() {
    for(auto b : *_beta) {
        // Add a Metropolis-Hastings sampler with a modified log probability
        // density function, which has an inverse temperature multiplied in.
        _sampler.push_back(std::make_shared<MHSampler>(
            apply_temperature(_lnL, b),
            apply_temperature(_ln_prior, b),
            _n_dim
        ));
    }
}


std::shared_ptr<const Chain> PTSampler::get_chain(uint idx) const {
    return _sampler.at(idx)->get_chain();
    // return _chains.at(idx);
}


std::shared_ptr<const MHSampler> PTSampler::get_sampler(uint idx) const {
    return _sampler.at(idx);
}


void PTSampler::set_state(uint idx, std::vector<double>& x0) {
    _sampler.at(idx)->set_state(x0);
}

void PTSampler::set_state(uint idx, shared_vector x0) {
    _sampler.at(idx)->set_state(x0);
}

void PTSampler::set_state(vector_generator v) {
    for(auto s : _sampler) {
        s->set_state(v);
    }
}


void PTSampler::clear_chain(bool clear_swap_stats) {
    for(auto s : _sampler) {
        s->clear_chain();
    }
    if(clear_swap_stats) {
        _n_swaps_accepted = 0;
        _n_swaps_rejected = 0;
    }
}


void PTSampler::update_proposal(double persistence, double epsilon) {
    for(auto s : _sampler) {
        s->update_proposal(persistence, epsilon);
    }
}


void PTSampler::tune_step_size(double accept_target, double granularity) {
    for(auto s : _sampler) {
        s->tune_step_size(accept_target, granularity);
    }
}

void PTSampler::_alter_beta_spacing(double factor) {
    for(auto && b : *_beta) {
        // Multiply log(beta) by factor
        b = pow(b, factor);
    }
}

void PTSampler::tune_beta(double accept_target, double granularity) {
    double a = swap_accept_frac();
    if(a < 0.8 * accept_target) {
        // Decrease beta spacing
        _alter_beta_spacing(1. - granularity);
    } else if(a > 1.2 * accept_target) {
        // Increase beta spacing
        _alter_beta_spacing(1. + granularity);
    }
}

// int n_rounds=50;
// int n_swaps_per_round=100;
// int n_steps_per_swap=50;
// double step_accept=0.25;
// double swap_accept=0.25;
// double step_granularity=0.05;
// double swap_granularity=0.05;
// double prop_persist=0.95;
// double prop_epsilon=1.e-5;

void PTSampler::tune_all() {
    PTTuningParameters p;
    tune_all(p);
}

void PTSampler::tune_all(const PTTuningParameters& p) {
    for(int n=0; n<p.n_rounds; n++) {
        for(int i=0; i<p.n_swaps_per_round; i++) {
            // Take multiple Metropolis-Hastings steps in each
            // single-temperature sampler, and then tune either
            // the step size or the proposal covariance.
            for(int k=0; k<p.n_steps_per_swap; k++) {
                step();
            }
            if(i & 1) {
                tune_step_size(p.step_accept, p.step_granularity);
            } else {
                update_proposal(p.prop_persist, p.prop_epsilon);
            }

            // Take a swap step
            swap();

            // Clear the chain, but keep swap statistics intact.
            clear_chain(false);
        }

        // Tune the temperature spacing
        tune_beta(p.swap_accept, p.swap_granularity);

        // Clear the chain, and clear swap statistics
        clear_chain();
    }
}


void PTSampler::step() {
    // for(std::vector<std::shared_ptr<MHSampler> >::iterator it = _sampler.begin(); it != _sampler.end(); ++it) {
    //     (*it)->step();
    // }
    for(auto it : _sampler) {
        it->step();
    }
}


void PTSampler::swap() {
    if(_n_temps == 1) { return; }

    // Choose which temperatures to swap
    int idx_low = _r.uniform_int(0, _n_temps-2);

    // Calculate acceptance probability
    std::shared_ptr<const State> state_low = _sampler.at(idx_low)->get_state();
    std::shared_ptr<const State> state_high = _sampler.at(idx_low+1)->get_state();

    double beta_low = _beta->at(idx_low);
    double beta_high = _beta->at(idx_low+1);
    double lnp_low = state_low->lnp / beta_low;
    double lnp_high = state_high->lnp / beta_high;

    double ln_alpha = (beta_high-beta_low) * (lnp_low-lnp_high);

    double beta_factor_tmp;

    if((ln_alpha > 0.) || (ln_alpha > log(_r.uniform()))) {
        // Swap states
        std::shared_ptr<State> state_low_copy = std::make_shared<State>(state_low);
        std::shared_ptr<State> state_high_copy = std::make_shared<State>(state_high);

        // Have to fix log probabilities before swapping, since they have the
        // wrong betas multiplied in.
        beta_factor_tmp = beta_high / beta_low;
        state_low_copy->lnL *= beta_factor_tmp;
        state_low_copy->ln_prior *= beta_factor_tmp;
        state_low_copy->lnp *= beta_factor_tmp;

        beta_factor_tmp = beta_low / beta_high;
        state_high_copy->lnL *= beta_factor_tmp;
        state_high_copy->ln_prior *= beta_factor_tmp;
        state_high_copy->lnp *= beta_factor_tmp;

        _sampler.at(idx_low)->force_step(state_high_copy);
        _sampler.at(idx_low+1)->force_step(state_low_copy);

        _n_swaps_accepted++;
    } else {
        // Increase multiplicity of current state by one
        _sampler.at(idx_low)->null_step();
        _sampler.at(idx_low+1)->null_step();

        _n_swaps_rejected++;
    }
}


void PTSampler::step_multiple(int n_swaps, int n_steps_per_swap) {
    for(int k=0; k<n_swaps; k++) {
        for(int i=0; i<n_steps_per_swap; i++) {
            step();
        }
        swap();
    }
}


double PTSampler::swap_accept_frac() const {
    return (double)_n_swaps_accepted / (double)(_n_swaps_accepted + _n_swaps_rejected);
}

shared_const_vector PTSampler::get_beta() const {
    return _beta;
}


} // namespace sampler
