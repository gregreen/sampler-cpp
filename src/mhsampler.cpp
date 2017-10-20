
#include "mhsampler.h"


MHSampler::MHSampler(pdensity lnL, pdensity ln_prior, uint n_dim)
    : _lnL(lnL), _ln_prior(ln_prior),
      _n_dim(n_dim), _normal(n_dim),
      _step_size(0.25)
{
    _chain = std::make_shared<Chain>(n_dim);
    _current = std::make_shared<State>(n_dim);
    _proposed = std::make_shared<State>(n_dim);

    // Seed the pseudorandom number generator with real entropy
    // std::random_device _rd;
    // _rand_gen.seed(_rd());

    // Initialize statistics
    _n_accepted = 0;
    _n_rejected = 0;
}

// MHSampler::~MHSampler() {}


/*
 *  Mutators
 */

void MHSampler::step() {
    // Generate a proposal
    _normal.draw(_proposed->x);

    for(int i=0; i<_n_dim; i++) {
        _proposed->x[i] *= _step_size;
        _proposed->x[i] += _current->x[i];// + step_size * _r.normal();
    }
    _proposed->lnL = _lnL(_proposed->x.data());
    _proposed->ln_prior = _ln_prior(_proposed->x.data());
    _proposed->lnp = _proposed->lnL + _proposed->ln_prior;

    // Metropolis-Hastings acceptance rule
    if(_proposed->lnp > _current->lnp) {
        _accept_proposal();
    } else if(exp(_proposed->lnp - _current->lnp) > _r.uniform()) {
        _accept_proposal();
    } else if(std::isinf(_current->lnp) && !std::isinf(_proposed->lnp)) {
        _accept_proposal();
    } else {
        _reject_proposal();
    }
}

void MHSampler::_accept_proposal() {
    _current.swap(_proposed);
    _chain->add(_current, 1.0);
    _n_accepted++;
}

void MHSampler::_reject_proposal() {
    _chain->increment_last(1.0);
    _n_rejected++;
}


void MHSampler::force_step(std::shared_ptr<const State> s) {
    // Copy state s into the proposed state and then accept it
    _proposed->copy(s);
    _accept_proposal();
}


void MHSampler::null_step() {
    _chain->increment_last(1.0);
}


void MHSampler::set_state(std::vector<double>& x0) {
    double lnL = _lnL(x0.data());
    double ln_prior = _ln_prior(x0.data());
    _current->set(x0, lnL, ln_prior);
}


void MHSampler::set_state(shared_vector x0) {
    double lnL = _lnL(x0->data());
    double ln_prior = _ln_prior(x0->data());
    _current->set(x0, lnL, ln_prior);
}


void MHSampler::set_state(vector_generator v) {
    shared_vector x0 = v();
    set_state(x0);
}


void MHSampler::clear_chain() {
    _chain->clear();
    _n_accepted = 0;
    _n_rejected = 0;
}


void MHSampler::update_proposal(double persistence, double epsilon) {
    shared_matrix cov_tmp = _chain->calc_covariance();

    // Check for NaNs in the new covariance matrix
    for(int i=0; i<cov_tmp->rows(); i++) {
        for(int j=0; j<cov_tmp->cols(); j++) {
            if(!std::isfinite((*cov_tmp)(i,j))) { return; }
        }
    }

    *cov_tmp *= (1. - persistence);
    *cov_tmp += persistence * *_normal.get_covariance();
    cov_tmp->diagonal().array() += epsilon;
    _normal.set_covariance(cov_tmp);//_chain->calc_covariance());
}


void MHSampler::tune_step_size(double accept_target, double granularity) {
    double a = accept_frac();
    if(a < 0.8 * accept_target) {
        _step_size *= 1. - granularity;
        // std::cerr << "Decreasing step size to " << _step_size << std::endl;
    } else if(a > 1.2 * accept_target) {
        _step_size *= 1. + granularity;
        // std::cerr << "Increasing step size to " << _step_size << std::endl;
    }
}


/*
 *  Getters
 */

std::shared_ptr<const Chain> MHSampler::get_chain() const {
    return _chain;
}


std::shared_ptr<const State> MHSampler::get_state() const {
    return _current;
}


shared_const_matrix MHSampler::get_proposal_cov() const {
    return _normal.get_covariance();
}


double MHSampler::accept_frac() const {
    return (double)_n_accepted / (double)(_n_accepted + _n_rejected);
}
