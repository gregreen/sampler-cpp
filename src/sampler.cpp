
#include "sampler.h"
namespace sampler {

// #include <iostream>

/*
 *  Constructor/Destructor
 */

State::State(uint n_dim) {
    x.reserve(n_dim);
    x.resize(n_dim);
}


State::State(std::shared_ptr<const State> s) {
    copy(s);
    // x.reserve(s->x.size());
    // std::copy(s->x.begin(), s->x.end(), std::back_inserter(x));
    // lnL = s->lnL;
    // ln_prior = s->ln_prior;
    // lnp = lnL + ln_prior;
}


void State::copy(std::shared_ptr<const State> s) {
    x.clear();
    x.reserve(s->x.size());
    std::copy(s->x.begin(), s->x.end(), std::back_inserter(x));
    lnL = s->lnL;
    ln_prior = s->ln_prior;
    lnp = lnL + ln_prior;
}


// State::~State() {
//     std::cout << "destroying State" << std::endl;
// }

Chain::Chain(uint n_dim)
    : _n_dim(n_dim)
{
    _length = 0;

    _x = std::make_shared<std::vector<double> >();
    // _x.reserve(_n_dim * _capacity);
    _w = std::make_shared<std::vector<double> >();
    // _w.reserve(_capacity);
    _lnL = std::make_shared<std::vector<double> >();
    // _lnL.reserve(_capacity);
    _ln_prior = std::make_shared<std::vector<double> >();
    // _ln_prior.reserve(_capacity);

    _capacity = 100;
    set_capacity(_capacity);
}

// Chain::~Chain() {}


/*
 *  Mutators
 */

void Chain::add(double* x, double w, double lnL, double ln_prior) {
    for(int i=0; i<_n_dim; i++) {
        _x->push_back(x[i]);
    }
    _w->push_back(w);
    _lnL->push_back(lnL);
    _ln_prior->push_back(ln_prior);
    _length++;
}

void Chain::add(std::shared_ptr<const State> s, double w) {
    for(int i=0; i<_n_dim; i++) {
        // std::cout << s->x.size() << " " << _n_dim << std::endl;
        double tmp = s->x[i];
        _x->push_back(tmp);
    }
    _w->push_back(w);
    _lnL->push_back(s->lnL);
    _ln_prior->push_back(s->ln_prior);
    _length++;
}

void Chain::increment_last(double dw) {
    if(_w->size() > 0) {
        _w->back() += dw;
    }
}

void Chain::clear() {
    _x->clear();
    _w->clear();
    _lnL->clear();
    _ln_prior->clear();
    _length = 0;
}


/*
 *  Setters
 */

void Chain::set_capacity(size_t n) {
    _x->reserve(n * _n_dim);
    _w->reserve(n);
    _lnL->reserve(n);
    _ln_prior->reserve(n);

    _capacity = n;
}

void State::set(std::vector<double>& _x, double _lnL, double _ln_prior) {
    x = _x;
    lnL = _lnL;
    ln_prior = _ln_prior;
    lnp = _lnL + _ln_prior;
}

void State::set(shared_vector _x, double _lnL, double _ln_prior) {
    // std::copy(_x->begin(), _x->end(), std::back_inserter(x));
    x = *_x;
    lnL = _lnL;
    ln_prior = _ln_prior;
    lnp = _lnL + _ln_prior;
}


/*
 *  Getters
 */

size_t Chain::get_n_dim() const {
    return _n_dim;
}

size_t Chain::get_length() const {
    return _length;
}

size_t Chain::get_capacity() const {
    return _capacity;
}

// const std::vector<double>& Chain::get_elements() const {
//     return _x;
// }
//
// const std::vector<double>& Chain::get_weights() const {
//     return _w;
// }

shared_const_vector Chain::get_elements() const {
    return _x;
}

shared_const_vector Chain::get_weights() const {
    return _w;
}

shared_const_vector Chain::get_lnL() const {
    return _lnL;
}

shared_const_vector Chain::get_ln_prior() const {
    return _ln_prior;
}


shared_matrix Chain::calc_covariance() const {
    return weighted_covariance(_x, _w, _n_dim);
}


shared_vector Chain::calc_mean() const {
    return weighted_mean(_x, _w, _n_dim);
}


std::pair<shared_vector, shared_vector> Chain::calc_histogram(
        uint axis, double x0, double x1, uint n_bins, bool normalize) const {
    return weighted_histogram(_x, _w, _n_dim, axis, x0, x1, n_bins, normalize);
}


} // namespace sampler
