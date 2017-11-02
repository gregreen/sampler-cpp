#ifndef _CHAIN_H__
#define _CHAIN_H__
namespace sampler {

class Chain {
public:
    Chain(uint n_dim);
    ~Chain();

private:
    uint _n_dim;
    std::vector<double> _x, _w;
}

} // namespace sampler
#endif /* end of include guard: _CHAIN_H__ */
