#ifndef _CHAIN_H__
#define _CHAIN_H__


class Chain {
public:
    Chain(uint n_dim);
    ~Chain();

private:
    uint _n_dim;
    std::vector<double> _x, _w;
}


#endif /* end of include guard: _CHAIN_H__ */
