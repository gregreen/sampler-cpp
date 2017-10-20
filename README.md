sampler-cpp
===========

A collection of Markov Chain Monte Carlo sampling routines, written in C++14.


Available Samplers
==================

* Metropolis-Hastings
* Parallel Tempering


Dependencies
============

This project depends on the linear algebra library [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page).

On Ubuntu, you can install Eigen3 with the command

    sudo apt-get install libeigen3-dev


Compilation
===========

This project uses `autotools`. Download the repository, and then from the base
directory, type the following into a terminal:

    autoreconf --install
    ./configure
    make

There should then be two binaries in the base directory: `ptdemo` and `runtests`.


Running the example
===================

After compiling the project, you should be able to run the Parallel Tempo demo with

    ./ptdemo

This example will sample a multimodal distribution, and drop the results in
ASCII files named `output/res_beta_?.txt`.

To plot the results, type

    python scripts/plot_pt_results.py output/res_beta_?.txt --output output/pt.png

The plots will be written to `output/pt_*.png`.

You can see the PT demo code in `examples/ptdemo.cpp`.


Running the tests
=================

This project comes with tests to verify that various components function as desired.
You can run these tests with

    ./runtests

You can see the testing code in the `tests/` directory.
