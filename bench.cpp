#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/adapter/block_matrix.hpp>

#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

#include <vexcl/vexcl.hpp>
#include <vexcl/sparse/ell.hpp>

//---------------------------------------------------------------------------
int poisson3d(int n, std::vector<int> &ptr, std::vector<int> &col, std::vector<double> &val) {
    int n3  = n * n * n;

    ptr.clear();
    col.clear();
    val.clear();

    ptr.reserve(n3 + 1);
    col.reserve(n3 * 7);
    val.reserve(n3 * 7);

    ptr.push_back(0);
    for(int k = 0, idx = 0; k < n; ++k) {
        for(int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i, ++idx) {
                if (k > 0) {
                    col.push_back(idx - n * n);
                    val.push_back(-0.25);
                }

                if (j > 0) {
                    col.push_back(idx - n);
                    val.push_back(-0.25);
                }

                if (i > 0) {
                    col.push_back(idx - 1);
                    val.push_back(-0.25);
                }

                col.push_back(idx);
                val.push_back(1.0);

                if (i + 1 < n) {
                    col.push_back(idx + 1);
                    val.push_back(-0.25);
                }

                if (j + 1 < n) {
                    col.push_back(idx + n);
                    val.push_back(-0.25);
                }

                if (k + 1 < n) {
                    col.push_back(idx + n * n);
                    val.push_back(-0.25);
                }

                ptr.push_back( col.size() );
            }
        }
    }

    return n3;
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    namespace po = boost::program_options;
    namespace io = amgcl::io;

    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "Show this help.")
        (
         "block-size,b",
         po::value<int>()->default_value(1),
         "The block size of the system matrix. "
         "When specified, the system matrix is assumed to have block-wise structure. "
         "This usually is the case for problems in elasticity, structural mechanics, "
         "for coupled systems of PDE (such as Navier-Stokes equations), etc. "
         "Valid choices are 2, 3, 4, and 6."
        )
        (
         "size,n",
         po::value<int>()->default_value(100),
         "The size of the Poisson problem to solve when no system matrix is given. "
         "Specified as number of grid nodes along each dimension of a unit cube. "
         "The resulting system will have n*n*n unknowns. "
        )
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    vex::Context ctx(vex::Filter::Env && vex::Filter::Count(1));
    std::cout << ctx << std::endl;

    vex::profiler<> prof(ctx);
    prof.tic_cpu("assemble");
    std::vector<int> ptr, col;
    std::vector<double> val;
    int n = poisson3d(vm["size"].as<int>(), ptr, col, val);
    prof.toc("assemble");

    prof.tic_cl("transfer");
    vex::sparse::ell<double,int> A(ctx, n, n, ptr, col, val);
    prof.toc("transfer");

    vex::vector<double> x(ctx, n), y(ctx, n);

    x = 1;
    y = A * x;

    prof.tic_cl("spmv (scalar) x100");
    for(int i = 0; i < 100; ++i)
        y = A * x;
    prof.toc("spmv (scalar) x100");

    std::cout << prof << std::endl;
}
