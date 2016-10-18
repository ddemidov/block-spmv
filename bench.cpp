#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/backend/vexcl_static_matrix.hpp>

#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

#include <vexcl/vexcl.hpp>
#include <vexcl/sparse/ell.hpp>

template <int N, int M>
struct Block {
    typedef amgcl::static_matrix<double, N, M> type;
};

template <>
struct Block<1,1> {
    typedef double type;
};

template <int N, int M>
using block = typename Block<N,M>::type;

//---------------------------------------------------------------------------
template <class B>
int poisson3d(
        int n,
        std::vector<int> &ptr,
        std::vector<int> &col,
        std::vector< B > &val
        )
{
    namespace math = amgcl::math;

    int n3  = n * n * n;

    ptr.clear();
    col.clear();
    val.clear();

    ptr.reserve(n3 + 1);
    col.reserve(n3 * 7);
    val.reserve(n3 * 7);

    B one = math::constant<B>(1.0);
    ptr.push_back(0);
    for(int k = 0, idx = 0; k < n; ++k) {
        for(int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i, ++idx) {
                if (k > 0) {
                    col.push_back(idx - n * n);
                    val.push_back(-(1.0/6) * one);
                }

                if (j > 0) {
                    col.push_back(idx - n);
                    val.push_back(-(1.0/6) * one);
                }

                if (i > 0) {
                    col.push_back(idx - 1);
                    val.push_back(-(1.0/6) * one);
                }

                col.push_back(idx);
                val.push_back(1.0 * one);

                if (i + 1 < n) {
                    col.push_back(idx + 1);
                    val.push_back(-(1.0/6) * one);
                }

                if (j + 1 < n) {
                    col.push_back(idx + n);
                    val.push_back(-(1.0/6) * one);
                }

                if (k + 1 < n) {
                    col.push_back(idx + n * n);
                    val.push_back(-(1.0/6) * one);
                }

                ptr.push_back( col.size() );
            }
        }
    }

    return n3;
}

//---------------------------------------------------------------------------
template <int B>
vex::backend::kernel& spmv_kernel(vex::backend::command_queue &q) {
    using namespace vex;
    using namespace vex::detail;
    static kernel_cache cache;

    auto K = cache.find(q);
    if (K == cache.end()) {
        backend::source_generator src(q);

        src.kernel("spmv").open("(")
            .template parameter<int>("n")
            .template parameter<int>("ell_width")
            .template parameter<int>("ell_pitch")
            .template parameter< global_ptr<const int> >("ell_col")
            .template parameter< global_ptr<const double> >("ell_val")
            .template parameter< global_ptr<const double> >("X")
            .template parameter< global_ptr<double> >("Y")
            .close(")").open("{").grid_stride_loop().open("{");

        typedef block<B,B> A_type;
        typedef block<B,1> v_type;

        src.new_line() << type_name<A_type>() << " A;";
        src.new_line() << type_name<v_type>() << " x;";
        src.new_line() << type_name<v_type>() << " y = {};";

        src.new_line() << "for(int j = 0; j < ell_width; ++j)";
        src.open("{");
        src.new_line() << "int c = ell_col[j * ell_pitch + idx];";
        src.new_line() << "if (c == -1) break;";
        for(int k = 0; k < B*B; ++k)
            src.new_line() << "A(" << k << ") = ell_val[" << k << " * ell_pitch * ell_width + j * ell_pitch + idx];";
        for(int k = 0; k < B; ++k)
            src.new_line() << "x(" << k << ") = X[" << k << " * n + c];";
        src.new_line() << "y += A * x;";
        src.close("}");
        for(int k = 0; k < B; ++k)
            src.new_line() << "Y[" << k << " * n + idx] = y(" << k << ");";
        src.close("}");
        src.close("}");

        K = cache.insert(q, backend::kernel(q, src.str(), "spmv"));
    }

    return K->second;
}

//---------------------------------------------------------------------------
template <int B>
void run_benchmark(int m) {
    namespace math = amgcl::math;

    vex::Context ctx(vex::Filter::Env && vex::Filter::Count(1));
    std::cout << ctx << std::endl;

#if defined(VEXCL_BACKEND_CUDA)
    vex::push_compile_options(ctx, "-Xcompiler -std=c++03");
#endif

    amgcl::backend::enable_static_matrix_for_vexcl(ctx);

    vex::profiler<> prof(ctx);
    prof.tic_cpu("assemble");
    std::vector<int> ptr, col;
    std::vector< block<B,B> > val;
    int n = poisson3d(m, ptr, col, val);
    prof.toc("assemble");

    // VexCL
    {
        prof.tic_cl("vexcl");
        prof.tic_cl("transfer");
        vex::sparse::ell<block<B,B>,int> A(ctx, n, n, ptr, col, val);
        prof.toc("transfer");

        vex::vector<block<B,1>> x(ctx, n), y(ctx, n);

        x = math::constant<block<B,1>>(1.0);
        y = A * x;

        prof.tic_cl("spmv x100");
        for(int i = 0; i < 100; ++i)
            y = A * x;
        prof.toc("spmv x100");
        prof.toc("vexcl");
    }

    // CPU
    {
        prof.tic_cpu("cpu");

        std::vector<block<B,1>> x(n), y(n);

        for(auto &v : x) v = math::constant<block<B,1>>(1.0);

        prof.tic_cpu("spmv x100");
        for(int i = 0; i < 100; ++i)
            amgcl::backend::spmv(1.0, boost::tie(n, ptr, col, val), x, 0.0, y);
        prof.toc("spmv x100");
        prof.toc("cpu");
    }

    // Block-optimized
    {
        prof.tic_cl("custom");
        prof.tic_cl("transfer");
        vex::sparse::ell<block<B,B>,int> A(ctx, n, n, ptr, col, val);
        prof.toc("transfer");

        prof.tic_cl("convert");
        vex::vector<double> ell_val(ctx, A.ell_pitch * A.ell_width * B * B);
        typedef block<B,B> block_BB;
        VEX_FUNCTION(void, chidx, (int, i)(int, n)(block_BB*, v_in)(double*, v_out)(int, B),
                for(int j = 0, m = 0; j < B; ++j)
                    for(int k = 0; k < B; ++k, ++m)
                        v_out[m * n + i] = v_in[i](j,k);
                );
        vex::eval(chidx(
                    vex::element_index(0, ell_val.size()), A.ell_pitch * A.ell_width,
                    vex::raw_pointer(vex::vector<block<B,B>>(ctx.queue(0), A.ell_val)),
                    vex::raw_pointer(ell_val),
                    B),
                ctx, {0, A.ell_pitch * A.ell_width});
        prof.toc("convert");

        vex::vector<block<B,1>> x(ctx, n), y(ctx, n);
        x = math::constant<block<B,1>>(1.0);

        auto &K = spmv_kernel<B>(ctx.queue(0));
        K(ctx.queue(0), (int)n, (int)A.ell_width, (int)A.ell_pitch,
                A.ell_col, ell_val(0), x(0), y(0));

        prof.tic_cl("spmv x100");
        for(int i = 0; i < 100; ++i)
            K(ctx.queue(0), (int)n, (int)A.ell_width, (int)A.ell_pitch,
                    A.ell_col, ell_val(0), x(0), y(0));
        prof.toc("spmv x100");
        prof.toc("custom");
    }

    std::cout << prof << std::endl;
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
         po::value<int>()->default_value(4),
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

    switch(vm["block-size"].as<int>()) {
        case 1:
            run_benchmark<1>(vm["size"].as<int>());
            break;
	case 2:
            run_benchmark<2>(vm["size"].as<int>());
            break;
	case 3:
            run_benchmark<3>(vm["size"].as<int>());
            break;
        case 4:
            run_benchmark<4>(vm["size"].as<int>());
            break;
	case 5:
            run_benchmark<5>(vm["size"].as<int>());
            break;
        default:
            std::cerr << "Unsupported block size" << std::endl;
            return 1;
    }
}
