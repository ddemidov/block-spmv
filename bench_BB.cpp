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
                    val.push_back(-0.25 * one);
                }

                if (j > 0) {
                    col.push_back(idx - n);
                    val.push_back(-0.25 * one);
                }

                if (i > 0) {
                    col.push_back(idx - 1);
                    val.push_back(-0.25 * one);
                }

                col.push_back(idx);
                val.push_back(1.0 * one);

                if (i + 1 < n) {
                    col.push_back(idx + 1);
                    val.push_back(-0.25 * one);
                }

                if (j + 1 < n) {
                    col.push_back(idx + n);
                    val.push_back(-0.25 * one);
                }

                if (k + 1 < n) {
                    col.push_back(idx + n * n);
                    val.push_back(-0.25 * one);
                }

                ptr.push_back( col.size() );
            }
        }
    }

    return n3;
}

//---------------------------------------------------------------------------
template <int B>
vex::backend::kernel& blocked_spmv_kernel(vex::backend::command_queue &q) {
    using namespace vex;
    using namespace vex::detail;
    static kernel_cache cache;

    auto K = cache.find(q);
    if (K == cache.end()) {
        backend::source_generator src(q);

         /* The following kernel uses B*B threads for each B*B block -> continguous memory reads for A
         *  - Performance for B=1 is similar (within 10 percent) to 'normal' kernel
         *  - Performance for B=2 is about 2x slower than the 'naive' kernel
         *  - Performance for B=4 is 10x faster than the 'naive' kernel, because it avoids strided memory access
         * Performances are from a Tesla C2070.
         */
       src.kernel("blocked_spmv").open("(")
            .template parameter<int>("N")
            .template parameter<int>("ell_width")
            .template parameter<int>("ell_pitch")
            .template parameter< global_ptr<const int> >("ell_col")
            .template parameter< global_ptr<const double> >("ell_val")
            .template parameter< global_ptr<const double> >("x")
            .template parameter< global_ptr<double> >("y")
            .close(")").open("{");

        src.new_line() << " size_t global_id   = " << src.global_id(0) << ";";
        src.new_line() << " size_t global_size = " << src.global_size(0) << ";";
 
        src.new_line() << "#define B   " << B;
        src.new_line() << " size_t subwarp_size = B * B;";
        src.new_line() << " size_t subwarp_idx = " << src.local_size(0) << " % subwarp_size;";
        src.new_line() << " size_t subwarp_i = subwarp_idx / B;";
        src.new_line() << " size_t subwarp_j = subwarp_idx % B;";

        src.new_line().smem_static_var("double", "row_A[1024]");
        src.new_line().smem_static_var("double", "row_x[1024/B]");
#ifdef VEXCL_BACKEND_OPENCL
        src.new_line().smem_static_var("double", "*my_A = row_A + subwarp_idx * subwarp_size");
        src.new_line().smem_static_var("double", "*my_x = row_x + subwarp_idx * B");
#else
        src.new_line() << " double *my_A = row_A + subwarp_idx * subwarp_size;";
        src.new_line() << " double *my_x = row_x + subwarp_idx * B;";
#endif
        src.new_line() << " double my_y;";

        src.new_line() << " for (size_t row = global_id / subwarp_size; row < N; row += global_size / subwarp_size)";
        src.open("{");
        src.new_line() << "   my_y = 0;";
        src.new_line() << "   size_t offset = row;";
        src.new_line() << "   for (size_t i = 0; i < ell_width; ++i, offset += ell_pitch) {";
        src.new_line() << "     int c = ell_col[offset];";

        src.new_line() << "     my_A[subwarp_i * B + subwarp_j] = (c >= 0) ? ell_val[subwarp_size * offset + subwarp_i * B + subwarp_j] : 0.0;";
        src.new_line() << "     my_x[subwarp_i]                 = (c >= 0) ? x[B * c + subwarp_i] : 0.0;";

        src.new_line() << "     for (size_t k=0; k<B; ++k)";
        src.new_line() << "       my_y += my_A[subwarp_i * B + k] * my_x[k];";
        src.new_line() << "  }";

        src.new_line() << "  if (subwarp_j == 0) ";
        src.new_line() << "   y[row+subwarp_i] = my_y;";

        src.close("}"); // for
        src.close("}"); // kernel

        K = cache.insert(q, backend::kernel(q, src.str(), "blocked_spmv"));
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

        prof.tic_cl("spmv (block) x100");
        for(int i = 0; i < 100; ++i)
            y = A * x;
        prof.toc("spmv (block) x100");
        prof.toc("vexcl");
    }

    // blocked kernel
    {
        prof.tic_cl("custom kernel");

        vex::sparse::ell<block<B,B>,int> A(ctx, n, n, ptr, col, val);
        vex::vector<block<B,1>> x(ctx, n), y(ctx, n);
        x = math::constant<block<B,1>>(1.0);
        y = math::constant<block<B,1>>(1.0);

        auto &K = blocked_spmv_kernel<B>(ctx.queue(0));
        K(ctx.queue(0), (int)n, (int)A.ell_width, (int)A.ell_pitch,
                A.ell_col, A.ell_val, x(0), y(0));

        prof.tic_cl("spmv (custom) x100");
        for(int i = 0; i < 100; ++i)
          K(ctx.queue(0), (int)n, (int)A.ell_width, (int)A.ell_pitch,
                A.ell_col, A.ell_val, x(0), y(0));

        prof.toc("spmv (custom) x100");
        prof.toc("custom kernel");
    }

    // CPU
    {
        prof.tic_cpu("cpu");

        std::vector<block<B,1>> x(n), y(n);

        for(auto &v : x) v = math::constant<block<B,1>>(1.0);

        prof.tic_cpu("spmv (block) x100");
        for(int k = 0; k < 100; ++k) {
            amgcl::backend::spmv(1.0, boost::tie(n, ptr, col, val), x, 0.0, y);
            /*
            for(int i = 0; i < n; ++i) {
                block<B,1> s = math::zero<block<B,1>>();
                for(int j = ptr[i], e = ptr[i+1]; j < e; ++j)
                    s += val[j] * x[col[j]];
                y[i] = s;
            }
            */
        }
        prof.toc("spmv (block) x100");
        prof.toc("cpu");
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
