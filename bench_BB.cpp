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

        std::stringstream ss;
        ss << "#define B    " << B << std::endl;   // NOTE: This is emulating CUDA templates without the problems of external linkage

        /* The following kernel uses B*B threads for each B*B block -> continguous memory reads for A
         *  - Performance for B=1 is similar (within 10 percent) to 'normal' kernel
         *  - Performance for B=2 is about 2x slower than the 'naive' kernel
         *  - Performance for B=4 is 10x faster than the 'naive' kernel, because it avoids strided memory access
         * Performances are from a Tesla C2070.
         */
        vex::backend::kernel blocked_spmv(ctx.queue(0), ss.str() + VEX_STRINGIZE_SOURCE(
             extern "C"
             __global__ void blocked_spmv(unsigned long N, \n
                                          const int *ell_col, const double *ell_val, unsigned long ell_width, unsigned long ell_pitch, \n
                                          const int *csr_ptr, const int *csr_col, const double *csr_val,\n
                                          const double *x, double *y)\n
             {\n
               size_t global_id   = blockDim.x * blockIdx.x + threadIdx.x;\n
               size_t global_size = gridDim.x * blockDim.x;\n
               
               size_t subwarp_size = B * B;\n
               size_t subwarp_idx = blockDim.x % subwarp_size;\n
               size_t subwarp_i = subwarp_idx / B;\n
               size_t subwarp_j = subwarp_idx % B;\n

               // shared memory, one thread per block element. Assuming a maximum of 1024 threads.
               // TODO: use actual thread block size
               __shared__ double row_A[1024]; double *my_A = row_A + subwarp_idx * subwarp_size;\n
               __shared__ double row_x[1024/B]; double *my_x = row_x + subwarp_idx * B;\n
               double my_y;\n

               for (size_t row = global_id / subwarp_size; row < N; row += global_size / subwarp_size)\n
               {\n
                 my_y = 0;\n

                 // ELL part:\n
                 size_t offset = row;\n
                 for (size_t i = 0; i < ell_width; ++i, offset += ell_pitch) {\n
                   int c = ell_col[offset];\n
                   
                   // load data to shared memory
                   my_A[subwarp_i * B + subwarp_j] = (c >= 0) ? ell_val[subwarp_size * offset + subwarp_i * B + subwarp_j] : 0.0;\n
                   my_x[subwarp_i]                 = (c >= 0) ? x[B * c + subwarp_i] : 0.0;\n

                   // compute: (all threads participate. Maybe reduce to just one thread per row?)
                   for (size_t k=0; k<B; ++k)\n
                     my_y += my_A[subwarp_i * B + k] * my_x[k];\n
                   
                 }\n

                 // CSR part: TODO  \n
                 //if (csr_ptr) {\n
                 //  size_t col_end = csr_ptr[row+1]; \n
                 //  for (size_t j=csr_ptr[row]; j<col_end; ++j)\n
                 //    sum += x[csr_col[j]] * csr_val[j];\n
                 //}

                 // write result:
                 if (subwarp_j == 0)
                   y[row+subwarp_i] = my_y;\n
               }\n
             }\n        
          ),"blocked_spmv", 0);

        vex::vector<block<B,1>> x(ctx, n), y(ctx, n);
        x = math::constant<block<B,1>>(1.0);
        y = math::constant<block<B,1>>(1.0);

        blocked_spmv(ctx.queue(0), static_cast<cl_ulong>(n),
                     A.ell_col, A.ell_val, static_cast<cl_ulong>(A.ell_width), static_cast<cl_ulong>(A.ell_pitch),
                     A.csr_ptr, A.csr_col, A.csr_val,
                     x(0), y(0));

        prof.tic_cl("spmv (custom) x100");
        for(int i = 0; i < 100; ++i)
          blocked_spmv(ctx.queue(0), static_cast<cl_ulong>(n),
                       A.ell_col, A.ell_val, static_cast<cl_ulong>(A.ell_width), static_cast<cl_ulong>(A.ell_pitch),
                       A.csr_ptr, A.csr_col, A.csr_val,
                       x(0), y(0));
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
