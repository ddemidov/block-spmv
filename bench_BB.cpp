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
        src.new_line() << " size_t subwarp_idx = " << src.local_id(0) << " % subwarp_size;";
        src.new_line() << " size_t subwarp_gid = " << src.local_id(0) << " / subwarp_size;";
        src.new_line() << " size_t subwarp_i = subwarp_idx / B;";
        src.new_line() << " size_t subwarp_j = subwarp_idx % B;";

        src.new_line().smem_static_var("double", "row_A[256]");
#ifdef VEXCL_BACKEND_OPENCL
        src.new_line().smem_static_var("double", "*my_A = row_A + subwarp_gid * subwarp_size + subwarp_i * B");
#else
        src.new_line() << " double *my_A = row_A + subwarp_gid * subwarp_size + subwarp_i * B;";
#endif
        src.new_line() << " double my_y;";

        src.new_line() << " size_t loop_iters = (N-1) / (global_size / subwarp_size) + 1;";

        src.new_line() << " for (size_t iter = 0; iter < loop_iters; ++iter)";
        src.open("{");
        src.new_line() << "   size_t row = (global_id + iter * global_size) / subwarp_size;";
        src.new_line() << "   my_y = 0;";
        src.new_line() << "   size_t offset = min((int)row, (int)N-1);";
        src.new_line() << "   for (size_t i = 0; i < ell_width; ++i, offset += ell_pitch) {";
        src.new_line() << "     int c = ell_col[offset];";

        src.new_line() << "     my_A[subwarp_j] = (c >= 0) ? ell_val[subwarp_size * offset + subwarp_i * B + subwarp_j] * x[B * c + subwarp_i] : 0.0;";
        src.new_line().barrier();
        for (std::size_t stride = B/2; stride > 0; stride /= 2)
	  src.new_line() << "       my_A[subwarp_j] += my_A[subwarp_j ^ " << stride << "];";

        src.new_line() << "     my_y += my_A[subwarp_j];"; // only subwarp_j == 0 relevant
        src.new_line() << "   }";

        src.new_line() << "   if (subwarp_j == 0 && row < N) ";
        src.new_line() << "     y[B*row+subwarp_i] = my_y;";

        src.close("}"); // for
        src.close("}"); // kernel

        K = cache.insert(q, backend::kernel(q, src.str(), "blocked_spmv"));
    }

    return K->second;
}

//---------------------------------------------------------------------------
template <int B>
vex::backend::kernel& blocked_spmv_kernel2(vex::backend::command_queue &q) {
    using namespace vex;
    using namespace vex::detail;
    static kernel_cache cache;

    auto K = cache.find(q);
    if (K == cache.end()) {
        backend::source_generator src(q);

         /* The following kernel uses B threads for each B*B block -> more continguous memory reads for A
         * Performances are from a Tesla C2070.
         */
       src.kernel("blocked_spmv2").open("(")
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
 
        src.new_line() << " #define subwarp_size " << B;
        src.new_line() << " const size_t subwarp_gid = " << src.local_id(0) << " / subwarp_size;";
        src.new_line() << " const size_t subwarp_idx = " << src.local_id(0) << " % subwarp_size;";

        src.new_line().smem_static_var("double", "row_A[256*subwarp_size]");
#ifdef VEXCL_BACKEND_OPENCL
        src.new_line().smem_static_var("double", "*my_A = row_A + subwarp_gid * subwarp_size * subwarp_size");
#else
        src.new_line() << " double *my_A = row_A + subwarp_gid * subwarp_size * subwarp_size;";
#endif
        src.new_line() << " double my_x, my_y;";

        src.new_line() << " size_t loop_iters = (N-1) / (global_size / subwarp_size) + 1;";

        src.new_line() << " for (size_t iter = 0; iter < loop_iters; ++iter)";
        src.open("{");
        src.new_line() << "   size_t row = (global_id + iter * global_size) / subwarp_size;";
        src.new_line() << "   my_y = 0;";
        src.new_line() << "   size_t offset = min((int)row, (int)N-1);";
        src.new_line() << "   for (size_t i = 0; i < ell_width; ++i, offset += ell_pitch) {";
        src.new_line() << "     int c = ell_col[offset];";

        src.new_line() << "     size_t ell_val_offset = subwarp_size * subwarp_size * offset + subwarp_idx;";
        src.new_line() << "     my_x = (c >= 0) ? x[subwarp_size * c + subwarp_idx] : 0.0;";
        src.new_line() << "     for (size_t k=0; k<subwarp_size; ++k) ";
        src.new_line() << "       my_A[k * subwarp_size + subwarp_idx] = (c >= 0) ? ell_val[ell_val_offset + k * subwarp_size] * my_x : 0.0;";
        src.new_line().barrier();

        src.new_line() << "     for (size_t k=0; k<subwarp_size; ++k)";
        src.new_line() << "       my_y += my_A[subwarp_idx * subwarp_size + k];";
        src.new_line() << "   }";

        src.new_line() << "   if (row < N)";
        src.new_line() << "     y[subwarp_size*row+subwarp_idx] = my_y;";

        src.close("}"); // for
        src.close("}"); // kernel

        K = cache.insert(q, backend::kernel(q, src.str(), "blocked_spmv2"));
    }

    return K->second;
}

//---------------------------------------------------------------------------
template <int B>
vex::backend::kernel& blocked_spmv_kernel3(vex::backend::command_queue &q) {
    using namespace vex;
    using namespace vex::detail;
    static kernel_cache cache;

    auto K = cache.find(q);
    if (K == cache.end()) {
        backend::source_generator src(q);

         /* The following kernel uses B threads for each B*B block -> more continguous memory reads for A
         * Performances are from a Tesla C2070.
         */
       src.kernel("blocked_spmv3").open("(")
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
 
        src.new_line() << " #define subwarp_size " << B;
        src.new_line() << " const size_t subwarp_gid = " << src.local_id(0) << " / subwarp_size;";
        src.new_line() << " const size_t subwarp_idx = " << src.local_id(0) << " % subwarp_size;";

        src.new_line().smem_static_var("double", "row_A[256*subwarp_size]");
#ifdef VEXCL_BACKEND_OPENCL
        src.new_line().smem_static_var("double", "*my_A = row_A + subwarp_gid * subwarp_size * subwarp_size");
#else
        src.new_line() << " double *my_A = row_A + subwarp_gid * subwarp_size * subwarp_size;";
#endif
        src.new_line() << " double my_x, my_y;";

        src.new_line() << " size_t loop_iters = (N-1) / global_size + 1;";

        src.new_line() << " for (size_t iter = 0; iter < loop_iters; ++iter)";
        src.open("{");
        src.new_line() << "  for (size_t j=0; j<subwarp_size; ++j) { ";
        src.new_line() << "   size_t row = (global_id + j * global_size) / subwarp_size + iter * global_size;";
        src.new_line() << "   my_y = 0;";
        src.new_line() << "   size_t offset = min((int)row, (int)N-1);";
        src.new_line() << "   for (size_t i = 0; i < ell_width; ++i, offset += ell_pitch) {";
        src.new_line() << "     int c = ell_col[offset];";

        src.new_line() << "     size_t ell_val_offset = subwarp_size * subwarp_size * offset + subwarp_idx;";
        src.new_line() << "     my_x = (c >= 0) ? x[subwarp_size * c + subwarp_idx] : 0.0;";
        src.new_line() << "     for (size_t k=0; k<subwarp_size; ++k) ";
        src.new_line() << "       my_A[k * subwarp_size + subwarp_idx] = (c >= 0) ? ell_val[ell_val_offset + k * subwarp_size] * my_x : 0.0;";
        src.new_line().barrier();

        src.new_line() << "     for (size_t k=0; k<subwarp_size; ++k)";
        src.new_line() << "       my_y += my_A[subwarp_idx * subwarp_size + k];";
        src.new_line() << "   }";

        src.new_line() << "   if (row < N)";
        src.new_line() << "     y[subwarp_size*row+subwarp_idx] = my_y;";
        src.new_line() << "  }";


        src.close("}"); // for
        src.close("}"); // kernel

        K = cache.insert(q, backend::kernel(q, src.str(), "blocked_spmv3"));
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

    std::vector<block<B,1>> r(n);
    std::generate_n(reinterpret_cast<double*>(r.data()), n * B, drand48);
    /*for (size_t i=0; i<n; ++i)
      for (size_t k=0; k<B; ++k)
        r[i](k) = 1.0;*/

    vex::vector<block<B,1>> x(ctx, r);

    // CPU
    std::vector<block<B,1>> Y(n);
    {
        prof.tic_cpu("cpu");

        prof.tic_cpu("spmv (block) x100");
        for(int k = 0; k < 100; ++k)
            amgcl::backend::spmv(1.0, boost::tie(n, ptr, col, val), r, 0.0, Y);
        prof.toc("spmv (block) x100");
        prof.toc("cpu");
    }


    // VexCL
    {
        prof.tic_cl("vexcl");
        prof.tic_cl("transfer");
        vex::sparse::ell<block<B,B>,int> A(ctx, n, n, ptr, col, val);
        prof.toc("transfer");

        vex::vector<block<B,1>> y(ctx, n);

        x = math::constant<block<B,1>>(1.0);
        y = A * x;

        prof.tic_cl("spmv (block) x100");
        for(int i = 0; i < 100; ++i)
            y = A * x;
        prof.toc("spmv (block) x100");

        prof.tic_cl("checking");
        auto v = y.map(0);
        double delta = 0;
        for(int i = 0; i < n; ++i)
            for(int k = 0; k < B; ++k)
                delta += std::abs(Y[i](k) - v[i](k));
        std::cout << "delta = " << delta << std::endl;
        prof.toc("checking");

        prof.toc("vexcl");
    }

    // blocked kernel, BxB
    {
        prof.tic_cl("custom BxB kernel");

        vex::sparse::ell<block<B,B>,int> A(ctx, n, n, ptr, col, val);
        vex::vector<block<B,1>> y(ctx, n);
        y = math::constant<block<B,1>>(1.0);

        auto &K = blocked_spmv_kernel<B>(ctx.queue(0));
        K.config(256, 256);
        K(ctx.queue(0), (int)n, (int)A.ell_width, (int)A.ell_pitch,
                A.ell_col, A.ell_val, x(0), y(0));

        prof.tic_cl("spmv (BxB) x100");
        for(int i = 0; i < 100; ++i)
          K(ctx.queue(0), (int)n, (int)A.ell_width, (int)A.ell_pitch,
                A.ell_col, A.ell_val, x(0), y(0));

        prof.toc("spmv (BxB) x100");

        prof.tic_cl("checking");
        auto v = y.map(0);
        double delta = 0;
        for(int i = 0; i < n; ++i)
            for(int k = 0; k < B; ++k)
                delta += std::abs(Y[i](k) - v[i](k));
        std::cout << "delta = " << delta << std::endl;
        prof.toc("checking");

        prof.toc("custom BxB kernel");
    }

    // blocked kernel, B threads per BxB block
    {
        prof.tic_cl("custom B kernel");

        vex::sparse::ell<block<B,B>,int> A(ctx, n, n, ptr, col, val);
        vex::vector<block<B,1>> y(ctx, n);
        y = math::constant<block<B,1>>(1.0);

        auto &K = blocked_spmv_kernel2<B>(ctx.queue(0));
        K.config(256, 256);
        K(ctx.queue(0), (int)n, (int)A.ell_width, (int)A.ell_pitch,
                A.ell_col, A.ell_val, x(0), y(0));

        prof.tic_cl("spmv (B) x100");
        for(int i = 0; i < 100; ++i)
          K(ctx.queue(0), (int)n, (int)A.ell_width, (int)A.ell_pitch,
                A.ell_col, A.ell_val, x(0), y(0));

        prof.toc("spmv (B) x100");

        prof.tic_cl("checking");
        auto v = y.map(0);
        double delta = 0;
        for(int i = 0; i < n; ++i)
            for(int k = 0; k < B; ++k)
                delta += std::abs(Y[i](k) - v[i](k));
        std::cout << "delta = " << delta << std::endl;
        prof.toc("checking");

        prof.toc("custom B kernel");
    }

    // blocked kernel, B threads per BxB block, one thread per row overall
    {
        prof.tic_cl("custom 1 kernel");

        vex::sparse::ell<block<B,B>,int> A(ctx, n, n, ptr, col, val);
        vex::vector<block<B,1>> y(ctx, n);
        y = math::constant<block<B,1>>(1.0);

        auto &K = blocked_spmv_kernel3<B>(ctx.queue(0));
        K.config(256, 256);
        K(ctx.queue(0), (int)n, (int)A.ell_width, (int)A.ell_pitch,
                A.ell_col, A.ell_val, x(0), y(0));

        prof.tic_cl("spmv (1) x100");
        for(int i = 0; i < 100; ++i)
          K(ctx.queue(0), (int)n, (int)A.ell_width, (int)A.ell_pitch,
                A.ell_col, A.ell_val, x(0), y(0));

        prof.toc("spmv (1) x100");

        prof.tic_cl("checking");
        auto v = y.map(0);
        double delta = 0;
        for(int i = 0; i < n; ++i)
            for(int k = 0; k < B; ++k)
                delta += std::abs(Y[i](k) - v[i](k));
        std::cout << "delta = " << delta << std::endl;
        prof.toc("checking");

        prof.toc("custom 1 kernel");
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
            //run_benchmark<1>(vm["size"].as<int>());
            break;
	case 2:
            run_benchmark<2>(vm["size"].as<int>());
            break;
	case 3:
            //run_benchmark<3>(vm["size"].as<int>());
            break;
        case 4:
            run_benchmark<4>(vm["size"].as<int>());
            break;
	case 5:
            //run_benchmark<5>(vm["size"].as<int>());
            break;
	case 8:
            run_benchmark<8>(vm["size"].as<int>());
            break;
 	case 16:
            run_benchmark<16>(vm["size"].as<int>());
            break;
        default:
            std::cerr << "Unsupported block size" << std::endl;
            return 1;
    }
}
