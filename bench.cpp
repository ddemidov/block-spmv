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
vex::backend::kernel& coop_kernel(vex::backend::command_queue &q) {
    using namespace vex;
    using namespace vex::detail;
    static kernel_cache cache;

    auto K = cache.find(q);
    if (K == cache.end()) {
        backend::source_generator src(q);

        src.function<void>("load_blocks").open("(")
            .template parameter<int>("nblocks")
            .template parameter< shared_ptr<double> >("dst")
            .template parameter< global_ptr<const double> >("src")
            .template parameter< regstr_ptr<double> >("block")
            .close(")")
            .open("{");
        src.new_line() << "int n = nblocks * " << B * B << ";";
        src.new_line() << "int lid = " << src.local_id(0) << ";";
        src.new_line() << "int lsz = " << src.local_size(0) << ";";
        src.new_line() << "int beg = lid * " << B * B << ";";
        src.new_line() << "for(int j = lid; j < n; j += lsz)";
        src.new_line() << "  dst[j] = src[j];";
        src.new_line().barrier();
        src.new_line() << "for(int j = 0; j < " << B * B << "; ++j)";
        src.new_line() << "  block[j] = dst[beg + j];";
        src.new_line().barrier();
        src.close("}");

        src.kernel("spmv").open("(")
            .template parameter<int>("n")
            .template parameter<int>("ell_width")
            .template parameter<int>("ell_pitch")
            .template parameter< global_ptr<const int> >("ell_col")
            .template parameter< global_ptr<const block<B,B> > >("ell_val")
            .template parameter< global_ptr<const block<B,1> > >("X")
            .template parameter< global_ptr<block<B,1> > >("Y")
            .template smem_parameter<double>("S")
            .close(")").open("{");

        src.smem_declaration<double>("S");

        src.new_line() << "int block_start = " << src.group_id(0) << " * " << src.local_size(0) << ";";
        src.new_line() << "int lid = " << src.local_id(0) << ";";
        src.new_line() << "int lsz = " << src.local_size(0) << ";";
        src.new_line() << "int gsz = " << src.global_size(0) << ";";
        src.new_line() << "for(int idx = " << src.global_id(0) << ", pos = 0; pos < n; idx += gsz, pos += gsz, block_start += gsz)";
        src.open("{");
        src.new_line() << type_name<block<B,1>>() << " sum = " << block<B,1>() << ";";
        src.new_line() << type_name<block<B,B>>() << " A;";
        src.new_line() << "int blocks_to_load = min(block_start + lsz, n) - block_start;";
        src.new_line() << "for(int j = 0; j < ell_width; ++j) {";
        src.new_line() << "  load_blocks(blocks_to_load, S, reinterpret_cast<" << type_name< global_ptr<const double> >() << ">(ell_val + j * ell_pitch + block_start), reinterpret_cast<double*>(&A));";
        src.new_line() << "  if (idx >= n) continue;";
        src.new_line() << "  int c = ell_col[j * ell_pitch + idx];";
        src.new_line() << "  if (c < 0) continue;";
        src.new_line() << "  sum += A * X[c];";
        src.new_line() << "}";
        src.new_line() << "if (idx < n) Y[idx] = sum;";
        src.close("}");
        src.close("}");

        K = cache.insert(q, backend::kernel(q, src.str(), "spmv", B*B*sizeof(double)));
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

    // CPU
    std::vector<block<B,1>> Y(n);
    if (1) {
        prof.tic_cpu("cpu");

        std::vector<block<B,1>> x(n);

        for(auto &v : x) v = math::constant<block<B,1>>(1.0);

        prof.tic_cpu("spmv x100");
        for(int i = 0; i < 100; ++i)
            amgcl::backend::spmv(1.0, boost::tie(n, ptr, col, val), x, 0.0, Y);
        prof.toc("spmv x100");
        prof.toc("cpu");
    }

    // VexCL
    if (1){
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
        double tm = prof.toc("spmv x100");

        prof.tic_cl("checking");
        auto v = y.map(0);
        double delta = 0;
        for(int i = 0; i < n; ++i)
            for(int k = 0; k < B; ++k)
                delta += std::abs(Y[i](k) - v[i](k));
        prof.toc("checking");
        prof.toc("vexcl");
        std::cout << "vexcl: " << tm << ", delta = " << delta << std::endl;
    }

    // Block-optimized 1
    if (1){
        prof.tic_cl("reorder");
        prof.tic_cl("transfer");
        vex::sparse::ell<block<B,B>,int> A(ctx, n, n, ptr, col, val);
        prof.toc("transfer");

        prof.tic_cl("convert");
        vex::vector<double> ell_val(ctx, A.ell_pitch * A.ell_width * B * B);
        typedef block<B,B> block_BB;
        VEX_FUNCTION(void, chidx, (int, i)(int, n)(const block_BB*, v_in)(double*, v_out)(int, B),
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

        vex::vector<block<B,1>> x(ctx, n);
        vex::vector<double> y(ctx, n * B);
        x = math::constant<block<B,1>>(1.0);

        auto &K = spmv_kernel<B>(ctx.queue(0));
        K(ctx.queue(0), (int)n, (int)A.ell_width, (int)A.ell_pitch,
                A.ell_col, ell_val(0), x(0), y(0));

        prof.tic_cl("spmv x100");
        for(int i = 0; i < 100; ++i)
            K(ctx.queue(0), (int)n, (int)A.ell_width, (int)A.ell_pitch,
                    A.ell_col, ell_val(0), x(0), y(0));
        double tm = prof.toc("spmv x100");
        prof.tic_cl("checking");
        auto v = y.map(0);
        double delta = 0;
        for(int i = 0; i < n; ++i)
            for(int k = 0; k < B; ++k)
                delta += std::abs(Y[i](k) - v[k*n+i]);
        prof.toc("checking");
        prof.toc("reorder");
        std::cout << "reorder: " << tm << ", delta = " << delta << std::endl;

    }

    // Cooperative load
    {
        prof.tic_cl("cooperative");
        prof.tic_cl("transfer");
        vex::sparse::ell<block<B,B>,int> A(ctx, n, n, ptr, col, val);
        prof.toc("transfer");

        vex::vector<block<B,1>> x(ctx, n), y(ctx, n);

        x = math::constant<block<B,1>>(1.0);

        auto &K = coop_kernel<B>(ctx.queue(0));
        K.push_arg((int)n);
        K.push_arg((int)A.ell_width);
        K.push_arg((int)A.ell_pitch);
        K.push_arg(A.ell_col);
        K.push_arg(A.ell_val);
        K.push_arg(x(0));
        K.push_arg(y(0));
        K.set_smem(B*B*sizeof(double));
        K(ctx.queue(0));

        prof.tic_cl("spmv x100");
        for(int i = 0; i < 100; ++i)
            K(ctx.queue(0));
        double tm = prof.toc("spmv x100");

        prof.tic_cl("checking");
        auto v = y.map(0);
        double delta = 0;
        for(int i = 0; i < n; ++i)
            for(int k = 0; k < B; ++k)
                delta += std::abs(Y[i](k) - v[i](k));
        prof.toc("checking");
        prof.toc("cooperative");
        std::cout << "cooperative:  " << tm << ", delta = " << delta << std::endl;
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
        /*
        case 1:
            run_benchmark<1>(vm["size"].as<int>());
            break;
        */
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
