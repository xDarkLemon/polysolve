#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"

#include <Eigen/Core>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/gmres.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/profiler.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/relaxation/chebyshev.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/value_type/static_matrix.hpp>
// #include <amgcl/backend/vexcl.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/reorder.hpp>
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/profiler.hpp>

#include <amgcl/coarsening/rigid_body_modes.hpp>
#include <amgcl/coarsening/as_scalar.hpp>

#include <memory>
#include <type_traits>

////////////////////////////////////////////////////////////////////////////////
//
// WARNING:
// The matrix is assumed to be in row-major format, since AMGCL assumes that the
// outer index is for row. If the matrix is symmetric, you are fine, because CSR
// and CSC are the same. If the matrix is not symmetric and you pass in a
// column-major matrix, the solver will actually solve A^T x = b.
//

namespace polysolve::linear
{

    template <int BLOCK_SIZE>
    class AMGCL_Block : public Solver
    {

    public:
        AMGCL_Block();
        ~AMGCL_Block();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(AMGCL_Block)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void set_parameters(const json &params) override;

        // Retrieve information
        virtual void get_info(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyze_pattern(const StiffnessMatrix &A, const int precond_num) override { precond_num_ = precond_num; }

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "AMGCL_Block" + std::to_string(BLOCK_SIZE); }

    private:
        typedef amgcl::static_matrix<double, BLOCK_SIZE, BLOCK_SIZE> dmat_type; // matrix value type in double precision
        using Backend = amgcl::backend::builtin<dmat_type>;
        using Solver = amgcl::make_solver<
            amgcl::runtime::preconditioner<Backend>,
            amgcl::runtime::solver::wrapper<Backend>>;
        std::unique_ptr<Solver> solver_;
        json params_;
        typename Backend::params backend_params_;
        bool is_nullspace_=true;
        int precond_num_;
        std::vector<double> null;
        //Timer, test only
        amgcl::profiler<> prof;
        Eigen::MatrixXd reduced_vertices;

        // Output info
        size_t iterations_;
        double residual_error_;
    };

    class AMGCL : public Solver
    {

    public:
        AMGCL();
        ~AMGCL();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(AMGCL)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void set_parameters(const json &params) override;

        // Retrieve information
        virtual void get_info(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyze_pattern(const StiffnessMatrix &A, const int precond_num) override
        {
            if (block_size_ == 2)
            {
                block2_solver_.analyze_pattern(A, precond_num);
                return;
            }
            else if (block_size_ == 3)
            {
                block3_solver_.analyze_pattern(A, precond_num);
                return;
            }
            precond_num_ = precond_num;
        }

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "AMGCL"; }
        void factorize(const StiffnessMatrix &A, const std::vector<double> &coo);

    private:
        using Backend = amgcl::backend::builtin<double>;
        using Solver = amgcl::make_solver<
            amgcl::runtime::preconditioner<Backend>,
            amgcl::runtime::solver::wrapper<Backend>>;
        std::unique_ptr<Solver> solver_;
        json params_;
        typename Backend::params backend_params_;
        int precond_num_;
        int block_size_ = 1;

        // Output info
        size_t iterations_;
        double residual_error_;

        AMGCL_Block<2> block2_solver_;
        AMGCL_Block<3> block3_solver_;
    };

} // namespace polysolve::linear
