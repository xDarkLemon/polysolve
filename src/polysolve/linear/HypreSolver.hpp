#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

#include <HYPRE_utilities.h>
#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>

////////////////////////////////////////////////////////////////////////////////
//
// https://computation.llnl.gov/sites/default/files/public/hypre-2.11.2_usr_manual.pdf
// https://github.com/LLNL/hypre/blob/v2.14.0/docs/HYPRE_usr_manual.pdf
//

namespace polysolve::linear
{

    class HypreSolver : public Solver
    {

    public:
        HypreSolver();
        ~HypreSolver();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(HypreSolver)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void set_parameters(const json &params) override;

        // Retrieve memory information from Pardiso
        virtual void get_info(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyze_pattern(const StiffnessMatrix &A, const int precond_num) override { precond_num_ = precond_num; }

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "Hypre"; }

    protected:
        int dimension_ = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)
        int max_iter_ = 1000;
        int pre_max_iter_ = 1;
        double conv_tol_ = 1e-10;
        bool is_nullspace_=false; //  Cannot work since the trilinos nullspace vector does not work
        bool is_test=true;
        HYPRE_Int num_iterations;
        HYPRE_Complex final_res_norm;

    private:
        bool has_matrix_ = false;
        int precond_num_;

        HYPRE_IJMatrix A;
        HYPRE_ParCSRMatrix parcsr_A;
    };

} // namespace polysolve::linear
