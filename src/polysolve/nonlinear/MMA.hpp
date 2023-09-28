#pragma once

#include "MMAAux.hpp"
#include "BoxConstraints.hpp"
#include <polyfem/solver/forms/adjoint_forms/AdjointForm.hpp>

namespace polysolve::nonlinear
{
    template <typename ProblemType>
    class MMA : public BoxConstraints<ProblemType>
    {
    public:
        using Superclass = BoxConstraints<ProblemType>;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        MMA(const polyfem::json &solver_params, const double dt, const double characteristic_length)
            : Superclass(solver_params, dt, characteristic_length)
        {
            this->m_line_search = NULL;
        }

        void set_constraints(const std::vector<std::shared_ptr<polyfem::solver::AdjointForm>> &constraints) { constraints_ = constraints; }

        std::string name() const override { return "MMA"; }

    protected:
        virtual int default_descent_strategy() override { return 1; }

        using Superclass::descent_strategy_name;
        std::string descent_strategy_name(int descent_strategy) const override
        {
            switch (descent_strategy)
            {
            case 1:
                return "MMA";
            default:
                throw std::invalid_argument("invalid descent strategy");
            }
        }

        void increase_descent_strategy() override
        {
            assert(this->descent_strategy <= 1);
            this->descent_strategy++;
        }

    protected:
        std::shared_ptr<MMAAux> mma;

        std::vector<std::shared_ptr<polyfem::solver::AdjointForm>> constraints_;

        void reset(const int ndof) override;

        virtual bool compute_update_direction(
            ProblemType &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) override;
    };

} // namespace polysolve::nonlinear

#include "MMA.tpp"