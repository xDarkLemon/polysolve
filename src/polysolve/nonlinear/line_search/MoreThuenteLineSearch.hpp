#pragma once

#include "LineSearch.hpp"

#include <cppoptlib/linesearch/morethuente.h>

namespace polysolve::nonlinear::line_search
{
    template <typename ProblemType>
    class MoreThuenteLineSearch : public LineSearch<ProblemType>
    {
    public:
        using Superclass = LineSearch<ProblemType>;
        using typename Superclass::Scalar;
        using typename Superclass::TVector;

        double line_search(
            const TVector &x,
            const TVector &searchDir,
            ProblemType &objFunc) override
        {
            return cppoptlib::MoreThuente<ProblemType, 1>::linesearch(x, searchDir, objFunc);
        }
    };
} // namespace polysolve::nonlinear::line_search
