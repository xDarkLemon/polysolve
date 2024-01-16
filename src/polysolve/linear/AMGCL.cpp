#ifdef POLYSOLVE_WITH_AMGCL

////////////////////////////////////////////////////////////////////////////////
#include "AMGCL.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
////////////////////////////////////////////////////////////////////////////////

std::string indent(int level)
    {
        std::string s;
        for (int i = 0; i<level; i++) s += "  ";
        return s;
    }

void printTree(boost::property_tree::ptree &pt, int level)
{
    if (pt.empty())
    {
        std::cout << "\"" << pt.data() << "\"";
    }

    else
    {
        if (level) std::cout << std::endl;

        std::cout << indent(level) << "{" << std::endl;

        for (boost::property_tree::ptree::iterator pos = pt.begin(); pos != pt.end();)
        {
            std::cout << indent(level + 1) << "\"" << pos->first << "\": ";

            printTree(pos->second, level + 1);
            ++pos;
            if (pos != pt.end())
            {
                std::cout << ",";
            }
            std::cout << std::endl;
        }

        std::cout << indent(level) << " }";
    }
    std::cout << std::endl;
    return;
}

namespace polysolve::linear
{
    namespace
    {
        /* https://stackoverflow.com/questions/15904896/range-based-for-loop-on-a-dynamic-array */
        template <typename T>
        struct WrappedArray
        {
            WrappedArray(const T *first, const T *last)
                : begin_{first}, end_{last} {}
            WrappedArray(const T *first, std::ptrdiff_t size)
                : WrappedArray{first, first + size} {}

            const T *begin() const noexcept { return begin_; }
            const T *end() const noexcept { return end_; }
            const T &operator[](const size_t i) const { return begin_[i]; }

            const T *begin_;
            const T *end_;
        };

        json default_params()
        {
            json params = R"({
                "precond": {
                    "relax": {
                        "degree": 5,
                        "type": "chebyshev",
                        "power_iters": 100,
                        "higher": 1,
                        "lower": 0.03,
                        "scale": true
                    },
                    "class": "amg",
                    "max_levels": 25,
                    "direct_coarse": true,
                    "ncycle": 1,
                    "coarsening": {
                        "type": "smoothed_aggregation",
                        "estimate_spectral_radius": false,
                        "relax": 1.0,
                        "aggr": {
                            "eps_strong": 0.08
                        }
                    }
                },
                "solver": {
                    "tol": 1e-08,
                    "maxiter": 1000,
                    "type": "cg"
                }
        })"_json;

            return params;
        }

        void set_params(const json &params, json &out)
        {
            if (params.contains("AMGCL"))
            {
                // Patch the stored params with input ones
                // if (params["AMGCL"].contains("precond"))
                //     out["precond"].merge_patch(params["AMGCL"]["precond"]);
                // if (params["AMGCL"].contains("solver"))
                //     out["solver"].merge_patch(params["AMGCL"]["solver"]);

                // if (out["precond"]["class"] == "schur_pressure_correction")
                // {
                //     // Initialize the u and p solvers with a tolerance that is comparable to the main solver's
                //     if (!out["precond"].contains("usolver"))
                //     {
                //         out["precond"]["usolver"] = R"({"solver": {"maxiter": 100}})"_json;
                //         out["precond"]["usolver"]["solver"]["tol"] = 10 * out["solver"]["tol"].get<double>();
                //     }
                //     if (!out["precond"].contains("usolver"))
                //     {
                //         out["precond"]["psolver"] = R"({"solver": {"maxiter": 100}})"_json;
                //         out["precond"]["psolver"]["solver"]["tol"] = 10 * out["solver"]["tol"].get<double>();
                //     }
                // }
            }
        }
    } // namespace

    ////////////////////////////////////////////////////////////////////////////////

    AMGCL::AMGCL()
    {
        params_ = default_params();
        // NOTE: usolver and psolver parameters are only used if the
        // preconditioner class is "schur_pressure_correction"
        precond_num_ = 0;
    }

    // Set solver parameters
    void AMGCL::set_parameters(const json &params)
    {
        if (params.contains("AMGCL"))
        {
            // Specially named parameters to match other solvers
            if (params["AMGCL"].contains("block_size"))
            {
                block_size_ = params["AMGCL"]["block_size"];
            }
            if (block_size_ == 2)
            {
                block2_solver_.set_parameters(params);
                return;
            }
            else if (block_size_ == 3)
            {
                block3_solver_.set_parameters(params);
                return;
            }

            set_params(params, params_);
        }
    }

    void AMGCL::get_info(json &params) const
    {
        if (block_size_ == 2)
        {
            block2_solver_.get_info(params);
            return;
        }
        else if (block_size_ == 3)
        {
            block3_solver_.get_info(params);
            return;
        }
        params["num_iterations"] = iterations_;
        params["final_res_norm"] = residual_error_;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void AMGCL::factorize(const StiffnessMatrix &Ain)
    {
        if (block_size_ == 2)
        {
            block2_solver_.factorize(Ain);
            return;
        }
        else if (block_size_ == 3)
        {
            block3_solver_.factorize(Ain);
            return;
        }
        assert(precond_num_ > 0);

        int numRows = Ain.rows();

        WrappedArray<StiffnessMatrix::StorageIndex> ia(Ain.outerIndexPtr(), numRows + 1);
        WrappedArray<StiffnessMatrix::StorageIndex> ja(Ain.innerIndexPtr(), Ain.nonZeros());
        WrappedArray<StiffnessMatrix::Scalar> a(Ain.valuePtr(), Ain.nonZeros());
        if (params_["precond"]["class"] == "schur_pressure_correction")
        {
            std::vector<char> pmask(numRows, 0);
            for (size_t i = precond_num_; i < numRows; ++i)
                pmask[i] = 1;
            params_["precond"]["pmask"] = pmask;
        }

        // AMGCL takes the parameters as a Boost property_tree (i.e., another JSON data structure)
        std::stringstream ss_params;
        ss_params << params_;
        boost::property_tree::ptree pt_params;
        boost::property_tree::read_json(ss_params, pt_params);
        auto A = std::tie(numRows, ia, ja, a);
        solver_ = std::make_unique<Solver>(A, pt_params);
        iterations_ = 0;
        residual_error_ = 0;
    }

    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////

    void AMGCL::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        if (block_size_ == 2)
        {
            block2_solver_.solve(rhs, result);
            return;
        }
        else if (block_size_ == 3)
        {
            block3_solver_.solve(rhs, result);
            return;
        }
        assert(result.size() == rhs.size());
        std::vector<double> _rhs(rhs.data(), rhs.data() + rhs.size());
        std::vector<double> x(result.data(), result.data() + result.size());
        auto rhs_b = Backend::copy_vector(_rhs, backend_params_);
        auto x_b = Backend::copy_vector(x, backend_params_);

        assert(solver_ != nullptr);
        std::tie(iterations_, residual_error_) = (*solver_)(*rhs_b, *x_b);

        std::copy(&(*x_b)[0], &(*x_b)[0] + result.size(), result.data());
    }

    ////////////////////////////////////////////////////////////////////////////////

    AMGCL::~AMGCL()
    {
    }

    template <int BLOCK_SIZE>
    AMGCL_Block<BLOCK_SIZE>::AMGCL_Block()
    {
        params_ = default_params();

        // NOTE: usolver and psolver parameters are only used if the
        // preconditioner class is "schur_pressure_correction"
        precond_num_ = 0;
    }

    template <int BLOCK_SIZE>
    void AMGCL_Block<BLOCK_SIZE>::set_parameters(const json &params)
    {
        set_params(params, params_);
    }

    template <int BLOCK_SIZE>
    void AMGCL_Block<BLOCK_SIZE>::get_info(json &params) const
    {
        params["num_iterations"] = iterations_;
        params["final_res_norm"] = residual_error_;
    }

    ////////////////////////////////////////////////////////////////////////////////

    template <int BLOCK_SIZE>
    void AMGCL_Block<BLOCK_SIZE>::factorize(const StiffnessMatrix &Ain)
    {
        assert(precond_num_ > 0);

        int numRows = Ain.rows();

        WrappedArray<StiffnessMatrix::StorageIndex> ia(Ain.outerIndexPtr(), numRows + 1);
        WrappedArray<StiffnessMatrix::StorageIndex> ja(Ain.innerIndexPtr(), Ain.nonZeros());
        WrappedArray<StiffnessMatrix::Scalar> a(Ain.valuePtr(), Ain.nonZeros());

        if (params_["precond"]["class"] == "schur_pressure_correction")
        {
            std::vector<char> pmask(numRows, 0);
            for (size_t i = precond_num_; i < numRows; ++i)
                pmask[i] = 1;
            params_["precond"]["pmask"] = pmask;
        }

        // AMGCL takes the parameters as a Boost property_tree (i.e., another JSON data structure)
        std::stringstream ss_params;
        ss_params << params_;
        boost::property_tree::ptree pt_params;
        boost::property_tree::read_json(ss_params, pt_params);

        //  Set null space
        if (is_nullspace_)
        {
            std::vector<double> coo;
            reduced_vertices=remove_boundary_vertices(test_vertices,test_boundary_nodes);
            coo.resize(BLOCK_SIZE*reduced_vertices.rows());
            for (size_t i = 0; i < reduced_vertices.rows(); i++)
            {
                for (size_t j = 0; j < BLOCK_SIZE; j++)
                {
                    coo[j+i*BLOCK_SIZE]=reduced_vertices(i,j);
                }
                
            }
            
            int nv;
            nv = amgcl::coarsening::rigid_body_modes(BLOCK_SIZE, coo, null); // TODO: COO requires to change the form of vector<double>
            pt_params.put("precond.coarsening.nullspace.cols", nv);
            pt_params.put("precond.coarsening.nullspace.rows", Ain.rows());
            pt_params.put("precond.coarsening.nullspace.B",    &null[0]);
            pt_params.put("precond.coarsening.aggr.eps_strong", 0.00);
        }

        auto A = std::tie(numRows, ia, ja, a);
        auto Ab = amgcl::adapter::block_matrix<dmat_type>(A);
        solver_ = std::make_unique<Solver>(Ab, pt_params);
        iterations_ = 0;
        residual_error_ = 0;
    }

    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////

    template <int BLOCK_SIZE>
    void AMGCL_Block<BLOCK_SIZE>::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        assert(result.size() == rhs.size());
        std::vector<double> _rhs(rhs.data(), rhs.data() + rhs.size());
        std::vector<double> x(result.data(), result.data() + result.size());

        auto rhs_b = amgcl::backend::reinterpret_as_rhs<dmat_type>(_rhs);
        auto x_b = amgcl::backend::reinterpret_as_rhs<dmat_type>(x);

        assert(solver_ != nullptr);
        std::tie(iterations_, residual_error_) = (*solver_)(rhs_b, x_b);
        for (size_t i = 0; i < rhs.size() / BLOCK_SIZE; i++)
            for (size_t j = 0; j < BLOCK_SIZE; j++)
            {
                result[BLOCK_SIZE * i + j] = x_b[i](j);
            }
    }

    template <int BLOCK_SIZE>
    AMGCL_Block<BLOCK_SIZE>::~AMGCL_Block()
    {
    }

    template class AMGCL_Block<2>;
    template class AMGCL_Block<3>;
} // namespace polysolve::linear

#endif
