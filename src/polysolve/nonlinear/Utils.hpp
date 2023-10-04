#pragma once

#include <polysolve/Types.hpp>

#include "Problem.hpp"

#include <spdlog/spdlog.h>

namespace polysolve::nonlinear

#define POLYSOLVE_SCOPED_STOPWATCH(...) polysolve::nonlinear::StopWatch __polysolve_stopwatch(__VA_ARGS__)
{

    struct Timing
    {
        operator double() const { return time; }

        void operator+=(const double t)
        {
            time += t;
            ++count;
        }

        double time = 0;
        size_t count = 0;
    };

    class StopWatch
    {
    private:
        using clock = std::chrono::steady_clock;

    public:
        StopWatch(spdlog::logger &logger);
        StopWatch(const std::string &name, spdlog::logger &logger);
        StopWatch(double &total_time, spdlog::logger &logger);
        StopWatch(Timing &timing, spdlog::logger &logger);
        StopWatch(const std::string &name, double &total_time, spdlog::logger &logger);
        StopWatch(const std::string &name, Timing &timing, spdlog::logger &logger);

        virtual ~StopWatch();

        void start();
        void stop();

        double getElapsedTimeInSec();
        void log_msg();

    private:
        std::string m_name;
        std::chrono::time_point<clock> m_start, m_stop;
        double *m_total_time = nullptr;
        size_t *m_count = nullptr;
        bool is_running = false;

        spdlog::logger &m_logger;
    };

    [[noreturn]] void log_and_throw_error(spdlog::logger &logger, const std::string &msg);

    template <typename... Args>
    [[noreturn]] void log_and_throw_error(spdlog::logger &logger, const std::string &msg, const Args &...args)
    {
        log_and_throw_error(logger, fmt::format(msg, args...));
    }

    static void save_sampled_values(const std::string &filename,
                                    const typename Problem::TVector &x,
                                    const typename Problem::TVector &grad,
                                    Problem &objFunc,
                                    spdlog::logger &logger,
                                    const double starting_step_size = 1e-1,
                                    const int num_samples = 1000);

    Eigen::SparseMatrix<double> sparse_identity(int rows, int cols);
    bool has_hessian_nans(const polysolve::StiffnessMatrix &hessian);

} // namespace polysolve::nonlinear