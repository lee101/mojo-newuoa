#include <algorithm>
#include <cstdint>
#include <vector>

#include "upstream_newuoa.h"

using Callback = double (*)(std::int64_t, std::int64_t, std::int64_t);

struct Objective {
    Callback callback;
    std::int64_t context;
    std::int64_t calls = 0;

    double operator()(int n, double* x) {
        ++calls;
        return callback(n, reinterpret_cast<std::int64_t>(x), context);
    }
};

extern "C" double reference_min_newuoa(
    std::int64_t n,
    std::int64_t x_address,
    double rhobeg,
    double rhoend,
    std::int64_t maxfun,
    std::int64_t callback_address,
    std::int64_t context,
    std::int64_t* calls
) {
    Objective objective{
        reinterpret_cast<Callback>(callback_address),
        context,
    };
    auto* x = reinterpret_cast<double*>(x_address);
    double result = min_newuoa(
        static_cast<int>(n),
        x,
        objective,
        rhobeg,
        rhoend,
        static_cast<int>(maxfun)
    );
    *calls = objective.calls;
    return result;
}

extern "C" double reference_newuoa_npt(
    std::int64_t n,
    std::int64_t npt,
    std::int64_t x_address,
    double rhobeg,
    double rhoend,
    std::int64_t maxfun,
    std::int64_t callback_address,
    std::int64_t context,
    std::int64_t* calls
) {
    Objective objective{
        reinterpret_cast<Callback>(callback_address),
        context,
    };
    auto* x = reinterpret_cast<double*>(x_address);
    std::vector<double> workspace(
        (npt + 13) * (npt + n) + 3 * n * (n + 3) / 2 + 11,
        0.0
    );
    int returned_calls = 0;
    double result = newuoa_(
        static_cast<int>(n),
        static_cast<int>(npt),
        x,
        rhobeg,
        rhoend,
        &returned_calls,
        static_cast<int>(maxfun),
        workspace.data(),
        objective
    );
    *calls = returned_calls;
    return result;
}
