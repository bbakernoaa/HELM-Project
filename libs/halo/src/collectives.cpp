#include "halo/collectives.hpp"

#include <cstddef>
#include <sstream>

namespace halo {
namespace detail {

std::vector<int> prefix_sum(const std::vector<int> &counts) {
    // displ has the same length as counts; displ[0] = 0 and
    // displ[k] = displ[k-1] + counts[k-1] (Req 2.1).
    std::vector<int> displ(counts.size());
    int running = 0;
    for (std::size_t k = 0; k < counts.size(); ++k) {
        displ[k] = running;
        running += counts[k];
    }
    return displ;
}

void validate_counts(const std::vector<int> &counts, int comm_size) {
    // Wrong length: name the expected length (comm_size) (Req 2.5).
    if (static_cast<int>(counts.size()) != comm_size) {
        std::ostringstream oss;
        oss << "halo::detail::validate_counts: counts array has length " << counts.size()
            << " but expected length " << comm_size << " (the communicator size)";
        throw std::invalid_argument(oss.str());
    }

    // Any negative entry: name the offending rank index (Req 2.4).
    for (std::size_t k = 0; k < counts.size(); ++k) {
        if (counts[k] < 0) {
            std::ostringstream oss;
            oss << "halo::detail::validate_counts: negative count " << counts[k]
                << " at rank index " << k;
            throw std::invalid_argument(oss.str());
        }
    }
}

}  // namespace detail
}  // namespace halo
