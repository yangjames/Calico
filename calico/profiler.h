#ifndef CALICO_PROFILER_H_
#define CALICO_PROFILER_H_

#include <chrono>
#include <iostream>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"


namespace calico {

class Profiler {
 public:
  Profiler() {
    Tic();
  }

  void Tic() {
    stamp_prev_ = std::chrono::high_resolution_clock::now();
  }

  double Toc(absl::string_view msg = "") {
    const auto stamp = std::chrono::high_resolution_clock::now();
    const auto dt_nanos =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            stamp - stamp_prev_).count();
    const double dt = static_cast<double>(dt_nanos) * 1.0e-9;
    if (!msg.empty()) {
      std::cout << absl::StrFormat("Elapsed time: %.9fs - %s", dt, msg)
                << std::endl;
    }
    return dt;
  }

 private:
  std::chrono::high_resolution_clock::time_point stamp_prev_;
};

} // namespace calico

#endif //CALICO_PROFILER_H_
