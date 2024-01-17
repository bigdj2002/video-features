#pragma once

#include <chrono>

class StopWatch
{
public:
  void start()
  {
    t0 = std::chrono::high_resolution_clock::now();
  }

  double stop()
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch() - t0.time_since_epoch()).count() / 1.0e9;
  }

private:
  std::chrono::high_resolution_clock::time_point t0;
};
