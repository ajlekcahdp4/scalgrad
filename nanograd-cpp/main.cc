#include "engine.hpp"

#include <iostream>

int main() {
  red_engine::value<double> a(1.0);
  red_engine::value<double> b(2.0);
  auto c = a - b;
  c.backprop();
  std::cout << c << a << b << "\n";
}