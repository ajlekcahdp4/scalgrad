#include "engine.hpp"
// #include "nn.hpp"

#include <iostream>

int main() {
  auto a = red_engine::scalar<double>::create(1.0);
  auto b = red_engine::scalar<double>::create(2.0);
  auto nb = -b;
  auto c = a - b;
  c->draw_dot("gout.dot");
  c->backprop();
  std::cout << "a = " << a << "\nb = " << b << "\nc = a - b = " << c;
}