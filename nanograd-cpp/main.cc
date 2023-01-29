#include "engine.hpp"
// #include "nn.hpp"

#include <iostream>

int main() {
  auto a = red_engine::scalar<double>::create(1.0);
  auto b = red_engine::scalar<double>::create(2.0);
  auto c = a - b;
  c->backprop();
  c->draw_dot("gout.dot");

  std::cout << "a = " << a << "\nb = " << b << "\nc = a - b = " << c;
}