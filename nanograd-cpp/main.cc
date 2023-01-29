#include "engine.hpp"
// #include "nn.hpp"

#include <iostream>

int main() {
  auto a = red_engine::scalar<double>::create(6.0);
  auto b = red_engine::scalar<double>::create(-3.0);
  auto r = a / b;
  r->backprop();
  r->draw_dot("gout.dot");
}