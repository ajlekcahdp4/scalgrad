#include "engine.hpp"
#include "nn.hpp"

#include <iostream>

using scal = typename red_engine::scalar<double>;
int main() {
  std::vector<scal::pointer> x{scal::create(1.0), scal::create(-2.0)};

  auto n = nn::MLP<double>{2, {2, 1}};
  auto tmp = n(x);
  auto r = std::get<1>(tmp);
  r->backprop();
  r->draw_dot("dump.dot");
}
