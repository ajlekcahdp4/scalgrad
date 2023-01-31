# scalgrad
A minimal C++ autograd engine implementation for scalars. Inspired by [micrograd](https://github.com/karpathy/micrograd). Also implemented small neural network on top of the engine.

## Example
Here is the example of engine usage:
```cpp
#include "engine.hpp"

using scal = typename red_engine::scalar<double>;
int main() {
  auto a = scal::create(2.0);
  auto b = scal::create(3.0);
  auto c = scal::create(4.0);
  auto d = c - b;
  auto r = red_engine::exponentiate((a + b) * c + d);
  r->backprop();
  r->draw_dot("dump.dot");
```

Example of the neural network usage:
```cpp
#include "engine.hpp"
#include "nn.hpp"

using scal = typename red_engine::scalar<double>;
int main() {
  std::vector<scal::pointer> x{scal::create(1.0), scal::create(-2.0)};

  auto n = nn::MLP<double>{2, {2, 1}};
  auto tmp = n(x);
  auto r = std::get<1>(tmp);
  r->backprop();
  r->draw_dot("dump.dot");
}
```
The graph of the neural network example is [here](example/example.png).
You can also try to run the [example](example/src/main.cc) yourself.

## How to build
```terminal
git clone git@github.com:ajlekcahdp4/scalgrad.git
cd scalargrad
cmake -S . -B build -DNOGTEST=False
make -C build -j8 install
```
## How to run (after build)
```terminal
# to run unit tests:
cd build
ctest

# to run the example
cd example
bin/example
```
