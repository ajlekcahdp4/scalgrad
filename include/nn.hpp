#pragma once

#include "engine.hpp"

#include <cstdlib>

#include <algorithm>
#include <numeric>
#include <variant>
#include <vector>

namespace nn {

template <typename T> struct module {
  using value_type = T;
  using scal = typename red_engine::scalar<value_type>;
  using pointer = typename scal::pointer;

  virtual ~module() {}

  virtual std::vector<pointer> parameters() const = 0;

  void zero_grad() {
    for (auto &elem : parameters())
      elem.grad = 0;
  }
};

template <typename T> class neuron : public module<T> {
public:
  using typename module<T>::value_type;
  using typename module<T>::pointer;
  using typename module<T>::scal;

private:
  std::vector<pointer> weights;
  pointer              base;

public:
  neuron(std::size_t nin) : weights{nin} {
    srand(1337);
    base = scal::create(double(rand() % 5000) / 1000);
    for (auto i = 0; i < nin; ++i) {
      auto rn = scal::create(double(rand() % 1000) / 1000);
      weights[i] = rn;
    }
  }

  std::size_t nin() const { return weights.size(); }

  pointer operator()(const std::vector<pointer> &x) const {
    std::vector<pointer> prod{x.size()};
    // std::transform(x.begin(), x.end(), weights.begin(),
    //                std::back_inserter(prod), std::multiplies<pointer>{});
    for (unsigned i = 0; i < x.size(); ++i)
      prod[i] = (x[i] * weights[i]);
    auto act = std::accumulate(prod.begin(), prod.end(), base);
    return red_engine::relu(act);
  }

  std::vector<pointer> parameters() const override {
    auto res = weights;
    res.push_back(base);
    return res;
  }
};

template <typename T> class layer : public module<T> {
public:
  using typename module<T>::value_type;
  using typename module<T>::pointer;
  using typename module<T>::scal;

private:
  std::vector<neuron<value_type>> neurons;

public:
  layer(const std::size_t nin, const std::size_t nout) {
    if (!nout) throw std::out_of_range("Attempt to create a layer with 0 outputs.");
    neurons.reserve(nout);
    for (auto i = 0; i < nout; ++i)
      neurons.emplace_back(nin);
  }

  // Returns number of input of neurons in the layer (all the same).
  std::size_t nin() const { return neurons[0].nin(); }

  std::variant<std::vector<pointer>, pointer> operator()(const std::vector<pointer> &x) const {
    if (x.size() > nin()) throw std::out_of_range("Attempt to call a layer with too many inputs.");
    std::vector<pointer> res;
    res.reserve(neurons.size());
    std::transform(neurons.begin(), neurons.end(), std::back_inserter(res), [&x](auto &n) { return n(x); });
    return res.size() > 1 ? std::variant<std::vector<pointer>, pointer>(res)
                          : std::variant<std::vector<pointer>, pointer>(res[0]);
  }

  std::vector<pointer> parameters() const override {
    std::vector<pointer> res;
    res.reserve(neurons.size() * nin());
    std::for_each(neurons.begin(), neurons.end(), [&res](auto &n) {
      auto p = n.parameters();
      res.insert(res.end(), p.begin(), p.end());
    });
    return res;
  }
};

template <typename T> class MLP : module<T> {
public:
  using typename module<T>::value_type;
  using typename module<T>::pointer;
  using typename module<T>::scal;

private:
  std::vector<layer<value_type>> layers;

public:
  MLP(const std::size_t nin, const std::vector<std::size_t> nouts) {
    if (!nouts.size()) throw std::out_of_range("Attempt to create a MLP with 0 layers.");
    layers.reserve(nouts.size());
    std::vector<std::size_t> sz;
    sz.reserve(nouts.size() + 1);
    sz.push_back(nin);
    sz.insert(sz.end(), nouts.begin(), nouts.end());
    for (unsigned i = 0; i < nouts.size(); ++i)
      layers.emplace_back(sz[i], sz[i + 1]);
  }

  // Returns number of inputs of neurons in the input layer of MLP.
  std::size_t nin() const { return layers[0].nin(); }

  std::variant<std::vector<pointer>, pointer> operator()(std::vector<pointer> x) const {
    for (auto &l : layers) {
      auto tmp = l(x);
      if (tmp.index() == 0)
        x = std::get<0>(tmp);
      else
        return std::get<1>(tmp);
    }
    return x;
  }

  std::vector<pointer> parameters() const override {
    std::vector<pointer> res;
    res.reserve(layers.size() * nin());
    std::for_each(layers.begin(), layers.end(), [&res](auto &l) {
      auto p = l.parameters();
      res.insert(res.end(), p.begin(), p.end());
    });
    return res;
  }
};

} // namespace nn