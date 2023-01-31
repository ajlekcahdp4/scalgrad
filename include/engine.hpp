#pragma once

#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace red_engine {

template <typename T> class scalar {
public:
  using value_type = T;
  using pointer = typename std::shared_ptr<scalar>;

  value_type data{};
  value_type grad{};
  std::function<void()> backward = []() {};

private:
  using self = scalar<value_type>;

  std::set<pointer> prev;
  std::string op = "";

  scalar(value_type val, std::set<pointer> prevs = {}, std::string oper = "")
      : data{val}, prev{prevs}, op{oper} {}

public:
  template <typename... Args> static pointer create(Args &&...args) {
    auto raw_ptr = new scalar{(std::forward<Args>(args))...};
    return pointer(raw_ptr);
  }

  virtual ~scalar() {}

  std::string str() const {
    std::stringstream ss;
    ss << "scalar of (data = " << data << ", grad = " << grad << ")";
    return ss.str();
  }

  std::list<pointer> topological_sort() {
    auto visited = std::set<pointer>{};
    std::list<pointer> sorted;
    auto this_ptr = create(*this);

    if (visited.find(this_ptr) == visited.end()) {
      visited.insert(this_ptr);
      for (auto child : prev)
        sorted.splice(sorted.end(), child->topological_sort());
      sorted.push_back(this_ptr);
    }
    return sorted;
  }

  void backprop() {
    auto topo = topological_sort();
    grad = 1;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      auto &cur = *it;
      cur->backward();
    }
  }

private:
  auto trace() {
    std::set<pointer> nodes{};
    std::set<std::pair<pointer, pointer>> edges{};
    std::function<void(pointer)> build = [&nodes, &edges, &build](pointer cur) {
      if (nodes.find(cur) == nodes.end()) {
        nodes.insert(cur);
        for (auto kid : cur->prev) {
          edges.insert({kid, cur});
          build(kid);
        }
      }
    };
    auto this_ptr = create(*this);
    build(this_ptr);
    return std::pair{nodes, edges};
  }

public:
  void draw_dot(std::string filename) {
    std::fstream fs{filename, std::ofstream::out | std::ofstream::trunc};
    assert(fs.is_open());
    auto [nodes, edges] = trace();
    fs << "digraph G {\n"
       << "rankdir=LR\n";
    for (auto n : nodes) {
      fs << "node" << n.get() << " [shape=record, label=\" data = " << n->data
         << " | grad = " << n->grad << " \"];\n";
      if (n->op != "") {
        fs << "node" << n.get() << "_" << std::addressof(n->op)
           << " [label = \"" << n->op << "\"];\n";
        fs << "node" << n.get() << "_" << std::addressof(n->op) << " -> node"
           << n.get() << ";\n";
      }
    }

    for (auto [fst, snd] : edges) {
      fs << "node" << fst.get() << " -> node" << snd.get() << "_"
         << std::addressof(snd->op) << ";\n";
    }
    fs << "}\n";
    fs.close();
  }
};

template <typename T>
std::shared_ptr<scalar<T>> operator+(std::shared_ptr<scalar<T>> lhs,
                                     std::shared_ptr<scalar<T>> rhs) {
  auto res =
      scalar<T>::create(lhs->data + rhs->data,
                        std::set<std::shared_ptr<scalar<T>>>{lhs, rhs}, "+");
  res->backward = [lhs, rhs, res]() {
    lhs->grad += res->grad;
    rhs->grad += res->grad;
  };
  return res;
}

template <typename T>
std::shared_ptr<scalar<T>> operator*(std::shared_ptr<scalar<T>> lhs,
                                     std::shared_ptr<scalar<T>> rhs) {
  auto res =
      scalar<T>::create(lhs->data * rhs->data,
                        std::set<std::shared_ptr<scalar<T>>>{lhs, rhs}, "*");
  res->backward = [lhs, rhs, res]() {
    lhs->grad += res->grad * rhs->data;
    rhs->grad += res->grad * lhs->data;
  };
  return res;
}

template <typename T>
std::shared_ptr<scalar<T>> operator-(std::shared_ptr<scalar<T>> scal) {
  auto neg_one = scalar<T>::create(-1.0);
  return scal * neg_one;
}

template <typename T>
std::shared_ptr<scalar<T>> operator-(std::shared_ptr<scalar<T>> lhs,
                                     std::shared_ptr<scalar<T>> rhs) {
  auto neg_rhs = -rhs;
  return lhs + neg_rhs;
}

template <typename T>
std::shared_ptr<scalar<T>> operator/(std::shared_ptr<scalar<T>> lhs,
                                     std::shared_ptr<scalar<T>> rhs) {
  return lhs * (power(rhs, -1));
}

template <typename T>
std::shared_ptr<scalar<T>> relu(std::shared_ptr<scalar<T>> scal) {
  auto res =
      scalar<T>::create((scal->data > 0 ? scal->data : 0),
                        std::set<std::shared_ptr<scalar<T>>>{scal}, "ReLU");
  res->backward = [res, scal]() { scal->grad += (res->data > 0) * res->grad; };
  return res;
}

template <typename T>
std::shared_ptr<scalar<T>> exponentiate(std::shared_ptr<scalar<T>> scal) {
  auto res = scalar<T>::create(
      exp(scal->data), std::set<std::shared_ptr<scalar<T>>>{scal}, "exp");
  res->backward = [res, scal]() { scal->grad += res->data * res->grad; };
  return res;
}

template <typename T>
std::shared_ptr<scalar<T>> power(std::shared_ptr<scalar<T>> scal, double n) {
  std::stringstream opss;
  opss << "pow " << n;
  auto res =
      scalar<T>::create(pow(scal->data, n),
                        std::set<std::shared_ptr<scalar<T>>>{scal}, opss.str());
  res->backward = [res, scal, n]() {
    scal->grad += (n * pow(scal->data, n - 1)) * res->grad;
  };
  return res;
}

template <typename T>
std::ostream &operator<<(std::ostream &os,
                         const std::shared_ptr<scalar<T>> val) {
  return os << val->str();
}

} // namespace red_engine