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

template <typename T> class value {
public:
  using value_type = T;

  value_type data{};
  value_type grad{};

private:
  using pointer = value *;

  std::set<pointer> prev;

  std::function<void()> backward = [&]() {};

  std::vector<std::shared_ptr<value<value_type>>> childs{};

public:
  value(value_type val, std::set<pointer> prevs = {})
      : data{val}, prev{prevs} {}

  virtual ~value() {}

  std::string str() const {
    std::stringstream ss;
    ss << "value of (data = " << data << ", grad = " << grad << ")";
    return ss.str();
  }

  value<value_type> operator+(value<value_type> &other) {
    auto res = value<value_type>{data + other.data, {this, &other}};
    res.backward = [&]() {
      grad += res.grad;
      other.grad += res.grad;
    };
    return res;
  }

  value<value_type> operator*(value<value_type> &other) {
    auto res = value<value_type>{data * other.data, {this, &other}};
    res.backward = [&]() {
      grad += res.grad * other.data;
      other.grad += res.grad * data;
    };
    return res;
  }

  value<value_type> pow(double n) {
    auto res = value<value_type>{pow(data, n), {this}};
    res.backward = [&]() { grad += (n * pow(data, n - 1)) * res.grad; };
    return res;
  }

  value<value_type> operator-() {
    auto neg_one = std::make_shared<value<value_type>>(-1);
    childs.push_back(neg_one);
    return *this * *neg_one;
  }

  value<value_type> operator-(value<value_type> &other) {
    auto neg_other = std::make_shared<value<value_type>>(-other);
    childs.push_back(neg_other);
    return *this + *neg_other;
  }

  std::list<pointer> topological_sort() {
    auto visited = std::set<pointer>{};
    std::list<pointer> sorted;

    if (visited.find(this) == visited.end()) {
      visited.insert(this);
      for (auto child : prev)
        sorted.splice(sorted.end(), child->topological_sort());
      sorted.push_back(this);
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
    std::function<void(pointer)> build = [&](pointer cur) {
      if (nodes.find(cur) == nodes.end()) {
        nodes.insert(cur);
        for (auto kid : cur->prev) {
          edges.insert({kid, cur});
          build(kid);
        }
      }
    };
    build(this);
    return std::pair{nodes, edges};
  }

public:
  void draw_dot(std::string filename) {
    std::fstream fs{filename};
    assert(fs.is_open());
    auto [nodes, edges] = trace();
    fs << "digraph G {\n"
       << "rankdir=LR\n";
    for (auto n : nodes) {
      fs << "node" << n << " [shape=record, label=\" data = " << n->data
         << " | grad = " << n->grad << " \"];\n";
    }

    for (auto [fst, snd] : edges) {
      fs << "node" << fst << "-> node" << snd << "\n;";
    }
    fs << "}\n";
    fs.close();
  }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const value<T> &val) {
  return os << val.str();
}
} // namespace red_engine