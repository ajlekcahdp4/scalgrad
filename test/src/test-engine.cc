#include <cmath>
#include <gtest/gtest.h>

#include "engine.hpp"

template class red_engine::scalar<double>;
using scal = typename red_engine::scalar<double>;

TEST(test_engine, create) {
  auto a = scal::create(2.0);
  auto b = scal::create(-3);
}

TEST(test_engine, mult) {
  auto a = scal::create(2.0);
  auto b = scal::create(-3);
  auto c = a * b;
  EXPECT_EQ(c->data, -6.0);
  c->backprop();
  EXPECT_EQ(a->grad, b->data * c->grad);
  EXPECT_EQ(b->grad, a->data * c->grad);
}

TEST(test_engine, add) {
  auto a = scal::create(-2.0);
  auto b = scal::create(3.0);
  auto c = a + b;
  EXPECT_EQ(c->data, 1.0);
  c->backprop();
  EXPECT_EQ(a->grad, c->grad);
  EXPECT_EQ(b->grad, c->grad);
}

TEST(test_engine, neg) {
  auto a = scal::create(2.0);
  auto b = -a;
  EXPECT_EQ(b->data, -2.0);
  b->backprop();
  EXPECT_EQ(a->grad, -1.0);
}

TEST(test_engine, sub) {
  auto a = scal::create(-2.0);
  auto b = scal::create(3.0);
  auto c = b - a;
  EXPECT_EQ(c->data, 5.0);
  c->backprop();
  EXPECT_EQ(a->grad, -1.0);
  EXPECT_EQ(b->grad, 1.0);
}

TEST(test_engine, power) {
  auto a = scal::create(2.0);
  auto c = red_engine::power(a, 5);
  c->backprop();
  EXPECT_EQ(c->data, 32.0);
  EXPECT_EQ(a->grad, 80.0);
}

TEST(test_engine, div) {
  auto a = scal::create(2.0);
  auto b = scal::create(8.0);
  auto c = b / a;
  c->backprop();
  EXPECT_EQ(c->data, 4.0);
  EXPECT_EQ(a->grad, -2.0);
  EXPECT_EQ(b->grad, 0.5);
}

TEST(test_engine, exp) {
  auto a = scal::create(2.0);
  auto b = red_engine::exponentiate(a);
  b->backprop();
  EXPECT_EQ(b->data, exp(2.0));
  EXPECT_EQ(a->grad, b->data);
}

TEST(test_engine, ReLU_1) {
  auto a1 = scal::create(2.0);
  auto b1 = scal::create(-3);
  auto c1 = red_engine::relu(a1 + b1);
  c1->backprop();
  EXPECT_EQ(c1->data, 0.0);
  EXPECT_EQ(b1->grad, 0.0);
  EXPECT_EQ(a1->grad, 0.0);

  auto a2 = scal::create(-2.0);
  auto b2 = scal::create(3.0);
  auto c2 = red_engine::relu(a2 + b2);
  c2->backprop();
  EXPECT_EQ(c2->data, 1.0);
  EXPECT_EQ(b2->grad, 1.0);
  EXPECT_EQ(a2->grad, 1.0);
}

TEST(test_engine, ReLU_2) {
  auto a1 = scal::create(2.0);
  auto b1 = scal::create(-3);
  auto c1 = red_engine::relu(a1 * b1);
  c1->backprop();
  EXPECT_EQ(c1->data, 0.0);
  EXPECT_EQ(b1->grad, 0.0);
  EXPECT_EQ(a1->grad, 0.0);

  auto a2 = scal::create(2.0);
  auto b2 = scal::create(3.0);
  auto c2 = red_engine::relu(a2 * b2);
  c2->backprop();
  EXPECT_EQ(c2->data, 6.0);
  EXPECT_EQ(b2->grad, 2.0);
  EXPECT_EQ(a2->grad, 3.0);
}