#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>

float td_sum[] = {
    0, 0, 0, 0, 0, 0, //
    0, 0, 0, 1, 0, 1, //
    0, 1, 0, 0, 0, 1, //
    0, 1, 0, 1, 1, 0, //
};

float td_xor[] = {
    0, 0, 0, //
    0, 1, 1, //
    1, 0, 1, //
    1, 1, 0, //
};

float td_or[] = {
    0, 0, 0, //
    1, 0, 1, //
    0, 1, 1, //
    1, 1, 1, //
};

int main(void) {
  srand(time(0));

  float *td = td_sum;
  size_t stride = 6;
  size_t n = 4;
  Mat ti = {
      .rows = n,
      .cols = 4,
      .stride = stride,
      .es = td,
  };

  Mat to = {
      .rows = n,
      .cols = 2,
      .stride = stride,
      .es = td + 4,
  };

  size_t arch[] = {4, 2, 2};
  NN nn = nn_alloc(arch, ARRAY_LEN(arch));
  NN g = nn_alloc(arch, ARRAY_LEN(arch));
  nn_rand(nn, 0, 1);

  float eps = 1e-1;
  float rate = 1;

  printf("cost = %f\n", nn_cost(nn, ti, to));
  for (size_t i = 0; i < 5000; i++) {
#if 1
	nn_finite_diff(nn, g, eps, ti, to);
#else
	nn_backprop(nn, g, ti, to);
#endif
	nn_learn(nn, g, rate);
	printf("cost = %f\n", nn_cost(nn, ti, to));
  }
  printf("cost = %f\n", nn_cost(nn, ti, to));

  // NN_PRINT(nn);

#if 1
  printf("----------------------\n");
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 2; j++) {
      MAT_AT(NN_INPUT(nn), 0, 0) = 0;
      MAT_AT(NN_INPUT(nn), 0, 1) = i;
      MAT_AT(NN_INPUT(nn), 0, 2) = 0;
      MAT_AT(NN_INPUT(nn), 0, 3) = j;
      nn_forward(nn);
      float y0 = MAT_AT(NN_OUTPUT(nn), 0, 0);
      float y1 = MAT_AT(NN_OUTPUT(nn), 0, 1);
      printf("0%zu + 0%zu = %f %f\n", i, j, y0, y1);
    }
  }
#endif

  return 0;
}
