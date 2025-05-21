# A Simple Neural Network Framework Implemented in C

Reference: [Tsoding](https://youtu.be/PGSba51aRYU?si=t_VngrwOzOb0Zxlz)

```c
size_t arch[] = {4, 2, 2};
NN nn = nn_alloc(arch, ARRAY_LEN(arch));
NN g = nn_alloc(arch, ARRAY_LEN(arch));
nn_rand(nn, 0, 1);

float rate = 1;

// ti -- training input
// to -- taining output
printf("cost = %f\n", nn_cost(nn, ti, to));
for (size_t i = 0; i < 5000; i++) {
  nn_backprop(nn, g, ti, to);
  nn_learn(nn, g, rate);
  printf("cost = %f\n", nn_cost(nn, ti, to));
}
```
