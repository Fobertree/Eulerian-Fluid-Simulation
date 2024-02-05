#ifndef PARTIAL_H
#define PARTIAL_H

struct partial_derivative
{
    partial_derivative(F f, size_t i) : f(f), index(i) {}

    double operator()(double *x)
    {
        static const double eps = 1e-4;

        double old_xi = x[index];
        x[index] = old_xi + eps;
        double f_plus = f(x);

        // circumvent the fact that a + b - b != a in floating point arithmetic
        volatile actual_eps = x[index];
        x[index] = old_xi - eps;
        actual_2eps -= x[index] double f_minus = f(x);

        return (f_plus - f_minus) / actual_2eps;
    }

private:
    F f;
    size_t index;
};

template <typename F>
partial_derivative<F> partial(F f, index i)
{
    return partial_derivative<F>(f, i);
}

#endif