#include "functions.h"
#include <cmath>

extern std::vector<std::vector<Cell>> Grid;

// define all declared functions

float second_rk(std::vector<float> x_g, std::vector u_g, float dt)
{
    // 2nd order runge-kutta for semi-lagrangian advection
    // go backwards one timestep
    // x_g is new/current position
    // x_p is old position
    // page 32
    std::vector<float> u_mid, x_mid, x_p;
    x_mid = vec_subtract(x_g, scalar_mult(scalar_mult(u_g, dt), 0.5));
    // std::vector u_mid = interpolate(x_mid,dt); TO-DO: INTERPOLATION.
    // This should be weighted avg from MAC boundaries
    x_p = vec_subtract(x_g, scalar_mult(u_mid, dt));
    return x_p;
}

std::vector<float> project(float dt, float u_vec) // make u divergence-free, enforce solid-wall boundary
{
    // calculate negative divergence b
    // set A
}
float interpolate(float dt, float u_vec);

// multiply a vector by a scalar
void scalar_mult(std::vector<float> &vec, float scalar)
{
    for (float &val : vec)
    {
        if (type(val) == != float) // detect matrix
        {
            scalar_mult(&val, scalar); // recurse through
        }
        else
        {
            val *= scalar;
        }
    }
}
// multiply a vector by a scalar
std::vector<float> scalar_mult(std::vector<float> vec, float scalar)
{
    // only single dimension
    for (float &val : vec)
    {
        val *= scalar;
    }

    return vec;
}
template <class T>
std::vector<std::vector<float>> matrixMult(T matrix_a, T matrix_b) // set up a template for this
{
    try
    {
        int r1 = matrix_a.size();
        int c1 = matrix_a[0].size();
        int r2 = matrix_b.size();
        int c2 = matrix_b.size();

        if (c1 != r2)
        {
            std::runtime_error("Improper dimensions for multiplying matrix.")
        }
        float mult[r1][c2];

        for (int i = 0; i < r1; ++i)
        {
            for (int j = 0; j < c2; ++j)
            {
                mult[i][j] = 0;
                for (int k = 0; k < r2; k++)
                {
                    mult[i][j] += matrix_a[i][k] * matrix_b[k][j];
                }
            }
        }

        return mult
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        printf("Here are matrix dimensions: r1: %f, c1: %f, r2: %f, c2: %f", r1, c1, r2, c2);
    }
}

// subtract two vectors
std::vector vec_subtract(std::vector<float> vec1, std::vector<float> vec2)
{
    // subtract 2x1 vectors
    if (vec1.size() != vec2.size() && vec1.size != 2)
    {
        printf("Encountering error in vec_subtract: vec1 size: %i, vec2 size: %i", vec1.size(), vec2.size());
        std::runtime_error("Invlid vector size for vector subtraction.")
    }
    std::vector<float> res = {vec1[0] - vec2[0], vec1[1] - vec2[1]};
    return res;
}
// linear interpolation
float linear_interpolate(std::vector<float> x_p, float dx, std::vector<std::vector<Cell>> &grid)
{
    // x_p is old point
    // aren't x positions supposed to be a vector?
    // revisit how to calc alpha this doesn't make sense. I think substracting position = pythagorean/eucliean norm
    // q_j, q_j1 come from boundaries of cell
    /*
    Create a function to get x_j, x_j+1, q_j, q_j+1
    */
    std::vector<std::vector<float>> j_vectors = get_xq_j_quantities(grid, x_p);
    // returns vector: {x_j,q_j,x_(j+1),q_(j+1)}
    std::vector<float> x_j = j_vectors[0];
    std::vector<float> q_j = j_vectors[1];
    std::vector<float> x_j1 = j_vectors[2];
    std::vector<float> q_j1 = j_vectors[3];

    float a = (vectorNorm(vec_subtract(x_p - x_j))) / dx;
    // j subscript: old location
    q_i1 = q_j * (1 - a) + a * (q_j1);
    return q_i1;
}

float cubic_interpolate();
float mic_zero();
float dot_product(std::vector<float> v1, std::vector<float> v2)
{
    if (v1.size() != v2.size())
    {
        throw std::invalid_argument(std::format("invalid vector sizes in dot product. Sizes {}, {}", v1.size(), v2.size()));
    }
    float res{0};
    int length{v1.size()};

    for (int i = 0; i < length; i++)
    {
        res += v1[i] * v2[i];
    }

    return res;
}

float vectorNorm(std::vector<float> vec)
{
    // Euclidean norm
    int res{0};

    for (&i : vec)
    {
        res += pow(i, 2);
    }

    return sqrt(res);
}

float advect(u_vec, dt, q); // simple advection from cell bounds
{
}
void applyPreconditioner(std::vector<std::vector<float>> preconditioner, std::vector<float> vector); // matrixMult
std::vector<float> semiLagrangian(std::vector<float> x_g, std::vector u_g, float dt, std::vector<std::vector<Cell>> &grid)
{
    // call runge-kutta to get x_p
    std::vector<float> x_p = second_rk(x_g, u_g, dt);
    // get velocity vector at x_p TO-DO
    linear_interpolate(x_p, dx, grid);
}

void checkTimestep(float &dt, float u_max, float dx, int cfl_num)
{
    float threshold = cfl_num * (dx / u_max);
    dt = min(dt, threshold);
}
float updatePressure(std::vector<float> u_n, float dt, float density, float dx, float p_next, float p_low);
float updateVelocity(Grid &grid, float dt, float density, float dx)
{
    // pressure gradient update
    float scale{dt / (density * dx)};
    for (int i = 0; i < grid.size(); i++)
    {
        for (int j = 0; j < grid[0].size(); j++)
        {
            // update u
            if (label(i - 1, j) == FLUID || label(i, j) == FLUID)
            {
                if (label(i, j - 1) == SOLID || label(i, j) == SOLID)
                {
                    u(i, j) = usolid(i, j);
                }
                else
                {
                    u(i, j) -= scale * (p(i, j) - p(i - 1, j));
                }
            }
            else
            {
                // mark as unknown
                u(i, j) = <NULL, NULL>;
            }
            // update v
            if (label(i, j - 1) == FLUID || label(i, j) == FLUID)
            {
                if (label(i, j - 1) == SOLID || label(i, j) == SOLID)
                {
                    v(i, j) = vsolid(i, j);
                }
                else
                {
                    v(i, j) -= scale * (p(i, j) - p(i, j - 1));
                }
            }
            else
            {
                // mark as unknwon
                v(i, j) = <NULL, NULL>;
            }
        }
    }
}

float rhs(float dt, float density, float dx, float p_next, float p_low)
{
    return (dt / density) * ((p_next - p_low) / dx);
}

std::vector<std::vector<float>> setupA(std::vector<std::vector<Cell>> &grid, float dx, float density)
{
    // prepare A
    using namespace Grid;

    int r{grid.size()};
    int c{grid[0].size()};
    float scale { dt / (density * dx * dx) }

    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            if (label(i, j) == FLUID)
            {
                // handle negative x neighbor
                if (label(i - 1, j) == FLUID)
                {
                    A(i, j) += scale;
                }
                // handle positive x neighbor
                if (label(i + 1, j) == FLUID)
                {
                    Adiag(i, j) += scale;
                    Ax(i, j) = -scale;
                }
                else if (label(i + 1, j) == EMPTY)
                {
                    Adiag(i, j) += scale;
                }
                // handle negative y neighbor
                if (label(i, j - 1) == FLUID)
                {
                    Adiag(i, j) += scale;
                }
                // handle positive y neighbor
                if (label(i, j + 1) == FLUID)
                {
                    Adiag(i, j) += scale;
                    Ay(i, j) = -scale;
                }
                else if (label(i, j + 1) == EMPTY)
                {
                    Adiag(i, j) += scale;
                }
            }
        }
    }
}

std::vector<std::vector<float>> get_xq_j_quantities(std::vector<std::vector<Cell>> &grid, std::vector<float> x_p)
{
    // return vector: {x_j,q_j,x_(j+1),q_(j+1)}
    // do we need to check if points are exact integer?
    std::vector<std::vector<float>> j_vecs;
    int r_low{floor(x_p[0])};
    int r_high{ceil(x_p[0])};
    int c_low{floor(x_p[1])};
    int c_high{ceil(x_p[1])};

    std::vector<float> x_j = {r_low, c_low};
    std::vector<float> q_j = {0, 0};
    std::vector<float> x_j1 = {r_high, c_high};
    std::vector<float> q_j1 = {0, 0};

    // Do something

    j_vecs.push_back(x_j);
    j_vecs.push_back(q_j);
    j_vecs.push_back(x_j1);
    j_vecs.push_back(q_j1);

    return j_vecs;
}