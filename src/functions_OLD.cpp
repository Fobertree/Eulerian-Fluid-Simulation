#include "functions_OLD.h"
#include <cmath>

extern std::vector<std::vector<Cell>> cellGrid;
extern Grid grid;
extern dx;

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

std::vector<float> project(float dt, Cell cell) // make u divergence-free, enforce solid-wall boundary
{
    // u_vec
    // 77
    // Ap = b, where b consists of the negative divergences of the fluid cell
    // both p and b must be stored in 2D or 3D cell, since they hold the values for all the cells
    // calculate negative divergence b
    // setup A
    static vvf b;
    // update b

    for (int i = 0; i < b.size(); i++)
    {
        for (int j = 0; j < b[0].size();j++)
        {
            // rewrite rhs to work around Cell Object
            b[i][j] = rhs(cellGrid[i][j]);
        }
    }

    vvf A = setupA(&grid, dt, density);
    // get MIC(0) preconditioner
    vvf precon = getPreconditioner(&grid);
    applyPreconditioner();

    // Perform PCG
    p = pcg(A,b,&grid);

    // return pressure gradient, which is mapped onto velocity field
    // this preserves incompressibility
    // can make this void then map inside the function
}

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

std::vector<float> linear_combination(vf v1, vf v2, float s1 = 1, float s2 = 1)
{
    if (s1 != 1)
    {
        scalar_mult(&v1, s1);
    }

    if (s2 != 1)
    {
        scalar_mult(&v2, s2);
    }

    std::assert(v1.size() == v2.size());

    for ()
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
        std::runtime_error("Invalid vector size for vector subtraction.")
    }
    std::vector<float> res = {vec1[0] - vec2[0], vec1[1] - vec2[1]};
    return res;
}
// linear interpolation
float linear_interpolate(std::vector<float> x_p, float dx, std::vector<std::vector<Cell>> &grid)
{
    // x_p is old point
    // aren't x positions supposed to be a vector?
    // revisit how to calc alpha this doesn't make sense. I think subtracting position = pythagorean/euclidean norm
    // q_j, q_j1 come from boundaries of cell
    /*
    Create a function to get x_j, x_j+1, q_j, q_j+1
    std::round
    */

    std::vector<float> cell_coords = {std::round(x_p[0]), std::round(x_p[1])};

    Cell gridCell = grid[cell_coords[0], cell_coords[1]]; // should avoid out-of-bounds

    //
    // std::vector<std::vector<float>> j_vectors = get_xq_j_quantities(grid, x_p);
    // returns vector: {x_j,q_j,x_(j+1),q_(j+1)}
    std::vector<std::vector<float>> j_vectors = gridCell.getVelocities();
    std::vector<float> a = {1 - (cell_coords[0] + 0.5 - x_p[0]), 1 - (cell_coords[1] + 0.5 - x_p[1])}; // 0 <= alpha <= 1
    // treat dx = 1, since coords are defined in units of dx
    // alpha is difference weight from lower bound

    // qn+1 = (1 − α)q^n_j + αq^n_j+1.

    std::vector<float> q;
    q.push_back((1 - a) * j_vectors[3] + a * j_vectors[1]);
    q.push_back((1 - a) * j_vectors[2] + a * j_vectors[0]);

    return q;
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

    // seems replaceable by std::inner_product

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

    for (auto &i : vec)
    {
        res += pow(i, 2);
    }

    return sqrt(res);
}

void applyPreconditioner(std::vector<std::vector<float>> precon, std::vector<float> r)
{
    // Perform both triangular solves to apply MIC preconditioner
    int nx{precon.size() - 1};
    int ny{precon[0].size() - 1};
    // Lq = r (forward substitution). L is lower triangular matrix
    // solve for q. A is our coefficient matrix and r is b vector
    float t;
    std::vector<std::vector<float>> q(ny, std::vector<float>(nx, 0));

    // optimization: put i on inner-loop because of how memory is arranged
    for (int j = 1; j <= ny; j++)
    {
        for (int i = 1; i <= nx; i++)
        {
            if (label(i, j) == FLUID)
            {
                t = r - getAx(i - 1, j) * precon[i - 1][j] * q[i - 1][j] - getAy(i, j - 1) * precon[i][j - 1] * q[i][j - 1];
                q[i][j] = t * precon[i][j];
            }
        }
    }

    // L^T z = q (back substitution)

    std::vector<std::vector<float>> z(ny, std::vector<float>(nx, 0));

    for (int j = ny; j >= 1; j--)
    {
        for (int i = nx; i >= 1; i--)
        {
            if (label(i, j) == FLUID)
            {
                t = q[i][j] - getAx(i, j) * precon[i][j] * z[i + 1][j] - getAy(i, j) * precon[i][j] * z[i][j + 1];
                z[i][j] = t * precon[i][j];
            }
        }
    }
    // Solve z = Mr

    return z;
}

std::vector<float> semiLagrangian(std::vector<float> x_g, std::vector u_g, float dt, std::vector<std::vector<Cell>> &grid)
{
    // This function serves as our advection algorithm
    // mainly advects velocity, but can be used to advect other quantities
    // This function is the same as advect(u_vec, dt, q) from the textbook
    // call runge-kutta to get x_p
    std::vector<float> x_p = second_rk(x_g, u_g, dt);
    std::vector<float> v_p;

    // linear interpolate to obtain velocity vector at x_p
    v_p = linear_interpolate(x_p, dx, grid);
}

void checkTimestep(float &dt, float u_max, float dx, int cfl_num)
{
    float threshold = cfl_num * (dx / u_max);
    dt = min(dt, threshold);
}
float updatePressure(std::vector<float> u_n, float dt, float density, float dx, float p_next, float p_low)
{
    // part of projection routine
    return 0.0;
}
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
                u(i, j) = {NULL, NULL};
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
                // mark as unknown
                v(i, j) = {NULL, NULL};
            }
        }
    }
}

float rhs(Cell cell)
{
    // returns singular rhs float for respective cell
    static float scale = 1/static_cast<float>(dx);
    // check if cell is on the edge
    return (-scale * *(u_1()) - *(u()) + *(v_1()) + (v()))
}

std::vector<std::vector<float>> setupA(std::vector<std::vector<Cell>> &grid, float dx, float density)
{
    // prepare A. 5.5
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
    // This function returns velocity quantities used in linear interpolation
    // return vector: {x_j,q_j,x_(j+1),q_(j+1)}
    // do we need to check if points are exact integer?
    std::vector<std::vector<float>> j_vecs;
    auto gridCell = grid[std::round(x_p[0]), std::round(x_p[1])];
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

// Projection Routine

std::vector<std::vector<float>> getPreconditioner(Grid &grid)
{
    // Perform MIC(0)
    /*
    Cholesky Decomposition:
    Diag: 𝐷_(𝑗,𝑗)=√(𝐴_(𝑗,𝑗)−∑_(𝑘=1)^(𝑗−1)(𝐷_(𝑗,𝑘)^2 ))
    Non-diag: 𝐷_(𝑖,𝑗)=(𝐴_(𝑖,𝑗)−∑_(𝑘=1)^(𝑗−1)(〖𝐷_(𝑖,𝑘) 𝐷_(𝑗,𝑘) 〗))/𝐷_(𝑗,𝑗)

    MIC compromises on accuracy but has a lot faster preconditioner by generating a sparser matrix
    */
    std::vector<std::vector<float>> precon(grid.size(), std::vector<float>(grid[0].size(), 0));

    float tau{0.97};   // tuning constant
    float sigma{0.25}; // safety constant
    float e{0};        // temporary value, helps when summing squares (see Cholesky decomp formula)

    for (int i = 1; i < grid.size(); i++)
    {
        for (int j = 1; j < grid.size(); j++)
        {
            if (label(i, j) == FLUID)
            {
                e = getAdiag(i, j) - pow(getAx(i, j) * precon[i - 1][j], 2) - pow(getAy(i, j) * precon[i][j - 1], 2) - tau * (getAx(i, j) * (getAy(i, j) * pow(precon[i - 1][j], 2)) + getAy(i, j) * (getAx(i, j) * pow(precon[i - 1][j], 2)));

                if (e < sigma * getAdiag(i, j))
                {
                    e = getAdiag(i, j);
                }
                precon[i][j] = 1 / (sqrt(e));
            }
        }
    }

    return precon;
}

std::vector<std::vector<float>> fowardSub(std::vector<std::vector> A, std::vector<float> b)
{
    // Lq = b
    // solve for q, which is pressure column vector
    int n{A.size()}; // # of rows
    std::vector<float> x = std::vector<float>(n, 0);
    float tmp;

    for (int i = 0; i < n; i++)
    {
        tmp = b[i];
        for (int j = 0; j < i - 1; j++)
        {
            tmp = tmp - (A[i][j] * x[j]);
        }
        x[i] = tmp / (L[i][i])
    }
    return x;
}

std::vector<float> pcg(std::vector<std::vector<float>> A, std::vector<float> r, Grid &grid)
{
    // Ap = b, where b is r
    // optimize with BLAS. This should be the most time-consuming task.
    const static int maxItr = 50;
    const static int tol = std::pow(10,-3);

    std::vector<float> p(A[0].size(), 0);

    // check if r is all zeros
    bool zeros = std::all_of(r.begin(), r.end(), [](int i)
                             { return i == 0; });
    if (zeros)
    {
        return p;
    }
    std::vector<float> s;
    float beta;
    float sigma;
    float sigma_new

        std::vector<std::vector<float>>
            precon = getPreconditioner(grid &);
    std::vector<float> z = applyPreconditioner(precon, r);
    sigma = dot_product(z, r);

    for (int i = 0; i < maxItr; i++)
    {
        z = applyA(s); // create apply A
        a = sigma / dot_product(z, s);

        // linear combination of vectors
        p = linear_combination(s, p, a);
        z = linear_combination(z, r, a);

        if (r <= tol)
        {
            return p;
        }
        z = applyPreconditioner(precon, r);
        sigma_new = dot_product(z, r);

        beta = sigma_new / sigma;
        s = linear_combination(s, z, beta);
        sigma = sigma_new;
    }

    std::cout << "Iteration limit exceeded." << std::endl;
    return p;
}

void addGravity(std::vectr<float> &u_a, float dt, float g)
{
    u_a[1] = u_a[1] + dt * g
}
