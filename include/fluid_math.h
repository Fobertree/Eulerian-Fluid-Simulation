//
// Created by Alexa on 9/8/2024.
//

#ifndef GRID_FLUID_MATH_H
#define GRID_FLUID_MATH_H

#include <Eigen/Core>
//#include <Eigen/Dense>

#include <utility> //pair
#include <cmath> //floor, ceil, pow
#include <iostream>

typedef Eigen::Vector2d vec2;
typedef Eigen::VectorXd vec_d;
//typedef Eigen::MatrixX2d mat2;
typedef Eigen::MatrixX<double> mat_d;
//typedef std::pair<int,int> pii;
//typedef std::pair<double, double> pdd;
//typedef std::pair<float,float> pff;

// Control backend
/*
 * Routines
 * Determine time step dt based on CFL
 * Advect: uA = advect(un, dt, un)
 * uB = uA + dt g
 * Project (ensure divergence-free and pressure update for incompressibility): project(dt, uB)
 */
constexpr float g{9.8};

typedef enum CellType
{
    FLUID,
    AIR,
    SOLID
} CellType;

// cant tell if im a genius or stupid
template <typename T = double>
struct dataStruct {
    size_t r_size{0}, c_size{0};
    std::vector<T> data = std::vector<T>();
    // setter
    T& operator() (int i, int j) {
        return &data.at(i * r_size + j);
    }

    void setData(const std::vector<T>& new_data)
    {
        data = std::move(new_data);
    }
};

typedef dataStruct<CellType> labelVec;

auto grav = [](double x) { return x + g; };
// velocity field

// Marker-and-Cell 2d grid
class MAC
{
public:
    MAC(size_t r_size, size_t c_size, double dx)
    {
        rows = r_size, cols = c_size;
        size_t rc{r_size * c_size};
        // stagger pressures due to MAC setup
        pressures = mat_d (r_size+1, c_size+1);
        // r, c -> [(r,c),(r,c+1), (r+1,c), (r+1,c+1)]
        velo_x = mat_d (r_size+1, c_size);
        velo_x = mat_d (r_size, c_size+1);

        // allocate size for A matrix
        Adiag = mat_d(rc, rc);
        Ax = mat_d(rc,rc);
        Ay = mat_d(rc,rc);

        // condense to 1d vec. Can revert if we find it better
        // overloading () for indexing sounds stupid
        labels = std::vector<CellType>(r_size * c_size, AIR);
        this->dx = dx;
    }
    // STEP 1
    void update_ts()
    {
        // CFL condition
        ts = CFL_num * (dx/u_max);
    }

    // STEP 2
    void body_force_update()
    {
        // update velocity with gravity (and external forces)
        // material derivative of u = g

        //map_mat(velocities, &MAC::update_gravity);
        // gravity update
        velo_y.unaryExpr([&](double x) { return x + g; });
    }

//    void static update_gravity(double& x)
//    {
//        x += g;
//    }

    void static map_mat(mat_d& mat, double MAC::*func(double))
    {
        // generalized map function
        mat.unaryExpr(func);
    }
    // STEP 3: ADVECTION
    // advect quantity q: velocity,

    vec2 advect(const vec2& ug, float q, const vec2& x_start)
    {
        // xg is x_start, or current pos at timestep t
        // we advect backwards to get xp, old pos at timestep t-1
        // then, interpolate values and return
        // This function will be mapped to every cell
        vec2 x_mid, x_p, u_mid, u_p;

        // TODO: consider abstracting second-order runge-kutta
        // second-order runge-kutta
        x_mid = advectPos(ug, x_start, ts * 0.5);
        u_mid = linear_interpolate_u(x_mid);
        x_p = advectPos(u_mid, x_mid, ts * 0.5);
        // VELOCITY ADVECTION RESULT
        u_p = linear_interpolate_u(x_p);

        return u_p;
    }

    [[nodiscard]] vec2 advectPos(const vec2& u, const vec2& pos, double dt = NULL) const {
        // removed t arg since 'global'
        // below feels spaghetti but default args must be known at compile-time (no non-static)
        // TODO: consider setting ts as static
        if (dt == NULL)
            dt = ts;

        vec2 du(u(0)/dt, u(1)/dt);

        return {pos(0) - du(0), pos(1) - du(1)};
    }
    vec2 linear_interpolate_u(const vec2& pos) {
        // interpolate velocity
        // bilinear interpolation for now. Consider cubic for later
        // TODO: do this
        // q^{n+1}_i = (1-\alpha) q_j ^n + \alpha q^n _ {j+1}

        double a_x, a_y;
        int i = static_cast<int>(floor(pos(0))),
            j = static_cast<int>(floor(pos(1)));
        double u,v;

        // alphas
        a_x = std::modf(pos(0), nullptr);
        a_y = std::modf(pos(1), nullptr);

        // interpolate x/u component along x-axis
        u = (velo_x(i,j) * a_x + velo_x(i,j) * (1.0f-a_x));

        // interpolate y/v component along y=axis, w/ gravity after?
        v = (velo_y(i,j) * a_y + velo_y(i,j) * (1.0f-a_y)) + g;

        return {u,v};
    }
    // PART 3: PROJECTION
    void project(float dt, vec2& u)
    {
        // TODO: main for project routine
        // update pressure values (including solids bounds check)
        // calculate negative divergence
        // get preconditioner via modified incomplete cholesky
        // perform pcg (apply preconditioner)
    }

    void update_pressure();
    void negative_divergence();
    void setupA()
    {
        // setup Ax, Ay, Adiag
        double scale = ts/(rho * dx * dx);
        size_t rs{rows * cols};

        // not sure if I need to reset to 0 every timestep
        Adiag.fill(0);
        Ax.fill(0);
        Ay.fill(0);

        for (int i = 0; i < rs; i++)
        {
            // symmetric matrix, so only need to compute triangular
            for (int j = 0; j < i; j++)
            {
                if (labels[i * rows + j] == FLUID)
                {
                    // X DIRECTION

                    // handle negative x neighbor
                    if (labels[(i-1) * rows + j] == FLUID)
                    {
                        Adiag(i,j) += scale;
                    }
                    // handle positive x neighbor
                    if (labels[(i+1) * rows + j] == FLUID)
                    {
                        Adiag(i,j) += scale;
                        Ax(i,j) = -scale;
                    }
                    else if (labels[(i+1) * rows + cols] == AIR)
                    {
                        Adiag(i,j) += scale;
                    }

                    // Y DIRECTION

                    // handle negative y neighbor
                    if (labels[i * rows + j-1] == FLUID)
                    {
                        Adiag(i,j) += scale;
                    }
                    // handle positive y neighbor
                    if (labels[i * rows + j+1] == FLUID)
                    {
                        Adiag(i,j) += scale;
                        Ay(i,j) = -scale;
                    }
                    else if (labels[(i+1) * rows + cols] == AIR)
                    {
                        Adiag(i,j) += scale;
                    }
                }
            }
        }
    }
    // ???????????????? wtf am i doing
    std::unique_ptr<mat_d> getPreconditioner()
    {
        // Modified Incomplete Cholesky Decomposition Level Zero
        // Create preconditioner here
        // TODO: Check this

        mat_d precon = mat_d(rows,cols);
        setupA();
        double e;
        // tau is split between mic and ic

        for (int i = 1; i < rows; i++)
        {
            for (int j=1; j < cols; j++)
            {
                if (labels.at(i * rows + j) == FLUID)
                {
                    e = Adiag(i,j) - pow(Ax(i-1,j) * precon(i-1,j), 2)
                            - pow((Ay(i,j-1) * precon(i,j-1)), 2)
                            - chol_tune_tau * (Ax(i-1,j) * (Ay(i-1,j)) * pow(precon(i-1,j),2)
                            +Ay(i,j-1)
                            * (Ax(i,j-1)) * pow(precon(i,j-1),2));

                    // Incomplete cholesky: ensure sparsity
                    if (e < chol_safety_sig)
                        e = Adiag(i,j);

                    precon(i,j) = 1/ sqrt(e);
                }
            }
        }
        return std::make_unique<mat_d>(precon);
    }

    std::unique_ptr<vec_d> pcg(const mat_d& precon, const vec_d& b)
    {
        int vec_size = static_cast<int>(rows * cols);
        mat_d A(vec_size,vec_size);
        // TODO: Set M to preconditioner
        mat_d M;
        vec_d p(vec_size), r(vec_size), z(vec_size), s(vec_size);
        // sigma: normalizing scalar
        // alpha: CG step size
        double sigma{0.f}, alpha{0.f}, sigma_new{0.f};
        int iters{0}; // for maximum iteration break

        // p = 0
        p.fill(0);
        // see if we can move instead of copy
        r = b.row(0); // r = b
        s = z.row(0); // s = z

        sigma = z.dot(r);

        while (iters++ < _max_pcg_iter && !test_tol(A, p,b))
        {
            z = A * s;
            alpha = sigma / z.dot(s);

            p = p + alpha * s;
            r = r - alpha * z;

            if (inf_norm(r) <= tol)
            {
                return std::make_unique<vec_d> (p);
            }

            z = M * r;
            sigma_new = z.dot(r);
            s = z + (sigma_new / sigma) * s;
            sigma = sigma_new;
        }

        if (iters > _max_pcg_iter)
        {
            std::cerr << "WARNING::EXCEEDED ITERATION LIMIT ON PCG" << std::endl;
        }
        else
        {
            std::cout << "PCG Iterations: " << iters <<"\n";
        }

        // TODO: fix issue where p goes out of scope. Unique_ptr?
        return std::make_unique<vec_d>(p);
    }

    void pressure_gradient_update()
    {
        // TODO: Pressure update. Handle solid boundary conditions
        double scale{ts * inv_rho / dx};

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // update u
                if ((labels[(i-1) * rows + j] == FLUID) || labels[i * rows + j] == FLUID)
                {
                    if (labels[(i-1) * rows + j] == SOLID || labels[i*rows + j] == SOLID)
                    {
                        // u(i,j) = usolid(i,j);
                    }
                    else
                    {
                        velo_x(i,j) -= scale * (pressures(i,j) - pressures(i-1,j));
                    }
                }
                else
                {
                    velo_x(i,j) = -1.0f; //marking unknown
                }

                // update v
                if ((labels[i * rows + j-1] == FLUID) || labels[i * rows + j] == FLUID)
                {
                    if (labels[i * rows + j-1] == SOLID || labels[i*rows + j] == SOLID)
                    {
                        // u(i,j) = usolid(i,j);
                    }
                    else
                    {
                        velo_y(i,j) -= scale * (pressures(i,j) - pressures(i,j-1));
                    }
                }
                else
                {
                    velo_y(i,j) = -1.0f; //marking unknown
                }
            }
        }
    }

    void get_pressure(int r, int c)
    {
        // TODO: get pressure based on coord
    }

    bool test_tol(const mat_d& A, const vec_d& p, const vec_d& b)
    {
        // If continue PCG iterations: returns false. If done, returns true.
        double norm_r = inf_norm(A * p);
        return norm_r <= tol * inf_norm(b);
    }

    template<typename T = vec_d>
    double inf_norm(const T& v)
    {
        return v.maxCoeff();
    }

    mat_d applyPrecon(mat_d& precon)
    {
        float t;
        // TODO: precon implementation, bounds + dimension checking
        for (int i = 1; i < rows; i++)
        {
            for (int j = 1; j < cols; j++)
            {
                if (labels[i * rows + j] == FLUID)
                {

                }
            }
        }
    }
private:
    // separating MAC into two grids (pressure, and velocity)
    // coefficient matrices for Ap=b
    mat_d Ax, Ay, Adiag;
    mat_d pressures;
    mat_d velo_x, velo_y, v_solid_x, v_solid_y;
    // consider slapping labels into a struct with () operator override for readability
    std::vector<CellType > labels;
    double u_max{0}, dx{0.f}, ts{1};
    // TODO: make sure glgrid is constrained to squares not just any rectangle. Honestly, this is probably just a GLSL thing
    //double dx;
    int CFL_num{5}; // C or \alpha
    static constexpr int _max_pcg_iter{200};
    static constexpr float chol_tune_tau{0.97}, chol_safety_sig{0.25}, rho{0.5};
    static constexpr float tol{10e-6f};
    static constexpr double inv_rho {1/rho};

    size_t rows, cols; // rows must equal cols
};

// up: v[i][j+1];
// down: v[i][j];
// left: u[i][j];
// right: u[i+1][j];


void project(float dt, vec2 u);


#endif //GRID_FLUID_MATH_H
