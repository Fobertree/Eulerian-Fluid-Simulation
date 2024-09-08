//
// Created by Alexa on 9/8/2024.
//

#ifndef GRID_FLUID_MATH_H
#define GRID_FLUID_MATH_H

#include <Eigen/Core>
//#include <Eigen/Dense>

#include <utility> //pair

typedef Eigen::Vector2d vec2;
typedef Eigen::MatrixX2d mat2;
typedef Eigen::MatrixX<double> mat_d;
typedef std::pair<int,int> pii;
typedef std::pair<double, double> pdd;
typedef std::pair<float,float> pff;

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

auto grav = [](double x) { return x + g; };
// velocity field

// Marker-and-Cell 2d grid
class MAC
{
public:
    MAC(size_t r_size, size_t c_size, double dx)
    {
        // stagger pressures due to MAC setup
        pressures = mat_d (r_size+1, c_size+1);
        // r, c -> [(r,c),(r,c+1), (r+1,c), (r+1,c+1)]
        velocities = mat_d (r_size, c_size);
        this->dx = dx;
        rows = r_size, cols = c_size;
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
        // update with gravity
        // material derivative of u = g

        //map_mat(velocities, &MAC::update_gravity);
        // gravity update
        velocities.unaryExpr([&](double x) { return x + g; });
    }

    double update_gravity(double x)
    {
        return x + g;
    }

    void map_mat(mat_d& mat, double MAC::*func(double))
    {
        // generalized map function
        mat.unaryExpr(func);
    }
    // STEP 3: ADVECTION
    // advect quantity q: velocity,

    pff advect(vec2 ug, float q, vec2 x_start)
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
    }

    [[nodiscard]] vec2 advectPos(vec2 u, vec2 pos, double dt = NULL) const {
        // removed t arg since 'global'
        // below feels spaghetti but default args must be known at compile-time (no non-static)
        // TODO: consider setting ts as static
        if (dt == NULL)
            dt = ts;

        vec2 du(u(0)/dt, u(1)/dt);

        return {pos(0) - du(0), pos(1) - du(1)};
    }
    vec2 linear_interpolate_u(vec2 pos) {
        // interpolate velocity
        // TODO: do this
        // q^{n+1}_i = (1-\alpha) q_j ^n + \alpha q^n _ {j+1}
        return {0.f,0.f};
    }
private:
    // separating MAC into two grids (pressure, and velocity)
    mat_d pressures;
    mat_d velocities;
    double u_max{0}, dx, ts{1};
    // TODO: make sure glgrid is constrained to squares not just any rectangle. Honestly, this is probably just a GLSL thing
    //double dx;
    int CFL_num{5}; // C or \alpha

    size_t rows, cols;
};


void project(float dt, vec2 u);


#endif //GRID_FLUID_MATH_H
