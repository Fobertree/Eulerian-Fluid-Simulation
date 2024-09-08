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

// Control backend
/*
 * Routines
 * Determine time step dt based on CFL
 * Advect: uA = advect(un, dt, un)
 * uB = uA + dt g
 * Project (ensure divergence-free and pressure update for incompressibility): project(dt, uB)
 */
constexpr float g{9.8};

auto grav = [](double x) { return x + g; };
// velocity field

// Marker-and-Cell 2d grid
class MAC
{
public:
    MAC(size_t r_size, size_t c_size)
    {
        // stagger pressures due to MAC setup
        pressures = mat_d (r_size+1, c_size+1);
        // r, c -> [(r,c),(r,c+1), (r+1,c), (r+1,c+1)]
        velocities = mat_d (r_size, c_size);
    }
    // STEP 1
    void update_ts()
    {
        // CFL condition
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
private:
    // separating MAC into two grids (pressure, and velocity)
    mat_d pressures;
    mat_d velocities;
    double u_max{0};
    float dx; // bro im so d

    size_t rows, cols;
};

void advect(vec2 u, float dt, float q);

void project(float dt, vec2 u);


#endif //GRID_FLUID_MATH_H
