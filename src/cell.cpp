#include "cell.h"
#include "functions.h"

// Constructors

// fluid
Cell::Cell(float p, float density, std::vector<float> vel)
{
    this->density = density;
    pressure = p;
    std::vector<float> velocities = vel;
    int rId = r;
    int cId = c;
    float velocity = v;
    cellType label = FLUID;
}

// constructor for unmoving boundary cell (rock solid)
Cell::Cell(float p, float density)
{
    this->density = density;
    pressure = p;
    std::vector<std::vector<float>> velocities = {{0, 0},
                                                  {0, 0},
                                                  {0, 0},
                                                  {0, 0}};
    int rId = r;
    int cId = c;
    float velocity = v;
    cellType label = SOLID;
}

// copy constructor
Cell::Cell(Cell &other)
{
    pressure = other.pressure;
    velocities = other.velocities;
    int rId = other.r;
    int cId = other.c;
    float velocity = other.v;
    cellType label = other.label;
}

// functions

std::vector<std::vector<float>> Cell::getVelocities()
{
    return velocities;
}

float Cell::getVelocity()
{
    return velocity;
}

float Cell::getDensity()
{
    return density;
}

cellType Cell::getLabel()
{
    return label;
}
void update_v(std::vector<float> &u_a, float dt, std::vector<float> g_vec) // g_vec is gravity
{
    // cannot treat g_vec as a scalar
    u_a + scalar_mult(u_a, g_vec);
}

void Cell::setVelocity(float v)
{
    this->velocities = v;
}
void Cell::setDensity(float rho)
{
    density = rho;
}
void Cell::setLabel(cellType cType)
{
    label = cType;
}

float Cell::p()
{
    // p_i,j
    return pressure;
}

float &Cell::u()
{
    // TODO: insert return statement here
}

float &Cell::v()
{
    // TODO: insert return statement here
}

float &Cell::u_1()
{
    // TODO: insert return statement here
}

float &Cell::&u()
{
    // u_(i-1/2,j)
    return &velocities[3];
}

float &Cell::&v()
{
    // u_(i,j-1/2)
    return &velocities[2];
}