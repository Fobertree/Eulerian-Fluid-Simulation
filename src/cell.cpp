#include "cell.h"
#include "functions.h"

// Constructors

// fluid
Cell::Cell(float p, std::vector<float> vel)
{
    float pressure = p;
    std::vector<float> velocities = vel;
    int rId = r;
    int cId = c;
    float velocity = v;
    cellType label = FLUID;
}

// constructor for unmoving boundary cell (rock solid)
Cell::Cell(float p)
{
    float pressure = p;
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

float Cell::get_velocity()
{
    return velocity;
}

cellType Cell::getLabel()
{
    return label;
}

float Cell::p()
{
    // p_i,j
    return pressure;
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