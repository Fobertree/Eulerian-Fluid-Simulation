#ifndef CELL_H
#define CELL_H

#include <vector>
// using namespace std;
// marker-and-cell
// use smth to get cell coord to interpolate
// store cells in an indexable 2d array
// semi-lagrangian advection here or in cell?
// Material Derivative? https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8460516/

class Cell
{
public:
    Cell(float p, std::vector<float> vel)
    {
        float pressure = p;
        std::vector<float> velocities = vel;
    }

    // constructor for unmoving boundary cell
    Cell(float p)
    {
        float pressure = p;
        std::vector<std::vector<float>> velocities = {{0, 0},
                                                      {0, 0},
                                                      {0, 0},
                                                      {0, 0}};
    }

    // copy constructor
    Cell(Cell &other)
    {
        pressure = other.pressure;
        velocities = other.velocities;
    }

    float getPressure()
    {
        return pressure;
    }

    std::vector<std::vector<float>> getVelocities()
    {
        return velocities;
    }

private:
    float pressure;
    std::vector<std::vector<float>> velocities;
};

#endif