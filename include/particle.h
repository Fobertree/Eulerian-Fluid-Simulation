#ifndef PARTICLE_H
#define PARTICLE_H

// implement partial and material derivatives
// advection
class Particle
{
public:
    Particle(int r, int c, float v)
    {
        int rId = r;
        int cId = c;
        float velocity = v;
    }

    float runge_kutta()
    {
    }

    float get_velocity()
    {
        return velocity;
    }

private:
    int rId;
    int cId;
    float velocity; // variables with normals?
    // shader - vertex and fragment
    // adjust opacity or create gradient basedd on velocity in shader
    float advect(float u_vec, float dt, float q);
    float project(float dt, float u_vec); // make u divergence-free, enforce solid-wall boundary
    float interpolate(float dt, float u_vec);
};

#endif