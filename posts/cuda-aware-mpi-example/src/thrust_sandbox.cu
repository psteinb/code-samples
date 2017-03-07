#ifndef THRUST_SANDBOX_H
#define THRUST_SANDBOX_H

#include "Jacobi.h"
#include "thrust/device_vector.h"
#include <thrust/device_ptr.h>

struct my_kernel{

    __host__ __device__ real operator()(const real& _old, const real& _new){
        return _old + _new;
    }

};

extern "C" real CallJacobiKernel(real * devBlocks[2], real * devResidue, const int4 * bounds, const int2 * size)
{
    real residue = 0.;

    const std::size_t len = size->x * size->y;

    thrust::device_vector<real> d_old(thrust::device_pointer_cast(devBlocks[0]),
                                      thrust::device_pointer_cast(devBlocks[0])+len);
    thrust::device_vector<real> d_new(thrust::device_pointer_cast(devBlocks[1]),
                                      thrust::device_pointer_cast(devBlocks[1])+len);
    auto kernel = [=](const real& _old, const real& _new){
        return _old + _new;
    };

    thrust::transform( d_old.begin(), d_old.end(),
                       d_new.begin(), d_new.begin(),
                       my_kernel()
        );

    thrust::copy(devResidue,devResidue+1,&residue);
    return residue;
}

#endif /* THRUST_SANDBOX_H */
