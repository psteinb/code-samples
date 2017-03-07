#ifndef THRUST_SANDBOX_H
#define THRUST_SANDBOX_H

#include "Jacobi.h"

#include "thrust/transform_reduce.h"
#include "thrust/inner_product.h"

#include "thrust/functional.h"

#include "thrust/device_vector.h"
#include "thrust/device_ptr.h"

namespace detail {

// The default implementation for atomic maximum
    template <typename T>
    __device__ void AtomicMax(T * const address, const T value)
    {
        atomicMax(address, value);
    }

/**
 * @brief Compute the maximum of 2 single-precision floating point values using an atomic operation
 *
 * @param[in]	address	The address of the reference value which might get updated with the maximum
 * @param[in]	value	The value that is compared to the reference in order to determine the maximum
 */
    template <>
    __device__ void AtomicMax(float * const address, const float value)
    {
        if (* address >= value)
        {
            return;
        }

        int * const address_as_i = (int *)address;
        int old = * address_as_i, assumed;

        do
        {
            assumed = old;
            if (__int_as_float(assumed) >= value)
            {
                break;
            }

            old = atomicCAS(address_as_i, assumed, __float_as_int(value));
        } while (assumed != old);
    }

/**
 * @brief Compute the maximum of 2 double-precision floating point values using an atomic operation
 *
 * @param[in]	address	The address of the reference value which might get updated with the maximum
 * @param[in]	value	The value that is compared to the reference in order to determine the maximum
 */
    template <>
    __device__ void AtomicMax(double * const address, const double value)
    {
        if (* address >= value)
        {
            return;
        }

        uint64 * const address_as_i = (uint64 *)address;
        uint64 old = * address_as_i, assumed;

        do
        {
            assumed = old;
            if (__longlong_as_double(assumed) >= value)
            {
                break;
            }

            old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
        } while (assumed != old);
    }

// Templates for computing the absolute value of a real number
    template<typename T> 	__device__ T 		rabs(T val) 		{ return abs(val);   }
    template<> 				__device__ float 	rabs(float val) 	{ return fabsf(val); }
    template<> 				__device__ double 	rabs(double val) 	{ return fabs(val);  }

}

template <typename T>
struct jacobi_functor
{
    int2 shape;
    int4 bounds;
    thrust::device_ptr<T> newBlock,oldBlock;
    // T* global_residue   ;

    __host__ __device__
    void operator()(std::size_t linear_idx)
        {
            int2 idx = make_int2(linear_idx / shape.x, linear_idx % shape.x);
            int memIdx = (idx.y + 1) * (shape.x + 2) + idx.x + 1;

            if ((idx.x < bounds.x) || (idx.x > bounds.z) || (idx.y < bounds.y) || (idx.y > bounds.w))
            {
                return;
            }

            real newVal = ((real)0.25) * (oldBlock[memIdx - 1] + oldBlock[memIdx + 1] +
                                          oldBlock[memIdx - shape.x - 2] + oldBlock[memIdx + shape.x + 2]);

            newBlock[memIdx] = newVal;

            // detail::AtomicMax<T>(global_residue, detail::rabs(newVal - oldBlock[memIdx]));
        };
};

template <typename T>
struct abs_diff : public thrust::binary_function<T,T,T>
{
    __host__ __device__
    T operator()(const T& a, const T& b)
    {
        return detail::rabs(b - a);
    }
};

extern "C" real CallJacobiKernel(real * devBlocks[2], real * devResidue, const int4 * bounds, const int2 * size)
{

    const std::size_t len = size->x * size->y;

    thrust::counting_iterator<std::size_t> begin{0};
    thrust::counting_iterator<std::size_t> end{len};
    auto kernel = jacobi_functor<real>{*size,*bounds,
                                       thrust::device_pointer_cast(devBlocks[1]),
                                       thrust::device_pointer_cast(devBlocks[0])// ,
                                       // thrust::raw_pointer_cast(d_res.data())
    };


    thrust::for_each(begin,end,
                     kernel);

    thrust::maximum<real> binary_op1;
    abs_diff<real>        binary_op2;

    thrust::device_vector<real> oldBlock(devBlocks[1], devBlocks[1]+len);
    thrust::device_vector<real> newBlock(devBlocks[0], devBlocks[0]+len);
    real residue = thrust::inner_product(oldBlock.begin(), oldBlock.end(), newBlock.begin(), real(0.), binary_op1, binary_op2);
    return residue;
}

#endif /* THRUST_SANDBOX_H */
