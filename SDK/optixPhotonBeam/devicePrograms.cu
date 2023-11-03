#include <optix_device.h>
#include <sutil/vec_math.h>
#include "LaunchParams.h"

extern "C" __constant__ LaunchParams optixLaunchParams;

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

static __forceinline__ __device__
void* unpackPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__
void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T* getPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__radiance()
{ /*! for this simple example, this will remain empty */
    
}

extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */
    const QuadSBTData& sbtData
        = *(const QuadSBTData*)optixGetSbtDataPointer();
    float transmittance = sbtData.transmittance;
    //
    //// compute normal:
    //const int   primID = optixGetPrimitiveIndex();
    //const int3 index = sbtData.index[primID];
    //const float3& A = sbtData.vertex[index.x];
    //const float3& B = sbtData.vertex[index.y];
    //const float3& C = sbtData.vertex[index.z];
    //const float3 Ng = normalize(cross(B - A, C - A));
    //
    //const float3 rayDir = optixGetWorldRayDirection();
    //const float cosDN = 0.2f + .8f * fabsf(dot(rayDir, Ng));
    float tmax = optixGetRayTmax();
    float3 rayDir = optixGetWorldRayDirection();
    float3 origin = optixGetWorldRayOrigin();
    float3 hitPoint = origin + rayDir * tmax;
    float disTransmittance = exp(-length(hitPoint - sbtData.start) * optixLaunchParams.mediumProp);
    float eyeTransmittance = exp(-tmax * optixLaunchParams.mediumProp);
    float3& prd = *(float3*)getPRD<float3>();
    prd += disTransmittance * transmittance * eyeTransmittance * make_float3(1.f, 1.f, 1.f) / 200.f;
    prd = clamp(prd, make_float3(0.f, 0.f, 0.f), make_float3(1.f, 1.f, 1.f));

    optixIgnoreIntersection();
}



//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance()
{ /*! for this simple example, this will remain empty */
    float3& prd = *(float3*)getPRD<float3>();
    // set to constant white as background color
    prd += make_float3(0.f, 0.f, 0.f);
}



//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame()
{
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const float3 position = make_float3(0, 0, -3);

    // our per-ray data for this example. what we initialize it to
    // won't matter, since this value will be overwritten by either
    // the miss or hit program, anyway
    float3 pixelColorPRD = make_float3(0.f, 0.f, 0.f);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer(&pixelColorPRD, u0, u1);

    // normalized screen plane position, in [0,1]^2
    const float2 screen(make_float2((ix + .5f) / optixLaunchParams.width, (iy + .5f) / optixLaunchParams.height));

    // generate ray direction
    float3 cameraDirection = make_float3(0, 0, 1);
    float3 rayDir = normalize(cameraDirection
        + (screen.x - 0.5f) * make_float3(1, 0, 0)
        + (screen.y - 0.5f) * make_float3(0, 1, 0));

    optixTrace(optixLaunchParams.handle,
        position,
        rayDir,
        0.f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,//OPTIX_RAY_FLAG_NONE,
        SURFACE_RAY_TYPE,             // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        SURFACE_RAY_TYPE,             // missSBTIndex 
        u0, u1);

    const int r = int(255.99f * pixelColorPRD.x);
    const int g = int(255.99f * pixelColorPRD.y);
    const int b = int(255.99f * pixelColorPRD.z);
    
    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000
        | (r << 0) | (g << 8) | (b << 16);
    
    // and write to frame buffer ...
    const uint32_t fbIndex = ix + iy * optixLaunchParams.width;
    optixLaunchParams.image[fbIndex] = make_uchar4(r, g, b, 255u);
}