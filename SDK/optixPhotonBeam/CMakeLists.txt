OPTIX_add_sample_executable( optixPhotonBeam target_name
  ${embedded_ptx_code}
  devicePrograms.cu
  PhotonTracer.cu
  CUDABuffer.h
  LaunchParams.h
  SampleRenderer.h
  PhotonTracer.h
  DisplayWindow.h
  SampleRenderer.cpp
  PhotonTracer.cpp
  DisplayWindow.cpp
  main.cpp

  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  )
