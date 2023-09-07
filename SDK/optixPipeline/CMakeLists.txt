OPTIX_add_sample_executable( optixPipeline target_name
  ${embedded_ptx_code}
  devicePrograms.cu
  CUDABuffer.h
  LaunchParams.h
  SampleRenderer.h
  DisplayWindow.h
  SampleRenderer.cpp
  DisplayWindow.cpp
  main.cpp

  OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  )
