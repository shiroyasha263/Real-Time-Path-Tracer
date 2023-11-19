OPTIX_add_sample_executable( optixPhotonBeam target_name
  ${embedded_ptx_code}
  default.frag
  default.vert
  devicePrograms.cu
  PhotonTracer.cu
  EBO.h
  VBO.h
  VAO.h
  Camera.h
  shaderClass.h
  CUDABuffer.h
  LaunchParams.h
  SampleRenderer.h
  PhotonTracer.h
  DisplayWindow.h
  EBO.cpp
  VBO.cpp
  VAO.cpp
  Camera.cpp
  shaderClass.cpp
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
