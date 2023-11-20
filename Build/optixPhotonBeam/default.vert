#version 330 core

// Positions/Coordinates
layout (location = 0) in vec3 beamSide;
// pos / neg
layout (location = 1) in float dirMult;
// beamDir
layout (location = 2) in vec3 beamDir;
// Thickness
layout (location = 3) in float thickness;
// Transmittance
layout (location = 4) in float aTransmittance;

// Outputs the color for the Fragment Shader
out float transmittance;
// Outputs the current position for the Fragment Shader
out vec3 crntPos;

// Imports the camera matrix from the main function
uniform mat4 camMatrix;
// Imports the model matrix from the main function
uniform mat4 model;
// Gets the position of the camera from the main function
uniform vec3 camPos;


void main()
{
	// calculates current position
	vec3 aPos = beamSide + dirMult * thickness * normalize(cross(beamSide - camPos, beamDir)) / 2.f;
	crntPos = vec3(model * vec4(aPos, 1.0f));
	// Outputs the positions/coordinates of all vertices
	gl_Position = camMatrix * vec4(crntPos, 1.0);

	// Assigns the colors from the Vertex Data to "color"
	transmittance = aTransmittance;
}