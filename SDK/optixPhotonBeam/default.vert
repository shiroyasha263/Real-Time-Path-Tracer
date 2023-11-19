#version 330 core

// Positions/Coordinates
layout (location = 0) in vec3 aPos;
// Transmittance
layout (location = 1) in float aTransmittance;


// Outputs the color for the Fragment Shader
out float transmittance;
// Outputs the current position for the Fragment Shader
out vec3 crntPos;

// Imports the camera matrix from the main function
uniform mat4 camMatrix;
// Imports the model matrix from the main function
uniform mat4 model;


void main()
{
	// calculates current position
	crntPos = vec3(model * vec4(aPos, 1.0f));
	// Outputs the positions/coordinates of all vertices
	gl_Position = camMatrix * vec4(crntPos, 1.0);

	// Assigns the colors from the Vertex Data to "color"
	transmittance = aTransmittance;
}