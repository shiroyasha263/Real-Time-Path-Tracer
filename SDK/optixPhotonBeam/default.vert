#version 330 core

// Start
layout (location = 0) in vec3 beamStart;
// PosMul
layout (location = 1) in float PosMul;
// pos / neg
layout (location = 2) in float dirMult;
// beamDir
layout (location = 3) in vec3 beamDir;
// Thickness
layout (location = 4) in float thickness;
// Transmittance
layout (location = 5) in float aTransmittance;

// Outputs the color for the Fragment Shader
out float transmittance;
// Outputs the current position for the Fragment Shader
out vec3 crntPos;

out float depth;

out float width;

out float sine;

out float thick;

// Imports the camera matrix from the main function
uniform mat4 camMatrix;
// Imports the model matrix from the main function
uniform mat4 model;
// Gets the position of the camera from the main function
uniform vec3 camPos;


void main()
{
	// calculates current position
	vec3 aPos = beamStart + PosMul * beamDir + dirMult * thickness * normalize(cross(beamStart - camPos, beamDir)) / 2.f;
	vec3 alongBeam = aPos - beamStart;
	vec3 nBeamDir = normalize(beamDir);
	vec3 projection = dot(alongBeam, nBeamDir) * nBeamDir;
	vec3 w = normalize(aPos - camPos);
	sine = length(cross(w, nBeamDir));

	width = 1.f * length(alongBeam - projection);
	depth = length(aPos - camPos);
	thick = thickness;

	crntPos = vec3(model * vec4(aPos, 1.0f));
	// Outputs the positions/coordinates of all vertices
	gl_Position = camMatrix * vec4(crntPos, 1.0);

	// Assigns the colors from the Vertex Data to "color"
	transmittance = aTransmittance;
}