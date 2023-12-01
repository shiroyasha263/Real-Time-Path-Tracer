#version 330 core

// Start
layout (location = 0) in vec3 beamStart;
// PosMul
layout (location = 1) in float PosMul;
// pos / neg
layout (location = 2) in float dirMult;
// beamDir
layout (location = 3) in vec3 beamEnd;
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
	vec3 crntStart = vec3(model * vec4(beamStart, 1.f));
	vec3 crntEnd = vec3(model * vec4(beamEnd, 1.f));
	vec3 beamDir = crntEnd - crntStart;
	crntPos = crntStart + PosMul * beamDir + dirMult * thickness * normalize(cross(crntStart - camPos, beamDir)) / 2.f;

	vec3 alongBeam = crntPos - crntStart;
	vec3 nBeamDir = normalize(beamDir);
	vec3 projection = dot(alongBeam, nBeamDir) * nBeamDir;
	vec3 w = normalize(crntPos - camPos);
	sine = length(cross(w, nBeamDir));

	width = length(cross(alongBeam, nBeamDir));
	depth = length(crntPos - camPos);
	thick = thickness;

	// Outputs the positions/coordinates of all vertices
	gl_Position = camMatrix * vec4(crntPos, 1.0);

	// Assigns the colors from the Vertex Data to "color"
	transmittance = aTransmittance;
}