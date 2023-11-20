#version 330 core

// Outputs colors in RGBA
out vec4 FragColor;


// Imports the color from the Vertex Shader
in float transmittance;
// Imports the current position from the Vertex Shader
in vec3 crntPos;

// Gets the color of the light from the main function
//uniform vec4 lightColor;
// Gets the position of the camera from the main function
uniform vec3 camPos;

void main()
{
//	// ambient lighting
//	float ambient = 0.20f;
//
//	// diffuse lighting
//	vec3 normal = normalize(Normal);
//	vec3 lightDirection = normalize(lightPos - crntPos);
//	float diffuse = max(dot(normal, lightDirection), 0.0f);
//
//	// specular lighting
//	float specularLight = 0.50f;
	vec3 viewDirection = normalize(camPos - crntPos);
//	vec3 reflectionDirection = reflect(-lightDirection, normal);
//	float specAmount = pow(max(dot(viewDirection, reflectionDirection), 0.0f), 8);
//	float specular = specAmount * specularLight;
	float something = crntPos.x + crntPos.y + crntPos.z;
	float t = transmittance;

	// outputs final color
	FragColor = vec4(0.8f, 0.3f, 0.02f, 1.f);
}