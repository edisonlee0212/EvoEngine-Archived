layout (location = 0) out vec4 FragColor;

layout (location = 0) in VS_OUT {
    vec2 TexCoord;
} vs_in;

layout(set = 0, binding = 0) uniform sampler2D screenTexture;

void main()
{
    FragColor = texture(screenTexture, vs_in.TexCoord).rgba;
} 