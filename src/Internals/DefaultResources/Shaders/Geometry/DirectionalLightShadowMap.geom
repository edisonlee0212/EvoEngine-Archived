layout (triangles) in;
layout (triangle_strip, max_vertices = 12) out;

layout(location = 0) out vec4 FragPos; // FragPos from GS (output per emitvertex)
void main()
{
    for(int split = 0; split < 4; ++split)
    {
        gl_Layer = split; // built-in variable that specifies to which face we render.
        for(int i = 0; i < 3; ++i) // for each triangle's vertices
        {
            FragPos = gl_in[i].gl_Position;
            gl_Position = EE_DIRECTIONAL_LIGHTS[EE_MATERIAL_INDEX].lightSpaceMatrix[split] * FragPos;
            EmitVertex();
        }
        EndPrimitive();
    }
} 