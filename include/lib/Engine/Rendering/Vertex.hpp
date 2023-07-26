#pragma once

namespace EvoEngine
{
    enum class VertexAttribute
    {
        Position = 1,
        Normal = 1 << 1, // 2
        Tangent = 1 << 2,
        TexCoord = 1 << 3, // 8
        Color = 1 << 4,    // 16
    };

    struct Vertex
    {
        glm::vec3 m_position = glm::vec3(0.0f);
        glm::vec3 m_normal = glm::vec3(0.0f);
        glm::vec3 m_tangent = glm::vec3(0.0f);
        glm::vec2 m_texCoord = glm::vec2(0.0f);
		glm::vec4 m_color = glm::vec4(1.0f);
    };

    struct SkinnedVertex
    {
        glm::vec3 m_position = glm::vec3(0.0f);
        glm::vec3 m_normal = glm::vec3(0.0f);
        glm::vec3 m_tangent = glm::vec3(0.0f);
        glm::vec2 m_texCoord = glm::vec2(0.0f);
    	glm::vec4 m_color = glm::vec4(1.0f);

        glm::ivec4 m_bondId;
        glm::vec4 m_weight;
        glm::ivec4 m_bondId2;
        glm::vec4 m_weight2;
    };
    enum class StrandPointAttribute
    {
        Position = 1,
        Thickness = 1 << 1, // 2
        Normal = 1 << 2,
        TexCoord = 1 << 3,    // 8
        Color = 1 << 4, // 16
    };
    struct StrandPoint
    {
        glm::vec3 m_position;
        float m_thickness = 0.1f;
        glm::vec3 m_normal;
        float m_texCoord = 0.0f;
        glm::vec4 m_color = glm::vec4(1.0f);
    };
}