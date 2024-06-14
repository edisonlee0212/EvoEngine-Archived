#pragma once

namespace evo_engine
{
    struct Vertex
    {
        glm::vec3 m_position = glm::vec3(0.0f);
        float m_vertexInfo1 = 0.0f;
    	glm::vec3 m_normal = glm::vec3(0.0f);
        float m_vertexInfo2 = 0.0f;
        glm::vec3 m_tangent = glm::vec3(0.0f);
        float m_vertexInfo3 = 0.0f;
        glm::vec4 m_color = glm::vec4(1.0f);
        glm::vec2 m_texCoord = glm::vec2(0.0f);
        glm::vec2 m_vertexInfo4 = glm::vec2(0.0f);
    };

    struct SkinnedVertex
    {
        glm::vec3 m_position = glm::vec3(0.0f);
        float m_vertexInfo1 = 0.0f;
        glm::vec3 m_normal = glm::vec3(0.0f);
        float m_vertexInfo2 = 0.0f;
        glm::vec3 m_tangent = glm::vec3(0.0f);
        float m_vertexInfo3 = 0.0f;
        glm::vec4 m_color = glm::vec4(1.0f);
        glm::vec2 m_texCoord = glm::vec2(0.0f);
        glm::vec2 m_vertexInfo4 = glm::vec2(0.0f);

        glm::ivec4 m_bondId = {};
        glm::vec4 m_weight = {};
        glm::ivec4 m_bondId2 = {};
        glm::vec4 m_weight2 = {};
    };
    
    struct StrandPoint
    {
        glm::vec3 m_position = glm::vec3(0.0f);
        float m_thickness = 0.0f;
        glm::vec3 m_normal = glm::vec3(0.0f);
        float m_texCoord = 0.0f;
        glm::vec4 m_color = glm::vec4(1.0f);
    };
}