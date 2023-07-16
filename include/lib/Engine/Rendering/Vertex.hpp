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

        static VkVertexInputBindingDescription GetBindingDescription() {
            VkVertexInputBindingDescription bindingDescription;
            bindingDescription.binding = 0;
            bindingDescription.stride = sizeof(Vertex);
            bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            return bindingDescription;
        }

        static std::vector<VkVertexInputAttributeDescription> GetAttributeDescriptions() {
            std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
            attributeDescriptions.resize(4);

            attributeDescriptions[0].binding = 0;
            attributeDescriptions[0].location = 0;
            attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributeDescriptions[0].offset = offsetof(Vertex, m_position);

            attributeDescriptions[1].binding = 0;
            attributeDescriptions[1].location = 1;
            attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributeDescriptions[1].offset = offsetof(Vertex, m_normal);

            attributeDescriptions[2].binding = 0;
            attributeDescriptions[2].location = 2;
            attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributeDescriptions[2].offset = offsetof(Vertex, m_tangent);

            attributeDescriptions[3].binding = 0;
            attributeDescriptions[3].location = 3;
            attributeDescriptions[3].format = VK_FORMAT_R32G32_SFLOAT;
            attributeDescriptions[3].offset = offsetof(Vertex, m_texCoord);

            attributeDescriptions[4].binding = 0;
            attributeDescriptions[4].location = 4;
            attributeDescriptions[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
            attributeDescriptions[4].offset = offsetof(Vertex, m_color);

            return attributeDescriptions;
        }
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