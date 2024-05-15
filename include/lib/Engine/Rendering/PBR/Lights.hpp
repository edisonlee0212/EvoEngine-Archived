#pragma once
#include "GraphicsResources.hpp"
#include "IPrivateComponent.hpp"

namespace EvoEngine
{
#pragma region Lights
    struct DirectionalLightInfo
    {
        glm::vec4 m_direction;
        glm::vec4 m_diffuse;
        glm::vec4 m_specular;
        glm::mat4 m_lightSpaceMatrix[4];
        glm::vec4 m_lightFrustumWidth;
        glm::vec4 m_lightFrustumDistance;
        glm::vec4 m_reservedParameters;
        glm::ivec4 m_viewPort;
    };
    class DirectionalLight : public IPrivateComponent
    {
    public:
        bool m_castShadow = true;
        glm::vec3 m_diffuse = glm::vec3(1.0f);
        float m_diffuseBrightness = 0.8f;
        float m_bias = 0.1f;
        float m_normalOffset = 0.001f;
        float m_lightSize = 0.01f;
        void OnCreate() override;
        bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
        void Serialize(YAML::Emitter& out) const override;
        void Deserialize(const YAML::Node& in) override;
        void PostCloneAction(const std::shared_ptr<IPrivateComponent>& target) override;
    };
    struct PointLightInfo
    {
        glm::vec4 m_position;
        glm::vec4 m_constantLinearQuadFarPlane;
        glm::vec4 m_diffuse;
        glm::vec4 m_specular;
        glm::mat4 m_lightSpaceMatrix[6];
        glm::vec4 m_reservedParameters;
        glm::ivec4 m_viewPort;
    };

    class PointLight : public IPrivateComponent
    {
    public:
        bool m_castShadow = true;
        float m_constant = 1.0f;
        float m_linear = 0.07f;
        float m_quadratic = 0.0015f;
        float m_bias = 0.05f;
        glm::vec3 m_diffuse = glm::vec3(1.0f);
        float m_diffuseBrightness = 0.8f;
        float m_lightSize = 0.01f;
        bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
        void OnCreate() override;
        void Serialize(YAML::Emitter& out) const override;
        void Deserialize(const YAML::Node& in) override;
        [[nodiscard]] float GetFarPlane() const;
        void PostCloneAction(const std::shared_ptr<IPrivateComponent>& target) override;
    };
    struct SpotLightInfo
    {
        glm::vec4 m_position;
        glm::vec4 m_direction;
        glm::mat4 m_lightSpaceMatrix;
        glm::vec4 m_cutOffOuterCutOffLightSizeBias;
        glm::vec4 m_constantLinearQuadFarPlane;
        glm::vec4 m_diffuse;
        glm::vec4 m_specular;
        glm::ivec4 m_viewPort;
    };
    class SpotLight : public IPrivateComponent
    {
    public:
        bool m_castShadow = true;
        float m_innerDegrees = 20;
        float m_outerDegrees = 30;
        float m_constant = 1.0f;
        float m_linear = 0.07f;
        float m_quadratic = 0.0015f;
        float m_bias = 0.001f;
        glm::vec3 m_diffuse = glm::vec3(1.0f);
        float m_diffuseBrightness = 0.8f;
        float m_lightSize = 0.01f;
        bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
        void OnCreate() override;
        void Serialize(YAML::Emitter& out) const override;
        void Deserialize(const YAML::Node& in) override;
        [[nodiscard]] float GetFarPlane() const;

        void PostCloneAction(const std::shared_ptr<IPrivateComponent>& target) override;
    };

    class Lighting
    {
        std::shared_ptr<Image> m_directionalLightShadowMap = {};
        std::shared_ptr<ImageView> m_directionalLightShadowMapView = {};
        std::vector<std::shared_ptr<ImageView>> m_directionalLightShadowMapLayeredViews = {};
        std::shared_ptr<Sampler> m_directionalShadowMapSampler = {};

        std::shared_ptr<Image> m_pointLightShadowMap = {};
        std::shared_ptr<ImageView> m_pointLightShadowMapView = {};
        std::vector<std::shared_ptr<ImageView>> m_pointLightShadowMapLayeredViews = {};
        std::shared_ptr<Sampler> m_pointLightShadowMapSampler = {};

        std::shared_ptr<Image> m_spotLightShadowMap = {};
        std::shared_ptr<ImageView> m_spotLightShadowMapView = {};
        std::shared_ptr<Sampler> m_spotLightShadowMapSampler = {};
        friend class RenderLayer;

        

        static void Consume(glm::vec2 location, uint32_t resolution, uint32_t remainingSize, std::vector<glm::uvec3>& results);

    public:
        std::shared_ptr<DescriptorSet> m_lightingDescriptorSet = VK_NULL_HANDLE;

        static void AllocateAtlas(uint32_t size, uint32_t maxResolution, std::vector<glm::uvec3>& results);

        Lighting();
        void Initialize();
        [[nodiscard]] VkRenderingAttachmentInfo GetDirectionalLightDepthAttachmentInfo(VkAttachmentLoadOp loadOp, VkAttachmentStoreOp storeOp) const;
        [[nodiscard]] VkRenderingAttachmentInfo GetPointLightDepthAttachmentInfo(VkAttachmentLoadOp loadOp, VkAttachmentStoreOp storeOp) const;
        [[nodiscard]] VkRenderingAttachmentInfo GetLayeredDirectionalLightDepthAttachmentInfo(uint32_t split, VkAttachmentLoadOp loadOp, VkAttachmentStoreOp storeOp) const;
        [[nodiscard]] VkRenderingAttachmentInfo GetLayeredPointLightDepthAttachmentInfo(uint32_t face, VkAttachmentLoadOp loadOp, VkAttachmentStoreOp storeOp) const;
        [[nodiscard]] VkRenderingAttachmentInfo GetSpotLightDepthAttachmentInfo(VkAttachmentLoadOp loadOp, VkAttachmentStoreOp storeOp) const;
    };
}
