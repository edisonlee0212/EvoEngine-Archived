#pragma once
#include "Bound.hpp"
#include "IPrivateComponent.hpp"
#include "Texture2D.hpp"
#include "RenderTexture.hpp"
#include "Transform.hpp"

namespace EvoEngine
{
    struct CameraInfoBlock
    {
        glm::mat4 m_projection = {};
        glm::mat4 m_view = {};
        glm::mat4 m_projectionView = {};
        glm::mat4 m_inverseProjection = {};
        glm::mat4 m_inverseView = {};
        glm::mat4 m_inverseProjectionView = {};
        glm::vec4 m_clearColor = {};
        glm::vec4 m_reservedParameters1 = {};
        glm::vec4 m_reservedParameters2 = {};

        [[nodiscard]] glm::vec3 Project(const glm::vec3& position) const;
        [[nodiscard]] glm::vec3 UnProject(const glm::vec3& position) const;
    };

    class Camera final : public IPrivateComponent
    {
        friend class Graphics;
        friend class RenderLayer;
        friend class EditorLayer;
        friend struct CameraInfoBlock;
        friend class PostProcessing;
        friend class Bloom;
        friend class SSAO;
        friend class SSR;
        
        std::shared_ptr<RenderTexture> m_renderTexture;
        
        //Deferred shading GBuffer
        std::shared_ptr<Image> m_gBufferDepth = {};
        std::shared_ptr<ImageView> m_gBufferDepthView = {};
        std::shared_ptr<Sampler> m_gBufferDepthSampler = {};
        ImTextureID m_gBufferDepthImTextureId = {};

        std::shared_ptr<Image> m_gBufferNormal = {};
        std::shared_ptr<ImageView> m_gBufferNormalView = {};
        std::shared_ptr<Sampler> m_gBufferNormalSampler = {};
        ImTextureID m_gBufferNormalImTextureId = {};

        std::shared_ptr<Image> m_gBufferAlbedo = {};
        std::shared_ptr<ImageView> m_gBufferAlbedoView = {};
        std::shared_ptr<Sampler> m_gBufferAlbedoSampler = {};
        ImTextureID m_gBufferAlbedoImTextureId = {};

        std::shared_ptr<Image> m_gBufferMaterial = {};
        std::shared_ptr<ImageView> m_gBufferMaterialView = {};
        std::shared_ptr<Sampler> m_gBufferMaterialSampler = {};
        ImTextureID m_gBufferMaterialImTextureId = {};

        size_t m_frameCount = 0;
        bool m_rendered = false;
        bool m_requireRendering = false;

        glm::uvec2 m_size = glm::uvec2(1, 1);

        std::shared_ptr<DescriptorSet> m_gBufferDescriptorSet = VK_NULL_HANDLE;
        void UpdateGBuffer();
    public:
        void TransitGBufferImageLayout(VkCommandBuffer commandBuffer, VkImageLayout targetLayout) const;

        void UpdateCameraInfoBlock(CameraInfoBlock& cameraInfoBlock, const GlobalTransform& globalTransform) const;
        void AppendGBufferColorAttachmentInfos(std::vector<VkRenderingAttachmentInfo>& attachmentInfos) const;
        [[nodiscard]] VkRenderingAttachmentInfo GetDepthAttachmentInfo() const;
        [[nodiscard]] float GetSizeRatio() const;

        [[nodiscard]] const std::shared_ptr<RenderTexture> &GetRenderTexture() const;
        [[nodiscard]] glm::vec2 GetSize() const;
        void Resize(const glm::uvec2& size);
        void OnCreate() override;
        [[nodiscard]] bool Rendered() const;
        void SetRequireRendering(bool value);
        Camera& operator=(const Camera& source);
        float m_nearDistance = 0.1f;
        float m_farDistance = 500.0f;
        float m_fov = 120;
        bool m_useClearColor = false;
        glm::vec3 m_clearColor = glm::vec3(0.0f);
        AssetRef m_skybox;

        static void CalculatePlanes(std::vector<Plane>& planes, const glm::mat4& projection, const glm::mat4& view);
        static void CalculateFrustumPoints(
            const std::shared_ptr<Camera>& cameraComponent,
            float nearPlane,
            float farPlane,
            glm::vec3 cameraPos,
            glm::quat cameraRot,
            glm::vec3* points);
        static glm::quat ProcessMouseMovement(float yawAngle, float pitchAngle, bool constrainPitch = true);
        static void ReverseAngle(
            const glm::quat& rotation, float& pitchAngle, float& yawAngle, const bool& constrainPitch = true);
       [[nodiscard]] glm::mat4 GetProjection() const;

        glm::vec3 GetMouseWorldPoint(GlobalTransform& ltw, glm::vec2 mousePosition) const;
        Ray ScreenPointToRay(GlobalTransform& ltw, glm::vec2 mousePosition) const;

        void Serialize(YAML::Emitter& out) override;
        void Deserialize(const YAML::Node& in) override;
        void OnDestroy() override;
        
        void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
        void CollectAssetRef(std::vector<AssetRef>& list) override;

    };
}
