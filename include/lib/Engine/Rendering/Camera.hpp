#pragma once
#include "Bound.hpp"
#include "IPrivateComponent.hpp"
#include "Texture2D.hpp"
#include "RenderTexture.hpp"
#include "Transform.hpp"

namespace EvoEngine
{
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
        std::unique_ptr<Framebuffer> m_framebuffer;
        
        //Deferred shading GBuffer
        std::unique_ptr<Image> m_gBufferDepth = {};
        std::unique_ptr<ImageView> m_gBufferDepthView = {};
        ImTextureID m_gBufferDepthImTextureId = {};

        std::unique_ptr<Image> m_gBufferNormal = {};
        std::unique_ptr<ImageView> m_gBufferNormalView = {};
        ImTextureID m_gBufferNormalImTextureId = {};

        std::unique_ptr<Image> m_gBufferAlbedo = {};
        std::unique_ptr<ImageView> m_gBufferAlbedoView = {};
        ImTextureID m_gBufferAlbedoImTextureId = {};

        std::unique_ptr<Image> m_gBufferMaterial = {};
        std::unique_ptr<ImageView> m_gBufferMaterialView = {};
        ImTextureID m_gBufferMaterialImTextureId = {};

        size_t m_frameCount = 0;
        bool m_rendered = false;
        bool m_requireRendering = false;

        glm::uvec2 m_size = glm::uvec2(1, 1);

        void UpdateGBuffer();
        void UpdateFrameBuffer();
    public:
        static const std::vector<VkAttachmentDescription>& GetAttachmentDescriptions();

        [[nodiscard]] float GetSizeRatio() const;

        [[nodiscard]] std::shared_ptr<RenderTexture> GetRenderTexture() const;
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
        float m_backgroundIntensity = 1.0f;
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
