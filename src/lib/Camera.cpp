#include "Camera.hpp"

#include "Application.hpp"
#include "ClassRegistry.hpp"
#include "EditorLayer.hpp"
#include "Serialization.hpp"
#include "Utilities.hpp"
#include "RenderLayer.hpp"
#include "Scene.hpp"
using namespace EvoEngine;

void Camera::UpdateGBuffer()
{
	m_gBufferDepthView.reset();
	m_gBufferNormalView.reset();
	m_gBufferAlbedoView.reset();
	m_gBufferMaterialView.reset();

	m_gBufferDepth.reset();
	m_gBufferNormal.reset();
	m_gBufferAlbedo.reset();
	m_gBufferMaterial.reset();


	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent = m_renderTexture->GetExtent();
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = Graphics::ImageFormats::m_gBufferDepth;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		m_gBufferDepth = std::make_unique<Image>(imageInfo);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = m_gBufferDepth->GetVkImage();
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = Graphics::ImageFormats::m_gBufferDepth;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		m_gBufferDepthView = std::make_unique<ImageView>(viewInfo);
	}
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent = m_renderTexture->GetExtent();
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = Graphics::ImageFormats::m_gBufferColor;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		m_gBufferNormal = std::make_unique<Image>(imageInfo);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = m_gBufferNormal->GetVkImage();
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = Graphics::ImageFormats::m_gBufferColor;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		m_gBufferNormalView = std::make_unique<ImageView>(viewInfo);
	}
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent = m_renderTexture->GetExtent();
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = Graphics::ImageFormats::m_gBufferColor;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		m_gBufferAlbedo = std::make_unique<Image>(imageInfo);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = m_gBufferAlbedo->GetVkImage();
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = Graphics::ImageFormats::m_gBufferColor;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		m_gBufferAlbedoView = std::make_unique<ImageView>(viewInfo);
	}
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent = m_renderTexture->GetExtent();
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = Graphics::ImageFormats::m_gBufferColor;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		m_gBufferMaterial = std::make_unique<Image>(imageInfo);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = m_gBufferMaterial->GetVkImage();
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = Graphics::ImageFormats::m_gBufferColor;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		m_gBufferMaterialView = std::make_unique<ImageView>(viewInfo);
	}
	if (const auto editorLayer = Application::GetLayer<EditorLayer>()) {
		//m_gBufferDepthImTextureId = editorLayer->GetTextureId(m_gBufferDepthView->GetVkImageView(), m_gBufferDepth->GetLayout());
		//m_gBufferNormalImTextureId = editorLayer->GetTextureId(m_gBufferNormalView->GetVkImageView(), m_gBufferNormal->GetLayout());
		m_gBufferAlbedo->TransitionImageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		m_gBufferAlbedoImTextureId = editorLayer->GetTextureId(m_gBufferAlbedoView->GetVkImageView(), m_gBufferAlbedo->GetLayout());
		//m_gBufferMaterialImTextureId = editorLayer->GetTextureId(m_gBufferMaterialView->GetVkImageView(), m_gBufferMaterial->GetLayout());
	}
}

void Camera::UpdateFrameBuffer()
{
	const auto renderLayer = Application::GetLayer<RenderLayer>();
	if (renderLayer) {
		const std::vector attachments = {
			m_gBufferDepthView->GetVkImageView(),
			m_gBufferNormalView->GetVkImageView(),
			m_gBufferAlbedoView->GetVkImageView(),
			m_gBufferMaterialView->GetVkImageView(),
			
			m_renderTexture->GetDepthImageView()->GetVkImageView(),
			m_renderTexture->GetColorImageView()->GetVkImageView()
		};

		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = renderLayer->GetRenderPass("DEFERRED_RENDERING")->GetVkRenderPass();
		framebufferInfo.attachmentCount = attachments.size();
		framebufferInfo.pAttachments = attachments.data();
		framebufferInfo.width = m_size.x;
		framebufferInfo.height = m_size.y;
		framebufferInfo.layers = 1;

		m_deferredRenderingFramebuffer.reset();
		m_deferredRenderingFramebuffer = std::make_unique<Framebuffer>(framebufferInfo);
	}
}

void Camera::UpdateCameraInfoBlock(CameraInfoBlock& cameraInfoBlock, const GlobalTransform& globalTransform) const
{
	const auto rotation = globalTransform.GetRotation();
	const auto position = globalTransform.GetPosition();
	const glm::vec3 front = rotation * glm::vec3(0, 0, -1);
	const glm::vec3 up = rotation * glm::vec3(0, 1, 0);
	const auto ratio = GetSizeRatio();
	cameraInfoBlock.m_projection =
		glm::perspective(glm::radians(m_fov * 0.5f), ratio, m_nearDistance, m_farDistance);
	cameraInfoBlock.m_view = glm::lookAt(position, position + front, up);
	cameraInfoBlock.m_projectionView = cameraInfoBlock.m_projection * cameraInfoBlock.m_view;
	cameraInfoBlock.m_inverseProjection = glm::inverse(cameraInfoBlock.m_projection);
	cameraInfoBlock.m_inverseView = glm::inverse(cameraInfoBlock.m_view);
	cameraInfoBlock.m_inverseProjectionView = glm::inverse(cameraInfoBlock.m_projection) * glm::inverse(cameraInfoBlock.m_view);
	cameraInfoBlock.m_reservedParameters1 = glm::vec4(
		m_nearDistance,
		m_farDistance,
		glm::tan(glm::radians(m_fov * 0.5f)),
		glm::tan(glm::radians(m_fov * 0.25f)));
	cameraInfoBlock.m_clearColor = glm::vec4(m_clearColor, 1.0f);
	cameraInfoBlock.m_reservedParameters2 = glm::vec4(m_size.x, m_size.y, static_cast<float>(m_size.x) / m_size.y, 0.0f);
	if (m_useClearColor)
	{
		cameraInfoBlock.m_clearColor.w = 1.0f;
	}
	else
	{
		cameraInfoBlock.m_clearColor.w = 0.0f;
	}
}

const std::unique_ptr<Framebuffer>& Camera::GetFramebuffer() const
{
	return m_deferredRenderingFramebuffer;
}

const std::vector<VkAttachmentDescription>& Camera::GetAttachmentDescriptions()
{
	static std::vector<VkAttachmentDescription> attachments{};

	if (attachments.empty()) {
		VkAttachmentDescription attachment{};

		attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		attachment.format = Graphics::ImageFormats::m_gBufferDepth;
		attachments.push_back(attachment);
		attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		attachment.format = Graphics::ImageFormats::m_gBufferColor;
		attachments.push_back(attachment);
		attachment.format = Graphics::ImageFormats::m_gBufferColor;
		attachments.push_back(attachment);
		attachment.format = Graphics::ImageFormats::m_gBufferColor;
		attachments.push_back(attachment);

		auto renderTextureAttachments = RenderTexture::GetAttachmentDescriptions();
		attachments.insert(attachments.end(), renderTextureAttachments.begin(), renderTextureAttachments.end());
	}
	return attachments;
}

float Camera::GetSizeRatio() const
{
	if (m_size.x == 0 || m_size.y == 0)
		return 0;
	return static_cast<float>(m_size.x) / static_cast<float>(m_size.y);
}

std::shared_ptr<RenderTexture> Camera::GetRenderTexture() const
{
	return m_renderTexture;
}

glm::vec2 Camera::GetSize() const
{
	return m_size;
}

void Camera::Resize(const glm::uvec2& size)
{
	if (size.x == 0 || size.y == 0) return;
	if (size.x > 16384 || size.y >= 16384) return;
	if (m_size == size) return;
	m_size = size;
	m_renderTexture->Resize({ m_size.x, m_size.y, 1 });
	UpdateGBuffer();
	UpdateFrameBuffer();
}

void Camera::OnCreate()
{
	m_size = glm::uvec2(1, 1);
	VkExtent3D extent;
	extent.width = m_size.x;
	extent.height = m_size.y;
	extent.depth = 1;
	m_renderTexture = std::make_unique<RenderTexture>(extent);
	UpdateGBuffer();
	UpdateFrameBuffer();
}

bool Camera::Rendered() const
{
	return m_rendered;
}
void Camera::SetRequireRendering(const bool value)
{
	m_requireRendering = m_requireRendering || value;
}

Camera& Camera::operator=(const Camera& source)
{
	if (this == &source) return *this;
	m_nearDistance = source.m_nearDistance;
	m_farDistance = source.m_farDistance;
	m_fov = source.m_fov;
	m_useClearColor = source.m_useClearColor;
	m_clearColor = source.m_clearColor;
	m_skybox = source.m_skybox;
	return *this;
}
void Camera::CalculatePlanes(std::vector<Plane>& planes, const glm::mat4& projection, const glm::mat4& view)
{
	glm::mat4 comboMatrix = projection * glm::transpose(view);
	planes[0].m_a = comboMatrix[3][0] + comboMatrix[0][0];
	planes[0].m_b = comboMatrix[3][1] + comboMatrix[0][1];
	planes[0].m_c = comboMatrix[3][2] + comboMatrix[0][2];
	planes[0].m_d = comboMatrix[3][3] + comboMatrix[0][3];

	planes[1].m_a = comboMatrix[3][0] - comboMatrix[0][0];
	planes[1].m_b = comboMatrix[3][1] - comboMatrix[0][1];
	planes[1].m_c = comboMatrix[3][2] - comboMatrix[0][2];
	planes[1].m_d = comboMatrix[3][3] - comboMatrix[0][3];

	planes[2].m_a = comboMatrix[3][0] - comboMatrix[1][0];
	planes[2].m_b = comboMatrix[3][1] - comboMatrix[1][1];
	planes[2].m_c = comboMatrix[3][2] - comboMatrix[1][2];
	planes[2].m_d = comboMatrix[3][3] - comboMatrix[1][3];

	planes[3].m_a = comboMatrix[3][0] + comboMatrix[1][0];
	planes[3].m_b = comboMatrix[3][1] + comboMatrix[1][1];
	planes[3].m_c = comboMatrix[3][2] + comboMatrix[1][2];
	planes[3].m_d = comboMatrix[3][3] + comboMatrix[1][3];

	planes[4].m_a = comboMatrix[3][0] + comboMatrix[2][0];
	planes[4].m_b = comboMatrix[3][1] + comboMatrix[2][1];
	planes[4].m_c = comboMatrix[3][2] + comboMatrix[2][2];
	planes[4].m_d = comboMatrix[3][3] + comboMatrix[2][3];

	planes[5].m_a = comboMatrix[3][0] - comboMatrix[2][0];
	planes[5].m_b = comboMatrix[3][1] - comboMatrix[2][1];
	planes[5].m_c = comboMatrix[3][2] - comboMatrix[2][2];
	planes[5].m_d = comboMatrix[3][3] - comboMatrix[2][3];

	planes[0].Normalize();
	planes[1].Normalize();
	planes[2].Normalize();
	planes[3].Normalize();
	planes[4].Normalize();
	planes[5].Normalize();
}

void Camera::CalculateFrustumPoints(
	const std::shared_ptr<Camera>& cameraComponent,
	float nearPlane,
	float farPlane,
	glm::vec3 cameraPos,
	glm::quat cameraRot,
	glm::vec3* points)
{
	const glm::vec3 front = cameraRot * glm::vec3(0, 0, -1);
	const glm::vec3 right = cameraRot * glm::vec3(1, 0, 0);
	const glm::vec3 up = cameraRot * glm::vec3(0, 1, 0);
	const glm::vec3 nearCenter = front * nearPlane;
	const glm::vec3 farCenter = front * farPlane;


	const float e = tanf(glm::radians(cameraComponent->m_fov * 0.5f));
	const float nearExtY = e * nearPlane;
	const float nearExtX = nearExtY * cameraComponent->GetSizeRatio();
	const float farExtY = e * farPlane;
	const float farExtX = farExtY * cameraComponent->GetSizeRatio();

	points[0] = cameraPos + nearCenter - right * nearExtX - up * nearExtY;
	points[1] = cameraPos + nearCenter - right * nearExtX + up * nearExtY;
	points[2] = cameraPos + nearCenter + right * nearExtX + up * nearExtY;
	points[3] = cameraPos + nearCenter + right * nearExtX - up * nearExtY;
	points[4] = cameraPos + farCenter - right * farExtX - up * farExtY;
	points[5] = cameraPos + farCenter - right * farExtX + up * farExtY;
	points[6] = cameraPos + farCenter + right * farExtX + up * farExtY;
	points[7] = cameraPos + farCenter + right * farExtX - up * farExtY;
}

glm::quat Camera::ProcessMouseMovement(float yawAngle, float pitchAngle, bool constrainPitch)
{
	// Make sure that when pitch is out of bounds, screen doesn't get flipped
	if (constrainPitch)
	{
		if (pitchAngle > 89.0f)
			pitchAngle = 89.0f;
		if (pitchAngle < -89.0f)
			pitchAngle = -89.0f;
	}

	glm::vec3 front;
	front.x = cos(glm::radians(yawAngle)) * cos(glm::radians(pitchAngle));
	front.y = sin(glm::radians(pitchAngle));
	front.z = sin(glm::radians(yawAngle)) * cos(glm::radians(pitchAngle));
	front = glm::normalize(front);
	const glm::vec3 right = glm::normalize(glm::cross(
		front, glm::vec3(0.0f, 1.0f, 0.0f))); // Normalize the vectors, because their length gets closer to 0 the more
	// you look up or down which results in slower movement.
	const glm::vec3 up = glm::normalize(glm::cross(right, front));
	return glm::quatLookAt(front, up);
}

void Camera::ReverseAngle(const glm::quat& rotation, float& pitchAngle, float& yawAngle, const bool& constrainPitch)
{
	const auto angle = glm::degrees(glm::eulerAngles(rotation));
	pitchAngle = angle.x;
	yawAngle = glm::abs(angle.z) > 90.0f ? 90.0f - angle.y : -90.0f - angle.y;
	if (constrainPitch)
	{
		if (pitchAngle > 89.0f)
			pitchAngle = 89.0f;
		if (pitchAngle < -89.0f)
			pitchAngle = -89.0f;
	}
}
glm::mat4 Camera::GetProjection() const
{
	return glm::perspective(glm::radians(m_fov * 0.5f), GetSizeRatio(), m_nearDistance, m_farDistance);
}


glm::vec3 Camera::GetMouseWorldPoint(GlobalTransform& ltw, glm::vec2 mousePosition) const
{
	const float halfX = static_cast<float>(m_size.x) / 2.0f;
	const float halfY = static_cast<float>(m_size.y) / 2.0f;
	const glm::vec4 start =
		glm::vec4(-1.0f * (mousePosition.x - halfX) / halfX, -1.0f * (mousePosition.y - halfY) / halfY, 0.0f, 1.0f);
	return start / start.w;
}

Ray Camera::ScreenPointToRay(GlobalTransform& ltw, glm::vec2 mousePosition) const
{
	const auto position = ltw.GetPosition();
	const auto rotation = ltw.GetRotation();
	const glm::vec3 front = rotation * glm::vec3(0, 0, -1);
	const glm::vec3 up = rotation * glm::vec3(0, 1, 0);
	const auto projection =
		glm::perspective(glm::radians(m_fov * 0.5f), GetSizeRatio(), m_nearDistance, m_farDistance);
	const auto view = glm::lookAt(position, position + front, up);
	const glm::mat4 inv = glm::inverse(projection * view);
	const float halfX = static_cast<float>(m_size.x) / 2.0f;
	const float halfY = static_cast<float>(m_size.y) / 2.0f;
	const auto realX = (mousePosition.x - halfX) / halfX;
	const auto realY = (mousePosition.y - halfY) / halfY;
	if (glm::abs(realX) > 1.0f || glm::abs(realY) > 1.0f)
		return { glm::vec3(FLT_MAX), glm::vec3(FLT_MAX) };
	glm::vec4 start = glm::vec4(realX, -1 * realY, -1, 1.0);
	glm::vec4 end = glm::vec4(realX, -1.0f * realY, 1.0f, 1.0f);
	start = inv * start;
	end = inv * end;
	start /= start.w;
	end /= end.w;
	const glm::vec3 dir = glm::normalize(glm::vec3(end - start));
	return { glm::vec3(ltw.m_value[3]) + m_nearDistance * dir, glm::vec3(ltw.m_value[3]) + m_farDistance * dir };
}

void Camera::Serialize(YAML::Emitter& out)
{
	out << YAML::Key << "m_resolutionX" << YAML::Value << m_size.x;
	out << YAML::Key << "m_resolutionY" << YAML::Value << m_size.y;
	out << YAML::Key << "m_useClearColor" << YAML::Value << m_useClearColor;
	out << YAML::Key << "m_clearColor" << YAML::Value << m_clearColor;
	out << YAML::Key << "m_nearDistance" << YAML::Value << m_nearDistance;
	out << YAML::Key << "m_farDistance" << YAML::Value << m_farDistance;
	out << YAML::Key << "m_fov" << YAML::Value << m_fov;

	out << YAML::Key << "m_backgroundIntensity" << YAML::Value << m_backgroundIntensity;
	m_skybox.Save("m_skybox", out);
}

void Camera::Deserialize(const YAML::Node& in)
{
	int resolutionX = in["m_resolutionX"].as<int>();
	int resolutionY = in["m_resolutionY"].as<int>();
	m_useClearColor = in["m_useClearColor"].as<bool>();
	m_clearColor = in["m_clearColor"].as<glm::vec3>();
	m_nearDistance = in["m_nearDistance"].as<float>();
	m_farDistance = in["m_farDistance"].as<float>();
	m_fov = in["m_fov"].as<float>();
	Resize({ resolutionX, resolutionY });
	if (in["m_backgroundIntensity"]) m_backgroundIntensity = in["m_backgroundIntensity"].as<float>();
	m_skybox.Load("m_skybox", in);
	m_rendered = false;
	m_requireRendering = false;
}

void Camera::OnDestroy()
{
	m_renderTexture.reset();
	m_skybox.Clear();

}

void Camera::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	if (!Application::GetLayer<RenderLayer>()->m_allowAutoResize)
	{
		glm::ivec2 resolution = { m_size.x, m_size.y };
		if (ImGui::DragInt2("Resolution", &resolution.x, 1, 1, 4096))
		{
			Resize({ resolution.x, resolution.y });
		}
	}
	ImGui::DragFloat("Background intensity", &m_backgroundIntensity, 0.1f, 0.0f, 1.0f);
	ImGui::Checkbox("Use clear color", &m_useClearColor);
	auto scene = GetScene();
	const bool savedState = (this == scene->m_mainCamera.Get<Camera>().get());
	bool isMainCamera = savedState;
	ImGui::Checkbox("Main Camera", &isMainCamera);
	if (savedState != isMainCamera)
	{
		if (isMainCamera)
		{
			scene->m_mainCamera = scene->GetOrSetPrivateComponent<Camera>(GetOwner()).lock();
		}
		else
		{
			Application::GetActiveScene()->m_mainCamera.Clear();
		}
	}
	if (m_useClearColor)
	{
		ImGui::ColorEdit3("Clear Color", (float*)(void*)&m_clearColor);
	}
	else
	{
		//Editor::DragAndDropButton<Cubemap>(m_skybox, "Skybox");
	}

	ImGui::DragFloat("Near", &m_nearDistance, m_nearDistance / 10.0f, 0, m_farDistance);
	ImGui::DragFloat("Far", &m_farDistance, m_farDistance / 10.0f, m_nearDistance);
	ImGui::DragFloat("FOV", &m_fov, 1.0f, 1, 359);

	FileUtils::SaveFile("Screenshot", "Texture2D", { ".png", ".jpg" }, [this](const std::filesystem::path& filePath) {
		//m_colorTexture->SetPathAndSave(filePath);
		});

	if (ImGui::TreeNode("Debug"))
	{
		m_requireRendering = true;
		static float debugSacle = 0.25f;
		ImGui::DragFloat("Scale", &debugSacle, 0.01f, 0.1f, 1.0f);
		debugSacle = glm::clamp(debugSacle, 0.1f, 1.0f);
		if (m_rendered)
		{
			ImGui::Image(m_renderTexture->GetColorImTextureId(),
				ImVec2(m_size.x * debugSacle, m_size.y * debugSacle),
				ImVec2(0, 1),
				ImVec2(1, 0));
			ImGui::Image(m_gBufferDepthImTextureId,
				ImVec2(m_size.x * debugSacle, m_size.y * debugSacle),
				ImVec2(0, 1),
				ImVec2(1, 0));
			ImGui::Image(m_gBufferNormalImTextureId,
				ImVec2(m_size.x * debugSacle, m_size.y * debugSacle),
				ImVec2(0, 1),
				ImVec2(1, 0));
			ImGui::Image(m_gBufferAlbedoImTextureId,
				ImVec2(m_size.x * debugSacle, m_size.y * debugSacle),
				ImVec2(0, 1),
				ImVec2(1, 0));
			ImGui::Image(m_gBufferMaterialImTextureId,
				ImVec2(m_size.x * debugSacle, m_size.y * debugSacle),
				ImVec2(0, 1),
				ImVec2(1, 0));
		}
		ImGui::TreePop();
	}
}

void Camera::CollectAssetRef(std::vector<AssetRef>& list)
{
	list.push_back(m_skybox);
}
