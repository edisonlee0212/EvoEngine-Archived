#include "Camera.hpp"
#include "Cubemap.hpp"
#include "Application.hpp"
#include "ClassRegistry.hpp"
#include "EditorLayer.hpp"
#include "Serialization.hpp"
#include "Utilities.hpp"
#include "RenderLayer.hpp"
#include "Scene.hpp"
#include "PostProcessingStack.hpp"
using namespace EvoEngine;


glm::vec3 CameraInfoBlock::Project(const glm::vec3& position) const
{
	return m_projection * m_view * glm::vec4(position, 1.0f);
}

glm::vec3 CameraInfoBlock::UnProject(const glm::vec3& position) const
{
	const glm::mat4 inverse = glm::inverse(m_projection * m_view);
	auto start = glm::vec4(position, 1.0f);
	start = inverse * start;
	return start / start.w;
}

void Camera::UpdateGBuffer()
{
	m_gBufferNormalView.reset();
	m_gBufferAlbedoView.reset();
	m_gBufferMaterialView.reset();

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
		imageInfo.format = Graphics::Constants::G_BUFFER_COLOR;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		m_gBufferNormal = std::make_unique<Image>(imageInfo);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = m_gBufferNormal->GetVkImage();
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = Graphics::Constants::G_BUFFER_COLOR;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		m_gBufferNormalView = std::make_unique<ImageView>(viewInfo);

		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = Graphics::GetVkPhysicalDeviceProperties().limits.maxSamplerAnisotropy;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

		m_gBufferNormalSampler = std::make_unique<Sampler>(samplerInfo);
	}
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent = m_renderTexture->GetExtent();
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = Graphics::Constants::G_BUFFER_COLOR;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		m_gBufferAlbedo = std::make_unique<Image>(imageInfo);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = m_gBufferAlbedo->GetVkImage();
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = Graphics::Constants::G_BUFFER_COLOR;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		m_gBufferAlbedoView = std::make_unique<ImageView>(viewInfo);

		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = Graphics::GetVkPhysicalDeviceProperties().limits.maxSamplerAnisotropy;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

		m_gBufferAlbedoSampler = std::make_unique<Sampler>(samplerInfo);
	}
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent = m_renderTexture->GetExtent();
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = Graphics::Constants::G_BUFFER_COLOR;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		m_gBufferMaterial = std::make_unique<Image>(imageInfo);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = m_gBufferMaterial->GetVkImage();
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = Graphics::Constants::G_BUFFER_COLOR;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		m_gBufferMaterialView = std::make_unique<ImageView>(viewInfo);

		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = Graphics::GetVkPhysicalDeviceProperties().limits.maxSamplerAnisotropy;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

		m_gBufferMaterialSampler = std::make_unique<Sampler>(samplerInfo);

	}
	Graphics::ImmediateSubmit([&](VkCommandBuffer commandBuffer)
		{
			TransitGBufferImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		});


	EditorLayer::UpdateTextureId(m_gBufferNormalImTextureId, m_gBufferNormalSampler->GetVkSampler(), m_gBufferNormalView->GetVkImageView(), m_gBufferNormal->GetLayout());
	EditorLayer::UpdateTextureId(m_gBufferAlbedoImTextureId, m_gBufferAlbedoSampler->GetVkSampler(), m_gBufferAlbedoView->GetVkImageView(), m_gBufferAlbedo->GetLayout());
	EditorLayer::UpdateTextureId(m_gBufferMaterialImTextureId, m_gBufferMaterialSampler->GetVkSampler(), m_gBufferMaterialView->GetVkImageView(), m_gBufferMaterial->GetLayout());

	{
		VkDescriptorImageInfo imageInfo{};
		imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfo.imageView = m_renderTexture->GetDepthImageView()->GetVkImageView();
		imageInfo.sampler = m_renderTexture->m_depthSampler->GetVkSampler();
		m_gBufferDescriptorSet->UpdateImageDescriptorBinding(18, imageInfo);
		imageInfo.imageView = m_gBufferNormalView->GetVkImageView();
		imageInfo.sampler = m_gBufferNormalSampler->GetVkSampler();
		m_gBufferDescriptorSet->UpdateImageDescriptorBinding(19, imageInfo);
		imageInfo.imageView = m_gBufferAlbedoView->GetVkImageView();
		imageInfo.sampler = m_gBufferAlbedoSampler->GetVkSampler();
		m_gBufferDescriptorSet->UpdateImageDescriptorBinding(20, imageInfo);
		imageInfo.imageView = m_gBufferMaterialView->GetVkImageView();
		imageInfo.sampler = m_gBufferMaterialSampler->GetVkSampler();
		m_gBufferDescriptorSet->UpdateImageDescriptorBinding(21, imageInfo);
	}
}

void Camera::TransitGBufferImageLayout(VkCommandBuffer commandBuffer, VkImageLayout targetLayout) const
{
	m_gBufferNormal->TransitImageLayout(commandBuffer, targetLayout);
	m_gBufferAlbedo->TransitImageLayout(commandBuffer, targetLayout);
	m_gBufferMaterial->TransitImageLayout(commandBuffer, targetLayout);
}

void Camera::UpdateCameraInfoBlock(CameraInfoBlock& cameraInfoBlock, const GlobalTransform& globalTransform)
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

	if (const auto cameraSkybox = m_skybox.Get<Cubemap>())
	{
		cameraInfoBlock.m_skyboxTextureIndex = cameraSkybox->GetTextureStorageIndex();
	}
	else {
		auto defaultCubemap = Resources::GetResource<Cubemap>("DEFAULT_SKYBOX");
		cameraInfoBlock.m_skyboxTextureIndex = defaultCubemap->GetTextureStorageIndex();
	}

	const auto cameraPosition = globalTransform.GetPosition();
	const auto scene = Application::GetActiveScene();
	auto lightProbe = scene->m_environment.GetLightProbe(cameraPosition);
	auto reflectionProbe = scene->m_environment.GetReflectionProbe(cameraPosition);
	if (!lightProbe)
	{
		lightProbe = Resources::GetResource<EnvironmentalMap>("DEFAULT_ENVIRONMENTAL_MAP")->m_lightProbe.Get<LightProbe>();
	}
	cameraInfoBlock.m_environmentalIrradianceTextureIndex = lightProbe->m_cubemap->GetTextureStorageIndex();
	if (!reflectionProbe)
	{
		reflectionProbe = Resources::GetResource<EnvironmentalMap>("DEFAULT_ENVIRONMENTAL_MAP")->m_reflectionProbe.Get<ReflectionProbe>();
	}
	cameraInfoBlock.m_environmentalPrefilteredIndex = reflectionProbe->m_cubemap->GetTextureStorageIndex();
}

void Camera::AppendGBufferColorAttachmentInfos(std::vector<VkRenderingAttachmentInfo>& attachmentInfos, const VkAttachmentLoadOp loadOp, const VkAttachmentStoreOp storeOp) const
{
	VkRenderingAttachmentInfo attachment{};
	attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;

	attachment.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
	attachment.loadOp = loadOp;
	attachment.storeOp = storeOp;

	attachment.clearValue = { 0, 0, 0, 0 };
	attachment.imageView = m_gBufferNormalView->GetVkImageView();
	attachmentInfos.push_back(attachment);

	attachment.clearValue = { 0, 0, 0, 0 };
	attachment.imageView = m_gBufferAlbedoView->GetVkImageView();
	attachmentInfos.push_back(attachment);

	attachment.clearValue = { 0, 0, 0, 0 };
	attachment.imageView = m_gBufferMaterialView->GetVkImageView();
	attachmentInfos.push_back(attachment);
}



float Camera::GetSizeRatio() const
{
	if (m_size.x == 0 || m_size.y == 0)
		return 0;
	return static_cast<float>(m_size.x) / static_cast<float>(m_size.y);
}

const std::shared_ptr<RenderTexture>& Camera::GetRenderTexture() const
{
	return m_renderTexture;
}

glm::uvec2 Camera::GetSize() const
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
}

void Camera::OnCreate()
{
	m_size = glm::uvec2(1, 1);
	RenderTextureCreateInfo renderTextureCreateInfo{};
	renderTextureCreateInfo.m_extent.width = m_size.x;
	renderTextureCreateInfo.m_extent.height = m_size.y;
	renderTextureCreateInfo.m_extent.depth = 1;
	m_renderTexture = std::make_unique<RenderTexture>(renderTextureCreateInfo);

	m_gBufferDescriptorSet = std::make_shared<DescriptorSet>(Graphics::GetDescriptorSetLayout("CAMERA_GBUFFER_LAYOUT"));
	UpdateGBuffer();
}

bool Camera::Rendered() const
{
	return m_rendered;
}
void Camera::SetRequireRendering(const bool value)
{
	m_requireRendering = m_requireRendering || value;
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
	//yawAngle = glm::abs(angle.z) > 90.0f ? 90.0f - angle.y : -90.0f - angle.y;
	glm::vec3 front = rotation * glm::vec3(0, 0, -1);
	front.y = 0;
	yawAngle = glm::degrees(glm::acos(glm::dot(glm::vec3(0, 0, 1), glm::normalize(front))));
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

void Camera::Serialize(YAML::Emitter& out) const
{
	out << YAML::Key << "m_resolutionX" << YAML::Value << m_size.x;
	out << YAML::Key << "m_resolutionY" << YAML::Value << m_size.y;
	out << YAML::Key << "m_useClearColor" << YAML::Value << m_useClearColor;
	out << YAML::Key << "m_clearColor" << YAML::Value << m_clearColor;
	out << YAML::Key << "m_nearDistance" << YAML::Value << m_nearDistance;
	out << YAML::Key << "m_farDistance" << YAML::Value << m_farDistance;
	out << YAML::Key << "m_fov" << YAML::Value << m_fov;

	m_skybox.Save("m_skybox", out);
	m_postProcessingStack.Save("m_postProcessingStack", out);
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
	m_skybox.Load("m_skybox", in);
	m_postProcessingStack.Load("m_postProcessingStack", in);
	m_rendered = false;
	m_requireRendering = false;
}

void Camera::OnDestroy()
{
	m_renderTexture.reset();
	m_skybox.Clear();

}

bool Camera::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	bool changed = false;
	if (ImGui::TreeNode("Contents"))
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

			ImGui::SameLine();
			ImGui::Image(m_gBufferNormalImTextureId,
				ImVec2(m_size.x * debugSacle, m_size.y * debugSacle),
				ImVec2(0, 1),
				ImVec2(1, 0));

			ImGui::Image(m_gBufferAlbedoImTextureId,
				ImVec2(m_size.x * debugSacle, m_size.y * debugSacle),
				ImVec2(0, 1),
				ImVec2(1, 0));
			ImGui::SameLine();
			ImGui::Image(m_gBufferMaterialImTextureId,
				ImVec2(m_size.x * debugSacle, m_size.y * debugSacle),
				ImVec2(0, 1),
				ImVec2(1, 0));
		}
		ImGui::TreePop();
	}


	ImGui::Checkbox("Use clear color", &m_useClearColor);
	const auto scene = GetScene();
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
	if (!isMainCamera || !Application::GetLayer<EditorLayer>()->m_mainCameraAllowAutoResize)
	{
		glm::ivec2 resolution = { m_size.x, m_size.y };
		if (ImGui::DragInt2("Resolution", &resolution.x, 1, 1, 4096))
		{
			Resize({ resolution.x, resolution.y });
		}
	}
	if (m_useClearColor)
	{
		ImGui::ColorEdit3("Clear Color", (float*)(void*)&m_clearColor);
	}
	else
	{
		editorLayer->DragAndDropButton<Cubemap>(m_skybox, "Skybox");
	}

	editorLayer->DragAndDropButton<PostProcessingStack>(m_postProcessingStack, "PostProcessingStack");
	const auto pps = m_postProcessingStack.Get<PostProcessingStack>();
	if(pps && ImGui::TreeNode("Post Processing"))
	{
		ImGui::Checkbox("SSAO", &pps->m_SSAO);
		ImGui::Checkbox("SSR", &pps->m_SSR);
		
		ImGui::Checkbox("Bloom", &pps->m_bloom);
		ImGui::TreePop();
	}
	if (ImGui::TreeNode("Intrinsic Settings")) {
		ImGui::DragFloat("Near", &m_nearDistance, m_nearDistance / 10.0f, 0, m_farDistance);
		ImGui::DragFloat("Far", &m_farDistance, m_farDistance / 10.0f, m_nearDistance);
		ImGui::DragFloat("FOV", &m_fov, 1.0f, 1, 359);
		ImGui::TreePop();
	}
	FileUtils::SaveFile("Screenshot", "Image", { ".png", ".jpg", ".hdr" }, [this](const std::filesystem::path& filePath) {
		m_renderTexture->Save(filePath);
		}, false
	);

	return changed;
}

void Camera::CollectAssetRef(std::vector<AssetRef>& list)
{
	list.push_back(m_skybox);
	list.push_back(m_postProcessingStack);
}
