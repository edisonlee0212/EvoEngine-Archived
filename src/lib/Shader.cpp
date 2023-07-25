#include "Shader.hpp"

#include "Application.hpp"
#include "Console.hpp"
#include "Graphics.hpp"
#include "RenderLayer.hpp"
#include "Utilities.hpp"

using namespace EvoEngine;

void Shader::Set(const ShaderType shaderType, const std::string& shaderCode)
{
	m_shaderType = shaderType;
	m_code = shaderCode;
}

const std::unique_ptr<ShaderEXT>& Shader::GetShaderEXT() const
{
	return m_shaderEXT;
}

void ShaderProgram::CheckDescriptorSetsReady()
{
	for (const auto& binding : m_descriptorSetLayoutBindings)
	{
		if (!binding.m_ready)
		{
			m_descriptorSetsReady = false;
		}
	}
	m_descriptorSetsReady = true;
}

void ShaderProgram::ClearDescriptorSets()
{
	m_descriptorSetLayoutBindings.clear();

	m_perFrameLayout.reset();
	m_perPassLayout.reset();
	m_perGroupLayout.reset();
	m_perCommandLayout.reset();

	vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perFrameDescriptorSets.size(), m_perFrameDescriptorSets.data());
	vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perPassDescriptorSets.size(), m_perPassDescriptorSets.data());
	vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perGroupDescriptorSets.size(), m_perGroupDescriptorSets.data());
	vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perCommandDescriptorSets.size(), m_perCommandDescriptorSets.data());

	m_perFrameDescriptorSets.clear();
	m_perPassDescriptorSets.clear();
	m_perGroupDescriptorSets.clear();
	m_perCommandDescriptorSets.clear();

	m_containStandardBindings = false;
	m_layoutReady = false;
	m_descriptorSetsReady = false;
}

void ShaderProgram::InitializeStandardBindings()
{
	if (m_containStandardBindings)
	{
		EVOENGINE_ERROR("Already contain standard layout!");
		return;
	}
	if (!m_descriptorSetLayoutBindings.empty())
	{
		EVOENGINE_ERROR("Already contain bindings!");
		return;
	}

	PushDescriptorBinding(DescriptorSetType::PerFrame, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL);
	PushDescriptorBinding(DescriptorSetType::PerFrame, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL);
	PushDescriptorBinding(DescriptorSetType::PerPass, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL);
	PushDescriptorBinding(DescriptorSetType::PerGroup, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL);
	PushDescriptorBinding(DescriptorSetType::PerCommand, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL);

	m_containStandardBindings = true;
	m_layoutReady = false;
	m_descriptorSetsReady = false;
}

uint32_t ShaderProgram::PushDescriptorBinding(const DescriptorSetType setType, const VkDescriptorType type, const VkShaderStageFlags stageFlags)
{
	VkDescriptorSetLayoutBinding binding{};
	binding.binding = m_descriptorSetLayoutBindings.size();
	binding.descriptorCount = 1;
	binding.descriptorType = type;
	binding.pImmutableSamplers = nullptr;
	binding.stageFlags = stageFlags;
	m_descriptorSetLayoutBindings.push_back({ setType, binding, false });

	m_descriptorSetsReady = false;
	m_layoutReady = false;
	return binding.binding;
}

bool ShaderProgram::ContainStandardBindings() const
{
	return m_containStandardBindings;
}

void ShaderProgram::PrepareLayouts()
{
	m_perFrameLayout.reset();
	m_perPassLayout.reset();
	m_perGroupLayout.reset();
	m_perCommandLayout.reset();

	vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perFrameDescriptorSets.size(), m_perFrameDescriptorSets.data());
	vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perPassDescriptorSets.size(), m_perPassDescriptorSets.data());
	vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perGroupDescriptorSets.size(), m_perGroupDescriptorSets.data());
	vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perCommandDescriptorSets.size(), m_perCommandDescriptorSets.data());

	m_perFrameDescriptorSets.clear();
	m_perPassDescriptorSets.clear();
	m_perGroupDescriptorSets.clear();
	m_perCommandDescriptorSets.clear();

	std::vector<VkDescriptorSetLayoutBinding> perFrameBindings;
	std::vector<VkDescriptorSetLayoutBinding> perPassBindings;
	std::vector<VkDescriptorSetLayoutBinding> perGroupBindings;
	std::vector<VkDescriptorSetLayoutBinding> perCommandBindings;

	for (const auto& binding : m_descriptorSetLayoutBindings)
	{
		switch (binding.m_type)
		{
		case DescriptorSetType::PerFrame:
		{
			perFrameBindings.push_back(binding.m_binding);
		}break;
		case DescriptorSetType::PerPass:
		{
			perPassBindings.push_back(binding.m_binding);
		}break;
		case DescriptorSetType::PerGroup:
		{
			perGroupBindings.push_back(binding.m_binding);
		}break;
		case DescriptorSetType::PerCommand:
		{
			perCommandBindings.push_back(binding.m_binding);
		}break;
		}
	}

	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();
	VkDescriptorSetLayoutCreateInfo perFrameLayoutInfo{};
	perFrameLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	perFrameLayoutInfo.bindingCount = static_cast<uint32_t>(perFrameBindings.size());
	perFrameLayoutInfo.pBindings = perFrameBindings.data();
	m_perFrameLayout = std::make_unique<DescriptorSetLayout>(perFrameLayoutInfo);

	VkDescriptorSetLayoutCreateInfo perPassLayoutInfo{};
	perPassLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	perPassLayoutInfo.bindingCount = static_cast<uint32_t>(perPassBindings.size());
	perPassLayoutInfo.pBindings = perPassBindings.data();
	m_perPassLayout = std::make_unique<DescriptorSetLayout>(perPassLayoutInfo);

	VkDescriptorSetLayoutCreateInfo perGroupLayoutInfo{};
	perGroupLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	perGroupLayoutInfo.bindingCount = static_cast<uint32_t>(perGroupBindings.size());
	perGroupLayoutInfo.pBindings = perGroupBindings.data();
	m_perGroupLayout = std::make_unique<DescriptorSetLayout>(perGroupLayoutInfo);

	VkDescriptorSetLayoutCreateInfo perCommandLayoutInfo{};
	perCommandLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	perCommandLayoutInfo.bindingCount = static_cast<uint32_t>(perCommandBindings.size());
	perCommandLayoutInfo.pBindings = perCommandBindings.data();
	m_perCommandLayout = std::make_unique<DescriptorSetLayout>(perCommandLayoutInfo);

	std::vector setLayouts = { m_perFrameLayout->GetVkDescriptorSetLayout(), m_perPassLayout->GetVkDescriptorSetLayout(), m_perGroupLayout->GetVkDescriptorSetLayout(), m_perCommandLayout->GetVkDescriptorSetLayout() };
	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = setLayouts.size();
	pipelineLayoutInfo.pSetLayouts = setLayouts.data();
	m_pipelineLayout = std::make_unique<PipelineLayout>(pipelineLayoutInfo);

	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = Graphics::GetDescriptorPool()->GetVkDescriptorPool();

	const std::vector perFrameLayouts(maxFramesInFlight, m_perFrameLayout->GetVkDescriptorSetLayout());
	allocInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	allocInfo.pSetLayouts = perFrameLayouts.data();
	m_perFrameDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, m_perFrameDescriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}

	const std::vector perPassLayout(maxFramesInFlight, m_perPassLayout->GetVkDescriptorSetLayout());
	allocInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	allocInfo.pSetLayouts = perPassLayout.data();
	m_perPassDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, m_perPassDescriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}

	const std::vector perObjectGroupLayout(maxFramesInFlight, m_perGroupLayout->GetVkDescriptorSetLayout());
	allocInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	allocInfo.pSetLayouts = perObjectGroupLayout.data();
	m_perGroupDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, m_perGroupDescriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}

	const std::vector perObjectLayout(maxFramesInFlight, m_perCommandLayout->GetVkDescriptorSetLayout());
	allocInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	allocInfo.pSetLayouts = perObjectLayout.data();
	m_perCommandDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, m_perCommandDescriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}
	m_layoutReady = true;
}

void ShaderProgram::UpdateStandardBindings()
{
	if (!m_containStandardBindings)
	{
		EVOENGINE_ERROR("No standard layout!");
		return;
	}
	auto& graphics = Graphics::GetInstance();
	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();
	VkDescriptorBufferInfo bufferInfos[5] = {};
	bufferInfos[0].offset = 0;
	bufferInfos[0].range = sizeof(RenderInfoBlock);
	bufferInfos[1].offset = 0;
	bufferInfos[1].range = sizeof(EnvironmentInfoBlock);
	bufferInfos[2].offset = 0;
	bufferInfos[2].range = sizeof(CameraInfoBlock);
	bufferInfos[3].offset = 0;
	bufferInfos[3].range = sizeof(MaterialInfoBlock);
	bufferInfos[4].offset = 0;
	bufferInfos[4].range = sizeof(ObjectInfoBlock);
	for (size_t i = 0; i < maxFramesInFlight; i++) {
		bufferInfos[0].buffer = graphics.m_standardDescriptorBuffers[i * 4 + 0]->GetVkBuffer();
		vmaMapMemory(Graphics::GetVmaAllocator(), graphics.m_standardDescriptorBuffers[i * 4 + 0]->GetVmaAllocation(), &graphics.m_renderInfoBlockMemory[i]);

		bufferInfos[1].buffer = graphics.m_standardDescriptorBuffers[i * 4 + 1]->GetVkBuffer();
		vmaMapMemory(Graphics::GetVmaAllocator(), graphics.m_standardDescriptorBuffers[i * 4 + 1]->GetVmaAllocation(), &graphics.m_environmentalInfoBlockMemory[i]);

		bufferInfos[2].buffer = graphics.m_standardDescriptorBuffers[i * 4 + 2]->GetVkBuffer();
		vmaMapMemory(Graphics::GetVmaAllocator(), graphics.m_standardDescriptorBuffers[i * 4 + 2]->GetVmaAllocation(), &graphics.m_cameraInfoBlockMemory[i]);

		bufferInfos[3].buffer = graphics.m_standardDescriptorBuffers[i * 4 + 3]->GetVkBuffer();
		vmaMapMemory(Graphics::GetVmaAllocator(), graphics.m_standardDescriptorBuffers[i * 4 + 3]->GetVmaAllocation(), &graphics.m_materialInfoBlockMemory[i]);

		bufferInfos[4].buffer = graphics.m_standardDescriptorBuffers[i * 4 + 4]->GetVkBuffer();
		vmaMapMemory(Graphics::GetVmaAllocator(), graphics.m_standardDescriptorBuffers[i * 4 + 4]->GetVmaAllocation(), &graphics.m_objectInfoBlockMemory[i]);

		VkWriteDescriptorSet renderInfo{};
		renderInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		renderInfo.dstSet = m_perFrameDescriptorSets[i];
		renderInfo.dstBinding = 0;
		renderInfo.dstArrayElement = 0;
		renderInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		renderInfo.descriptorCount = 1;
		renderInfo.pBufferInfo = &bufferInfos[0];

		VkWriteDescriptorSet environmentInfo{};
		environmentInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		environmentInfo.dstSet = m_perFrameDescriptorSets[i];
		environmentInfo.dstBinding = 1;
		environmentInfo.dstArrayElement = 0;
		environmentInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		environmentInfo.descriptorCount = 1;
		environmentInfo.pBufferInfo = &bufferInfos[1];

		VkWriteDescriptorSet cameraInfo{};
		cameraInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		cameraInfo.dstSet = m_perPassDescriptorSets[i];
		cameraInfo.dstBinding = 2;
		cameraInfo.dstArrayElement = 0;
		cameraInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		cameraInfo.descriptorCount = 1;
		cameraInfo.pBufferInfo = &bufferInfos[2];

		VkWriteDescriptorSet materialInfo{};
		materialInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		materialInfo.dstSet = m_perGroupDescriptorSets[i];
		materialInfo.dstBinding = 3;
		materialInfo.dstArrayElement = 0;
		materialInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		materialInfo.descriptorCount = 1;
		materialInfo.pBufferInfo = &bufferInfos[3];

		VkWriteDescriptorSet objectInfo{};
		objectInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		objectInfo.dstSet = m_perCommandDescriptorSets[i];
		objectInfo.dstBinding = 4;
		objectInfo.dstArrayElement = 0;
		objectInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		objectInfo.descriptorCount = 1;
		objectInfo.pBufferInfo = &bufferInfos[4];

		std::vector writeInfos = { renderInfo, environmentInfo, cameraInfo, materialInfo, objectInfo };
		vkUpdateDescriptorSets(Graphics::GetVkDevice(), writeInfos.size(), writeInfos.data(), 0, nullptr);
	}
	m_descriptorSetLayoutBindings[0].m_ready = true;
	m_descriptorSetLayoutBindings[1].m_ready = true;
	m_descriptorSetLayoutBindings[2].m_ready = true;
	m_descriptorSetLayoutBindings[3].m_ready = true;
	m_descriptorSetLayoutBindings[4].m_ready = true;
	CheckDescriptorSetsReady();
}

void ShaderProgram::UpdateBinding(const uint32_t binding, const VkDescriptorBufferInfo& bufferInfo)
{
	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();
	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		const auto& descriptorBinding = m_descriptorSetLayoutBindings[binding];
		VkWriteDescriptorSet writeInfo{};
		writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		switch (descriptorBinding.m_type)
		{
		case DescriptorSetType::PerFrame:
		{
			writeInfo.dstSet = m_perFrameDescriptorSets[i];
		}break;
		case DescriptorSetType::PerPass:
		{
			writeInfo.dstSet = m_perPassDescriptorSets[i];
		}break;
		case DescriptorSetType::PerGroup:
		{
			writeInfo.dstSet = m_perGroupDescriptorSets[i];
		}break;
		case DescriptorSetType::PerCommand:
		{
			writeInfo.dstSet = m_perCommandDescriptorSets[i];
		}break;
		default: break;
		}
		writeInfo.dstBinding = binding;
		writeInfo.dstArrayElement = 0;
		writeInfo.descriptorType = descriptorBinding.m_binding.descriptorType;
		writeInfo.descriptorCount = descriptorBinding.m_binding.descriptorCount;
		writeInfo.pBufferInfo = &bufferInfo;
		vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
	}
	m_descriptorSetLayoutBindings[binding].m_ready = true;
	CheckDescriptorSetsReady();
}

void ShaderProgram::UpdateBinding(const uint32_t binding, const VkDescriptorImageInfo& imageInfo)
{
	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();
	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		const auto& descriptorBinding = m_descriptorSetLayoutBindings[binding];
		VkWriteDescriptorSet writeInfo{};
		writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		switch (descriptorBinding.m_type)
		{
		case DescriptorSetType::PerFrame:
		{
			writeInfo.dstSet = m_perFrameDescriptorSets[i];
		}break;
		case DescriptorSetType::PerPass:
		{
			writeInfo.dstSet = m_perPassDescriptorSets[i];
		}break;
		case DescriptorSetType::PerGroup:
		{
			writeInfo.dstSet = m_perGroupDescriptorSets[i];
		}break;
		case DescriptorSetType::PerCommand:
		{
			writeInfo.dstSet = m_perCommandDescriptorSets[i];
		}break;
		default: break;
		}
		writeInfo.dstBinding = binding;
		writeInfo.dstArrayElement = 0;
		writeInfo.descriptorType = descriptorBinding.m_binding.descriptorType;
		writeInfo.descriptorCount = descriptorBinding.m_binding.descriptorCount;
		writeInfo.pImageInfo = &imageInfo;
		vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
	}
	m_descriptorSetLayoutBindings[binding].m_ready = true;
	CheckDescriptorSetsReady();
}

bool ShaderProgram::LayoutSetsReady() const
{
	return m_layoutReady;
}

bool ShaderProgram::DescriptorSetsReady() const
{
	return m_descriptorSetsReady;
}

void ShaderProgram::BindDescriptorSet(const VkCommandBuffer commandBuffer, const DescriptorSetType setType) const
{
	const auto frameIndex = Graphics::GetCurrentFrameIndex();
	VkDescriptorSet descriptorSet = {};
	switch (setType) {
	case DescriptorSetType::PerFrame:
	{
		descriptorSet = m_perFrameDescriptorSets[frameIndex];
	}break;
	case DescriptorSetType::PerPass:
	{
		descriptorSet = m_perPassDescriptorSets[frameIndex];
	}break;
	case DescriptorSetType::PerGroup:
	{
		descriptorSet = m_perGroupDescriptorSets[frameIndex];
	}break;
	case DescriptorSetType::PerCommand:
	{
		descriptorSet = m_perCommandDescriptorSets[frameIndex];
	}break;
	default: break;
	}
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		m_pipelineLayout->GetVkPipelineLayout(), 0, 1, &descriptorSet, 0, nullptr);

}

void ShaderProgram::BindAllDescriptorSets(const VkCommandBuffer commandBuffer) const
{
	const auto frameIndex = Graphics::GetCurrentFrameIndex();
	const std::vector descriptorSets = {
		m_perFrameDescriptorSets[frameIndex],
		m_perPassDescriptorSets[frameIndex],
		m_perGroupDescriptorSets[frameIndex],
		m_perCommandDescriptorSets[frameIndex],
	};
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		m_pipelineLayout->GetVkPipelineLayout(), 0, descriptorSets.size(), descriptorSets.data(), 0, nullptr);
}

void ShaderProgram::BindShaders(const VkCommandBuffer commandBuffer)
{
	constexpr  VkShaderStageFlagBits stages[6] =
	{
		VK_SHADER_STAGE_VERTEX_BIT,
		VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
		VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
		VK_SHADER_STAGE_GEOMETRY_BIT,
		VK_SHADER_STAGE_FRAGMENT_BIT,
		VK_SHADER_STAGE_COMPUTE_BIT
	};
	if (const auto vertShader = m_vertexShader.Get<Shader>()) vkCmdBindShadersEXT(commandBuffer, 1, &stages[0], &vertShader->GetShaderEXT()->GetVkShaderEXT());
	else vkCmdBindShadersEXT(commandBuffer, 1, &stages[0], nullptr);
	if (const auto tessControlShader = m_tessellationControlShader.Get<Shader>()) vkCmdBindShadersEXT(commandBuffer, 1, &stages[1], &tessControlShader->GetShaderEXT()->GetVkShaderEXT());
	else vkCmdBindShadersEXT(commandBuffer, 1, &stages[1], nullptr);
	if (const auto tessEvaluationShader = m_tessellationEvaluationShader.Get<Shader>()) vkCmdBindShadersEXT(commandBuffer, 1, &stages[2], &tessEvaluationShader->GetShaderEXT()->GetVkShaderEXT());
	else vkCmdBindShadersEXT(commandBuffer, 1, &stages[2], nullptr);
	if (const auto geometryShader = m_geometryShader.Get<Shader>()) vkCmdBindShadersEXT(commandBuffer, 1, &stages[3], &geometryShader->GetShaderEXT()->GetVkShaderEXT());
	else vkCmdBindShadersEXT(commandBuffer, 1, &stages[3], nullptr);
	if (const auto fragmentShader = m_fragmentShader.Get<Shader>()) vkCmdBindShadersEXT(commandBuffer, 1, &stages[4], &fragmentShader->GetShaderEXT()->GetVkShaderEXT());
	else vkCmdBindShadersEXT(commandBuffer, 1, &stages[4], nullptr);
	if (const auto computeShader = m_computeShader.Get<Shader>()) vkCmdBindShadersEXT(commandBuffer, 1, &stages[5], &computeShader->GetShaderEXT()->GetVkShaderEXT());
	else vkCmdBindShadersEXT(commandBuffer, 1, &stages[5], nullptr);

}

void ShaderProgram::LinkShaders()
{
	if (m_descriptorSetLayoutBindings.empty() && !m_layoutReady)
	{
		EVOENGINE_ERROR("Has descriptors but the layout is not ready!");
		return;
	}
	std::vector setLayouts = {
		m_perFrameLayout->GetVkDescriptorSetLayout(),
		m_perPassLayout->GetVkDescriptorSetLayout(),
		m_perGroupLayout->GetVkDescriptorSetLayout(),
		m_perCommandLayout->GetVkDescriptorSetLayout()
	};
	if (const auto vertShader = m_vertexShader.Get<Shader>())
	{
		VkShaderCreateInfoEXT shaderCreateInfo{};
		shaderCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		const auto fragShaderBinary = ShaderUtils::CompileFile("VertShader", shaderc_vertex_shader, vertShader->m_code);

		shaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
		shaderCreateInfo.pNext = nullptr;
		shaderCreateInfo.flags = 0;

		shaderCreateInfo.nextStage = 0;
		shaderCreateInfo.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
		shaderCreateInfo.codeSize = fragShaderBinary.size() * sizeof(uint32_t);
		shaderCreateInfo.pCode = fragShaderBinary.data();
		shaderCreateInfo.pName = "main";
		if (!m_descriptorSetLayoutBindings.empty()) {
			shaderCreateInfo.setLayoutCount = setLayouts.size();
			shaderCreateInfo.pSetLayouts = setLayouts.data();
		}
		else
		{
			shaderCreateInfo.setLayoutCount = 0;
			shaderCreateInfo.pSetLayouts = nullptr;
		}
		shaderCreateInfo.pushConstantRangeCount = 0;
		shaderCreateInfo.pPushConstantRanges = nullptr;
		shaderCreateInfo.pSpecializationInfo = nullptr;

		vertShader->m_shaderEXT.reset();
		vertShader->m_shaderEXT = std::make_unique<ShaderEXT>(shaderCreateInfo);
	}
	if (const auto tessControlShader = m_tessellationControlShader.Get<Shader>())
	{
		VkShaderCreateInfoEXT shaderCreateInfo{};
		shaderCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		const auto fragShaderBinary = ShaderUtils::CompileFile("TessControlShader", shaderc_tess_control_shader, tessControlShader->m_code);

		shaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
		shaderCreateInfo.pNext = nullptr;
		shaderCreateInfo.flags = 0;

		shaderCreateInfo.nextStage = 0;
		shaderCreateInfo.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
		shaderCreateInfo.codeSize = fragShaderBinary.size() * sizeof(uint32_t);
		shaderCreateInfo.pCode = fragShaderBinary.data();
		shaderCreateInfo.pName = "main";
		if (!m_descriptorSetLayoutBindings.empty()) {
			shaderCreateInfo.setLayoutCount = setLayouts.size();
			shaderCreateInfo.pSetLayouts = setLayouts.data();
		}
		else
		{
			shaderCreateInfo.setLayoutCount = 0;
			shaderCreateInfo.pSetLayouts = nullptr;
		}
		shaderCreateInfo.pushConstantRangeCount = 0;
		shaderCreateInfo.pPushConstantRanges = nullptr;
		shaderCreateInfo.pSpecializationInfo = nullptr;

		tessControlShader->m_shaderEXT.reset();
		tessControlShader->m_shaderEXT = std::make_unique<ShaderEXT>(shaderCreateInfo);
	}
	if (const auto tessEvaluationShader = m_tessellationEvaluationShader.Get<Shader>())
	{
		VkShaderCreateInfoEXT shaderCreateInfo{};
		shaderCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		const auto fragShaderBinary = ShaderUtils::CompileFile("TessEvaluationShader", shaderc_tess_evaluation_shader, tessEvaluationShader->m_code);

		shaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
		shaderCreateInfo.pNext = nullptr;
		shaderCreateInfo.flags = 0;

		shaderCreateInfo.nextStage = 0;
		shaderCreateInfo.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
		shaderCreateInfo.codeSize = fragShaderBinary.size() * sizeof(uint32_t);
		shaderCreateInfo.pCode = fragShaderBinary.data();
		shaderCreateInfo.pName = "main";
		if (!m_descriptorSetLayoutBindings.empty()) {
			shaderCreateInfo.setLayoutCount = setLayouts.size();
			shaderCreateInfo.pSetLayouts = setLayouts.data();
		}
		else
		{
			shaderCreateInfo.setLayoutCount = 0;
			shaderCreateInfo.pSetLayouts = nullptr;
		}
		shaderCreateInfo.pushConstantRangeCount = 0;
		shaderCreateInfo.pPushConstantRanges = nullptr;
		shaderCreateInfo.pSpecializationInfo = nullptr;

		tessEvaluationShader->m_shaderEXT.reset();
		tessEvaluationShader->m_shaderEXT = std::make_unique<ShaderEXT>(shaderCreateInfo);
	}
	if (const auto geometryShader = m_geometryShader.Get<Shader>())
	{
		VkShaderCreateInfoEXT shaderCreateInfo{};
		shaderCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		const auto fragShaderBinary = ShaderUtils::CompileFile("GeometryShader", shaderc_geometry_shader, geometryShader->m_code);

		shaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
		shaderCreateInfo.pNext = nullptr;
		shaderCreateInfo.flags = 0;

		shaderCreateInfo.nextStage = 0;
		shaderCreateInfo.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
		shaderCreateInfo.codeSize = fragShaderBinary.size() * sizeof(uint32_t);
		shaderCreateInfo.pCode = fragShaderBinary.data();
		shaderCreateInfo.pName = "main";
		if (!m_descriptorSetLayoutBindings.empty()) {
			shaderCreateInfo.setLayoutCount = setLayouts.size();
			shaderCreateInfo.pSetLayouts = setLayouts.data();
		}
		else
		{
			shaderCreateInfo.setLayoutCount = 0;
			shaderCreateInfo.pSetLayouts = nullptr;
		}
		shaderCreateInfo.pushConstantRangeCount = 0;
		shaderCreateInfo.pPushConstantRanges = nullptr;
		shaderCreateInfo.pSpecializationInfo = nullptr;

		geometryShader->m_shaderEXT.reset();
		geometryShader->m_shaderEXT = std::make_unique<ShaderEXT>(shaderCreateInfo);
	}
	if (const auto fragmentShader = m_fragmentShader.Get<Shader>())
	{
		VkShaderCreateInfoEXT shaderCreateInfo{};
		shaderCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		const auto fragShaderBinary = ShaderUtils::CompileFile("FragmentShader", shaderc_fragment_shader, fragmentShader->m_code);

		shaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
		shaderCreateInfo.pNext = nullptr;
		shaderCreateInfo.flags = 0;

		shaderCreateInfo.nextStage = 0;
		shaderCreateInfo.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
		shaderCreateInfo.codeSize = fragShaderBinary.size() * sizeof(uint32_t);
		shaderCreateInfo.pCode = fragShaderBinary.data();
		shaderCreateInfo.pName = "main";
		if (!m_descriptorSetLayoutBindings.empty()) {
			shaderCreateInfo.setLayoutCount = setLayouts.size();
			shaderCreateInfo.pSetLayouts = setLayouts.data();
		}
		else
		{
			shaderCreateInfo.setLayoutCount = 0;
			shaderCreateInfo.pSetLayouts = nullptr;
		}
		shaderCreateInfo.pushConstantRangeCount = 0;
		shaderCreateInfo.pPushConstantRanges = nullptr;
		shaderCreateInfo.pSpecializationInfo = nullptr;

		fragmentShader->m_shaderEXT.reset();
		fragmentShader->m_shaderEXT = std::make_unique<ShaderEXT>(shaderCreateInfo);
	}
	if (const auto computeShader = m_computeShader.Get<Shader>())
	{
		VkShaderCreateInfoEXT shaderCreateInfo{};
		shaderCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		const auto fragShaderBinary = ShaderUtils::CompileFile("ComputeShader", shaderc_compute_shader, computeShader->m_code);

		shaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
		shaderCreateInfo.pNext = nullptr;
		shaderCreateInfo.flags = 0;

		shaderCreateInfo.nextStage = 0;
		shaderCreateInfo.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
		shaderCreateInfo.codeSize = fragShaderBinary.size() * sizeof(uint32_t);
		shaderCreateInfo.pCode = fragShaderBinary.data();
		shaderCreateInfo.pName = "main";
		if (!m_descriptorSetLayoutBindings.empty()) {
			shaderCreateInfo.setLayoutCount = setLayouts.size();
			shaderCreateInfo.pSetLayouts = setLayouts.data();
		}
		else
		{
			shaderCreateInfo.setLayoutCount = 0;
			shaderCreateInfo.pSetLayouts = nullptr;
		}
		shaderCreateInfo.pushConstantRangeCount = 0;
		shaderCreateInfo.pPushConstantRanges = nullptr;
		shaderCreateInfo.pSpecializationInfo = nullptr;

		computeShader->m_shaderEXT.reset();
		computeShader->m_shaderEXT = std::make_unique<ShaderEXT>(shaderCreateInfo);
	}
}
