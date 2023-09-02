#include "TextureStorage.hpp"

#include "Application.hpp"
#include "RenderLayer.hpp"

using namespace EvoEngine;


void TextureStorage::DeviceSync()
{
	auto& storage = GetInstance();
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	uint32_t currentArrayIndex = 0;
	const auto renderLayer = Application::GetLayer<RenderLayer>();
	for (int i = 0; i < storage.m_texture2DPendingUpdates[currentFrameIndex].size(); i++)
	{
		if (storage.m_texture2DPendingUpdates[currentFrameIndex][i])
		{
			VkDescriptorImageInfo imageInfo;
			imageInfo.imageLayout = storage.m_texture2Ds[currentArrayIndex]->GetLayout();
			imageInfo.imageView = storage.m_texture2Ds[currentArrayIndex]->GetVkImageView();
			imageInfo.sampler = storage.m_texture2Ds[currentArrayIndex]->GetVkSampler();
			renderLayer->m_perFrameDescriptorSets[currentFrameIndex]->UpdateImageDescriptorBinding(13, imageInfo, currentArrayIndex);
			storage.m_texture2DPendingUpdates[currentFrameIndex][i] = false;
		}
		currentArrayIndex++;
	}
	currentArrayIndex = 0;
	for (int i = 0; i < storage.m_cubemapPendingUpdates[currentFrameIndex].size(); i++)
	{
		if (storage.m_cubemapPendingUpdates[currentFrameIndex][i])
		{
			VkDescriptorImageInfo imageInfo;
			imageInfo.imageLayout = storage.m_cubemaps[currentArrayIndex]->m_image->GetLayout();
			imageInfo.imageView = storage.m_cubemaps[currentArrayIndex]->m_imageView->GetVkImageView();
			imageInfo.sampler = storage.m_cubemaps[currentArrayIndex]->m_sampler->GetVkSampler();
			renderLayer->m_perFrameDescriptorSets[currentFrameIndex]->UpdateImageDescriptorBinding(14, imageInfo, currentArrayIndex);
			storage.m_cubemapPendingUpdates[currentFrameIndex][i] = false;
		}
		currentArrayIndex++;
	}
}

void TextureStorage::UnRegisterTexture2D(const std::shared_ptr<Texture2D>& texture2D)
{
	auto& storage = GetInstance();
	const auto maxFrameInFlight = Graphics::GetMaxFramesInFlight();
	if (texture2D->m_textureStorageIndex != UINT32_MAX)
	{
		storage.m_texture2Ds[texture2D->m_textureStorageIndex] = storage.m_texture2Ds.back();
		storage.m_texture2Ds[texture2D->m_textureStorageIndex]->m_textureStorageIndex = texture2D->m_textureStorageIndex;
		storage.m_texture2Ds.pop_back();
		for (int i = 0; i < maxFrameInFlight; i++)
		{
			storage.m_texture2DPendingUpdates[i].pop_back();
			if (texture2D->m_textureStorageIndex < storage.m_texture2DPendingUpdates[i].size())
			{
				storage.m_texture2DPendingUpdates[i][texture2D->m_textureStorageIndex] = true;
			}
		}
		texture2D->m_textureStorageIndex = UINT32_MAX;
	}
}

void TextureStorage::UnRegisterCubemap(const std::shared_ptr<Cubemap>& cubemap)
{
	auto& storage = GetInstance();
	const auto maxFrameInFlight = Graphics::GetMaxFramesInFlight();
	if (cubemap->m_textureStorageIndex != UINT32_MAX)
	{
		storage.m_cubemaps[cubemap->m_textureStorageIndex] = storage.m_cubemaps.back();
		storage.m_cubemaps[cubemap->m_textureStorageIndex]->m_textureStorageIndex = cubemap->m_textureStorageIndex;
		storage.m_cubemaps.pop_back();
		for (int i = 0; i < maxFrameInFlight; i++)
		{
			storage.m_cubemapPendingUpdates[i].pop_back();
			if (cubemap->m_textureStorageIndex < storage.m_cubemapPendingUpdates[i].size())
			{
				storage.m_cubemapPendingUpdates[i][cubemap->m_textureStorageIndex] = true;
			}
		}
		cubemap->m_textureStorageIndex = UINT32_MAX;
	}
}

void TextureStorage::RegisterTexture2D(const std::shared_ptr<Texture2D>& texture2D)
{
	auto& storage = GetInstance();
	const auto maxFrameInFlight = Graphics::GetMaxFramesInFlight();
	if (texture2D->m_textureStorageIndex == UINT32_MAX)
	{
		texture2D->m_textureStorageIndex = storage.m_texture2Ds.size();
		storage.m_texture2Ds.emplace_back(texture2D);
		for (int i = 0; i < maxFrameInFlight; i++)
		{
			storage.m_texture2DPendingUpdates[i].emplace_back() = true;
		}
	}
	for (int i = 0; i < maxFrameInFlight; i++)
	{
		storage.m_texture2DPendingUpdates[i][texture2D->m_textureStorageIndex] = true;
	}
}

void TextureStorage::RegisterCubemap(const std::shared_ptr<Cubemap>& cubemap)
{
	auto& storage = GetInstance();
	const auto maxFrameInFlight = Graphics::GetMaxFramesInFlight();
	if (cubemap->m_textureStorageIndex == UINT32_MAX)
	{
		cubemap->m_textureStorageIndex = storage.m_cubemaps.size();
		storage.m_cubemaps.emplace_back(cubemap);
		for (int i = 0; i < maxFrameInFlight; i++)
		{
			storage.m_cubemapPendingUpdates[i].emplace_back() = true;
		}
	}

	for (int i = 0; i < maxFrameInFlight; i++)
	{
		storage.m_cubemapPendingUpdates[i][cubemap->m_textureStorageIndex] = true;
	}
}

void TextureStorage::Initialize()
{
	auto& storage = GetInstance();
	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();
	storage.m_texture2DPendingUpdates.resize(maxFramesInFlight);
	storage.m_cubemapPendingUpdates.resize(maxFramesInFlight);
}
