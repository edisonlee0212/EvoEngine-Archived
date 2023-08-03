#include "Application.hpp"
#include "Scene.hpp"
#include "AnimationLayer.hpp"
#include "Animator.hpp"
#include "Jobs.hpp"
#include "SkinnedMeshRenderer.hpp"
using namespace EvoEngine;

void AnimationLayer::PreUpdate()
{
	const auto scene = GetScene();
	if (const auto* owners =
		scene->UnsafeGetPrivateComponentOwnersList<Animator>())
	{
		std::vector<std::shared_future<void>> results;

		Jobs::ParallelFor(owners->size(), [&](unsigned i)
			{
				const auto animator = scene->GetOrSetPrivateComponent<Animator>(owners->at(i)).lock();
				if (animator->m_animatedCurrentFrame)
				{
					animator->m_animatedCurrentFrame = false;
				}
				if (!Application::IsPlaying() && animator->m_autoPlay)
				{
					animator->AutoPlay();
				}
				animator->Apply();
			}, results);
		for (const auto& i : results)
			i.wait();
	}
}

void AnimationLayer::LateUpdate()
{
	const auto scene = GetScene();

	if (const auto* owners = scene->UnsafeGetPrivateComponentOwnersList<SkinnedMeshRenderer>())
	{
		std::vector<std::shared_future<void>> results;
		Jobs::ParallelFor(owners->size(), [&](unsigned i)
			{
				const auto skinnedMeshRenderer = scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(owners->at(i)).lock();
				skinnedMeshRenderer->UpdateBoneMatrices();
			}, results);
		for (const auto& i : results)
			i.wait();

		for(const auto& i : *owners)
		{
			const auto skinnedMeshRenderer = scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(i).lock();
			skinnedMeshRenderer->m_boneMatrices->UploadData();
		}
	}
}
