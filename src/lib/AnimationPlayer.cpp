#include "AnimationPlayer.hpp"
#include "Times.hpp"
#include "Animator.hpp"
#include "Scene.hpp"

using namespace EvoEngine;

void AnimationPlayer::Update()
{
	if(const auto scene = GetScene())
	{
		if(scene->HasPrivateComponent<Animator>(GetOwner()))
		{
			auto animator = scene->GetOrSetPrivateComponent<Animator>(GetOwner()).lock();
            const auto animation = animator->GetAnimation();
            if (!animation)
                return;
			float currentAnimationTime = animator->GetCurrentAnimationTimePoint();
			currentAnimationTime += Times::DeltaTime() * m_autoPlaySpeed;
			if (currentAnimationTime > animation->GetAnimationLength(animator->GetCurrentAnimationName()))
				currentAnimationTime =
				glm::mod(currentAnimationTime, animation->GetAnimationLength(animator->GetCurrentAnimationName()));
			animator->Animate(currentAnimationTime);
		}
	}
}

bool AnimationPlayer::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	bool changed = false;
	ImGui::Checkbox("AutoPlay", &m_autoPlay);
	if(m_autoPlay)
	{
		ImGui::DragFloat("AutoPlay Speed", &m_autoPlaySpeed, 1.0f);
	}

	return changed;
}
