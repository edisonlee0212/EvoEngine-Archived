#include "AnimationPlayer.hpp"

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
		}
	}
}
