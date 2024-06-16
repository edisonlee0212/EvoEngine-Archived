#include "AnimationPlayer.hpp"
#include "Animator.hpp"
#include "Scene.hpp"
#include "Times.hpp"

using namespace evo_engine;

void AnimationPlayer::Update() {
  if (const auto scene = GetScene()) {
    if (scene->HasPrivateComponent<Animator>(GetOwner())) {
      const auto animator = scene->GetOrSetPrivateComponent<Animator>(GetOwner()).lock();
      const auto animation = animator->GetAnimation();
      if (!animation)
        return;
      float current_animation_time = animator->GetCurrentAnimationTimePoint();
      current_animation_time += Times::DeltaTime() * m_autoPlaySpeed;
      if (current_animation_time > animation->GetAnimationLength(animator->GetCurrentAnimationName()))
        current_animation_time =
            glm::mod(current_animation_time, animation->GetAnimationLength(animator->GetCurrentAnimationName()));
      animator->Animate(current_animation_time);
    }
  }
}

bool AnimationPlayer::OnInspect(const std::shared_ptr<EditorLayer>& editor_layer) {
  bool changed = false;
  ImGui::Checkbox("AutoPlay", &m_autoPlay);
  if (m_autoPlay) {
    ImGui::DragFloat("AutoPlay Speed", &m_autoPlaySpeed, 1.0f);
  }

  return changed;
}
