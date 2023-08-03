#pragma once
#include "ILayer.hpp"
namespace EvoEngine
{
class AnimationLayer : public ILayer
{
    void PreUpdate() override;
    void LateUpdate() override;
};
} // namespace EvoEngine