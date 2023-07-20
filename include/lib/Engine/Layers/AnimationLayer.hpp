#pragma once
#include "ILayer.hpp"
namespace EvoEngine
{
class AnimationLayer : public ILayer
{
  private:
    void PreUpdate() override;
};
} // namespace EvoEngine