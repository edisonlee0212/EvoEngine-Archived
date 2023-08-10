#pragma once
#include "Application.hpp"

#include "IPrivateComponent.hpp"

namespace EvoEngine{
class UnknownPrivateComponent : public IPrivateComponent
{
    std::string m_originalTypeName {};
    friend class Scene;
    friend class PrivateComponentHolder;
  public:
    void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
};

}
