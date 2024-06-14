#pragma once
#include "Application.hpp"

#include "IPrivateComponent.hpp"

namespace evo_engine{
class UnknownPrivateComponent : public IPrivateComponent
{
    std::string m_originalTypeName {};
    friend class Scene;
    friend class PrivateComponentHolder;
  public:
    bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
};

}
