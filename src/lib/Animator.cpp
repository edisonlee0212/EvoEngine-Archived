#include "Application.hpp"
#include "Animator.hpp"
#include "EditorLayer.hpp"
#include "ClassRegistry.hpp"
#include "Time.hpp"
using namespace EvoEngine;

void Animator::Setup()
{
	if (const auto animation = m_animation.Get<Animation>())
    {
        m_boneSize = animation->m_boneSize;
        if (animation->UnsafeGetRootBone() && m_boneSize != 0)
        {
            m_transformChain.resize(m_boneSize);
            m_names.resize(m_boneSize);
            m_bones.resize(m_boneSize);
            BoneSetter(animation->UnsafeGetRootBone());
            m_offsetMatrices.resize(m_boneSize);
            for (const auto &i : m_bones)
                m_offsetMatrices[i->m_index] = i->m_offsetMatrix.m_value;
            if (!animation->IsEmpty()) {
                m_currentActivatedAnimation = animation->GetFirstAvailableAnimationName();
                m_currentAnimationTime = 0.0f;
            }
        }
    }
}
void Animator::OnDestroy()
{
    m_transformChain.clear();
    m_offsetMatrices.clear();
    m_names.clear();
    m_animation.Clear();
    m_bones.clear();
}
void Animator::Setup(const std::shared_ptr<Animation> &targetAnimation)
{
    m_animation.Set<Animation>(targetAnimation);
    Setup();
}

void Animator::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
    auto animation = m_animation.Get<Animation>();
    const Animation *previous = animation.get();
    editorLayer->DragAndDropButton<Animation>(m_animation, "Animation");
    if (previous != animation.get() && animation)
    {
        Setup(animation);
        animation = m_animation.Get<Animation>();
    }
    if (animation)
    {
        if (m_boneSize != 0)
        {
            if (!animation->HasAnimation(m_currentActivatedAnimation))
            {
                m_currentActivatedAnimation = animation->GetFirstAvailableAnimationName();
                m_currentAnimationTime = 0.0f;
            }
            if (ImGui::BeginCombo(
                    "Animations##Animator",
                    m_currentActivatedAnimation
                        .c_str())) // The second parameter is the label previewed before opening the combo.
            {
                for (auto &i : animation->UnsafeGetAnimationLengths())
                {
                    const bool selected =
                        m_currentActivatedAnimation ==
                        i.first; // You can store your selection however you want, outside or inside your objects
                    if (ImGui::Selectable(i.first.c_str(), selected))
                    {
                        m_currentActivatedAnimation = i.first;
                        m_currentAnimationTime = 0.0f;
                    }
                    if (selected)
                    {
                        ImGui::SetItemDefaultFocus(); // You may set the initial focus when opening the combo (scrolling
                                                      // + for keyboard navigation support)
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::SliderFloat(
                "Animation time",
                &m_currentAnimationTime,
                0.0f,
                animation->GetAnimationLength(m_currentActivatedAnimation));
            /*
            static bool autoPlay = false;
            if(!Application::IsPlaying()) ImGui::Checkbox("AutoPlay", &autoPlay);
            static std::weak_ptr<Animator> previousAnimatorPtr;
            static std::string lastAnimationName = {};
            static float lastAnimationTime = 0;
            if (autoPlay) {
                static float autoPlaySpeed = 30;
                if(!previousAnimatorPtr.expired() && previousAnimatorPtr.lock().get() != this)
                {
	                const auto previousAnimator = previousAnimatorPtr.lock();
                    previousAnimator->Animate(lastAnimationName, lastAnimationTime);
                    if (animation->HasAnimation(m_currentActivatedAnimation)) {
                        lastAnimationName = m_currentActivatedAnimation;
                        lastAnimationTime = m_currentAnimationTime;
                    }else
                    {
                        lastAnimationName = {};
                        lastAnimationTime = 0.0f;
                    }
                    previousAnimatorPtr = std::dynamic_pointer_cast<Animator>(ProjectManager::GetAsset(GetHandle()));
                }

                ImGui::DragFloat("AutoPlay Speed", &autoPlaySpeed, 1.0f);
                if (animation->HasAnimation(m_currentActivatedAnimation)) {
                    m_currentAnimationTime += Time::DeltaTime() * autoPlaySpeed;
                    const float animationLength = animation->GetAnimationLength(m_currentActivatedAnimation);
                    if (m_currentAnimationTime > animationLength)
                        m_currentAnimationTime =
                        glm::mod(m_currentAnimationTime, animationLength);
                }
            }else if(!previousAnimatorPtr.expired() && previousAnimatorPtr.lock().get() == this)
            {
                const auto previousAnimator = previousAnimatorPtr.lock();
                previousAnimator->Animate(lastAnimationName, lastAnimationTime);
                if (animation->HasAnimation(m_currentActivatedAnimation)) {
                    lastAnimationName = m_currentActivatedAnimation;
                    lastAnimationTime = m_currentAnimationTime;
                }
                else
                {
                    lastAnimationName = {};
                    lastAnimationTime = 0.0f;
                }
                previousAnimatorPtr.reset();
            }*/
            
        }
    }
}
float Animator::GetCurrentAnimationTimePoint() const
{
    return m_currentAnimationTime;
}

std::string Animator::GetCurrentAnimationName()
{
    return m_currentActivatedAnimation;
}

void Animator::Animate(const std::string& animationName, const float time)
{
	const auto animation = m_animation.Get<Animation>();
    if (!animation)
        return;
	const auto search = animation->UnsafeGetAnimationLengths().find(animationName);
    if(search == animation->UnsafeGetAnimationLengths().end()){
        EVOENGINE_ERROR("Animation not found!");
        return;
    }
    m_currentActivatedAnimation = animationName;
    m_currentAnimationTime =
        glm::mod(time, search->second);
}
void Animator::Animate(const float time)
{
	const auto animation = m_animation.Get<Animation>();
    if (!animation)
        return;
    m_currentAnimationTime =
        glm::mod(time, animation->GetAnimationLength(m_currentActivatedAnimation));
}
void Animator::Apply()
{
    if (const auto animation = m_animation.Get<Animation>(); !animation->IsEmpty())
    {
        if (!animation->HasAnimation(m_currentActivatedAnimation))
        {
            m_currentActivatedAnimation = animation->GetFirstAvailableAnimationName();
            m_currentAnimationTime = 0.0f;
        }
        if (const auto owner = GetOwner(); owner.GetIndex() != 0)
        {
            animation->Animate(m_currentActivatedAnimation, m_currentAnimationTime, glm::mat4(1.0f), m_transformChain);
            ApplyOffsetMatrices();
        }
    }
}

void Animator::BoneSetter(const std::shared_ptr<Bone> &boneWalker)
{
    m_names[boneWalker->m_index] = boneWalker->m_name;
    m_bones[boneWalker->m_index] = boneWalker;
    for (auto &i : boneWalker->m_children)
    {
        BoneSetter(i);
    }
}

void Animator::Setup(const std::vector<std::string> &name, const std::vector<glm::mat4> &offsetMatrices)
{
    m_bones.clear();
    m_boneSize = 0;
    m_transformChain.resize(offsetMatrices.size());
    m_names = name;
    m_offsetMatrices = offsetMatrices;
}

void Animator::ApplyOffsetMatrices()
{
    for (int i = 0; i < m_transformChain.size(); i++)
    {
        m_transformChain[i] *= m_offsetMatrices[i];
    }
}

glm::mat4 Animator::GetReverseTransform(const int &index, const Entity& entity)
{
    return m_transformChain[index] * glm::inverse(m_bones[index]->m_offsetMatrix.m_value);
}
void Animator::PostCloneAction(const std::shared_ptr<IPrivateComponent> &target)
{
}

void Animator::CollectAssetRef(std::vector<AssetRef> &list)
{
    list.push_back(m_animation);
}

void Animator::Serialize(YAML::Emitter &out)
{
    if (m_animation.Get<Animation>())
    {
        m_animation.Save("m_animation", out);
        out << YAML::Key << "m_currentActivatedAnimation" << YAML::Value << m_currentActivatedAnimation;
        out << YAML::Key << "m_currentAnimationTime" << YAML::Value << m_currentAnimationTime;
    }
    if (!m_transformChain.empty())
    {
        out << YAML::Key << "m_transformChain" << YAML::Value
            << YAML::Binary(
                   reinterpret_cast<const unsigned char*>(m_transformChain.data()), m_transformChain.size() * sizeof(glm::mat4));
    }
    if (!m_offsetMatrices.empty())
    {
        out << YAML::Key << "m_offsetMatrices" << YAML::Value
            << YAML::Binary(
                   reinterpret_cast<const unsigned char*>(m_offsetMatrices.data()), m_offsetMatrices.size() * sizeof(glm::mat4));
    }
    if (!m_names.empty())
    {
        out << YAML::Key << "m_names" << YAML::Value << YAML::BeginSeq;
        for (int i = 0; i < m_names.size(); i++)
        {
            out << YAML::BeginMap;
            out << YAML::Key << "Name" << YAML::Value << m_names[i];
            out << YAML::EndMap;
        }
        out << YAML::EndSeq;
    }
}

void Animator::Deserialize(const YAML::Node &in)
{
    m_animation.Load("m_animation", in);
    if (m_animation.Get<Animation>())
    {
        m_currentActivatedAnimation = in["m_currentActivatedAnimation"].as<std::string>();
        m_currentAnimationTime = in["m_currentAnimationTime"].as<float>();
        Setup();
    }
    if (in["m_transformChain"])
    {
	    const auto chains = in["m_transformChain"].as<YAML::Binary>();
        m_transformChain.resize(chains.size() / sizeof(glm::mat4));
        std::memcpy(m_transformChain.data(), chains.data(), chains.size());
    }
    if (in["m_offsetMatrices"])
    {
	    const auto matrices = in["m_offsetMatrices"].as<YAML::Binary>();
        m_offsetMatrices.resize(matrices.size() / sizeof(glm::mat4));
        std::memcpy(m_offsetMatrices.data(), matrices.data(), matrices.size());
    }
    if (in["m_names"])
    {
        for (const auto &i : in["m_names"])
        {
            m_names.push_back(i["Name"].as<std::string>());
        }
    }
}

std::shared_ptr<Animation> Animator::GetAnimation()
{
    return m_animation.Get<Animation>();
}