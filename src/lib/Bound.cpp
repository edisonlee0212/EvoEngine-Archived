#include "Bound.hpp"
using namespace EvoEngine;

glm::vec3 Bound::Size() const
{
    return (m_max - m_min) / 2.0f;
}

glm::vec3 Bound::Center() const
{
    return (m_max + m_min) / 2.0f;
}

bool Bound::InBound(const glm::vec3& position) const
{
    glm::vec3 center = (m_min + m_max) / 2.0f;
    glm::vec3 size = (m_max - m_min) / 2.0f;
    if (glm::abs(position.x - center.x) > size.x)
        return false;
    if (glm::abs(position.y - center.y) > size.y)
        return false;
    if (glm::abs(position.z - center.z) > size.z)
        return false;
    return true;
}

void Bound::ApplyTransform(const glm::mat4& transform)
{
    std::vector<glm::vec3> corners;
    PopulateCorners(corners);
    m_min = glm::vec3(FLT_MAX);
    m_max = glm::vec3(FLT_MIN);

    // Transform all of the corners, and keep track of the greatest and least
    // values we see on each coordinate axis.
    for (int i = 0; i < 8; i++)
    {
        glm::vec3 transformed = transform * glm::vec4(corners[i], 1.0f);
        m_min = (glm::min)(m_min, transformed);
        m_max = (glm::max)(m_max, transformed);
    }
}

void Bound::PopulateCorners(std::vector<glm::vec3>& corners) const
{
    corners.resize(8);
    corners[0] = m_min;
    corners[1] = glm::vec3(m_min.x, m_min.y, m_max.z);
    corners[2] = glm::vec3(m_min.x, m_max.y, m_min.z);
    corners[3] = glm::vec3(m_max.x, m_min.y, m_min.z);
    corners[4] = glm::vec3(m_min.x, m_max.y, m_max.z);
    corners[5] = glm::vec3(m_max.x, m_min.y, m_max.z);
    corners[6] = glm::vec3(m_max.x, m_max.y, m_min.z);
    corners[7] = m_max;
}