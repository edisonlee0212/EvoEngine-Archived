#pragma once
#include "IDataComponent.hpp"

namespace EvoEngine
{
	struct Bound
	{
		glm::vec3 m_min = glm::vec3(FLT_MAX);
		glm::vec3 m_max = glm::vec3(FLT_MIN);
		[[nodiscard]] glm::vec3 Size() const;
		[[nodiscard]] glm::vec3 Center() const;
		[[nodiscard]] bool InBound(const glm::vec3& position) const;
		void ApplyTransform(const glm::mat4& transform);
		void PopulateCorners(std::vector<glm::vec3>& corners) const;
	};
	struct Ray : IDataComponent
	{
		glm::vec3 m_start;
		glm::vec3 m_direction;
		float m_length;
		Ray() = default;
		Ray(glm::vec3 start, glm::vec3 end);
		Ray(glm::vec3 start, glm::vec3 direction, float length);
		[[nodiscard]] bool Intersect(const glm::vec3& position, float radius) const;
		[[nodiscard]] bool Intersect(const glm::mat4& transform, const Bound& bound) const;
		[[nodiscard]] glm::vec3 GetEnd() const;
		[[nodiscard]] static glm::vec3 ClosestPointOnLine(const glm::vec3& point, const glm::vec3& a, const glm::vec3& b);
	};
	struct Plane
	{
		float m_a, m_b, m_c, m_d;
		Plane();
		void Normalize();
	};
}
