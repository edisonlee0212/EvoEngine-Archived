#include "Bound.hpp"
#include "Mesh.hpp"
#include "Transform.hpp"
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

Ray::Ray(const glm::vec3& start, const glm::vec3& end)
{
	m_start = start;
	m_direction = glm::normalize(end - start);
	m_length = glm::distance(start, end);
}

Ray::Ray(const glm::vec3& start, const glm::vec3& direction, float length)
{
	m_start = start;
	m_direction = direction;
	m_length = length;
}

bool Ray::Intersect(const glm::vec3& position, float radius) const
{
	const glm::vec3 rayEnd = m_start + m_direction * m_length;
	const auto cp = glm::closestPointOnLine(position, m_start, rayEnd);
	if (cp == m_start || cp == rayEnd)
		return false;
	return glm::distance(cp, position) <= radius;
}

bool Ray::Intersect(const glm::mat4& transform, const Bound& bound) const
{
	float tMin = 0.0f;
	float tMax = 100000.0f;
	GlobalTransform temp;
	temp.m_value = transform;
	glm::vec3 scale = temp.GetScale();
	temp.SetScale(glm::vec3(1.0f));
	glm::mat4 model = temp.m_value;

	glm::vec3 OBBWorldSpace(model[3].x, model[3].y, model[3].z);

	glm::vec3 delta = OBBWorldSpace - m_start;
	glm::vec3 AABBMin = scale * (bound.m_min);
	glm::vec3 AABBMax = scale * (bound.m_max);
	// Test intersection with the 2 planes perpendicular to the OBB's X axis
	{
		glm::vec3 xAxis(model[0].x, model[0].y, model[0].z);

		float e = glm::dot(xAxis, delta);
		float f = glm::dot(m_direction, xAxis);

		if (fabs(f) > 0.001f)
		{ // Standard case

			float t1 = (e + AABBMin.x) / f; // Intersection with the "left" plane
			float t2 = (e + AABBMax.x) / f; // Intersection with the "right" plane
			// t1 and t2 now contain distances betwen ray origin and ray-plane intersections

			// We want t1 to represent the nearest intersection,
			// so if it's not the case, invert t1 and t2
			if (t1 > t2)
			{
				float w = t1;
				t1 = t2;
				t2 = w; // swap t1 and t2
			}

			// tMax is the nearest "far" intersection (amongst the X,Y and Z planes pairs)
			if (t2 < tMax)
				tMax = t2;
			// tMin is the farthest "near" intersection (amongst the X,Y and Z planes pairs)
			if (t1 > tMin)
				tMin = t1;

			// And here's the trick :
			// If "far" is closer than "near", then there is NO intersection.
			// See the images in the tutorials for the visual explanation.
			if (tMax < tMin)
				return false;
		}
		else
		{ // Rare case : the ray is almost parallel to the planes, so they don't have any "intersection"
			if (-e + AABBMin.x > 0.0f || -e + AABBMax.x < 0.0f)
				return false;
		}
	}

	// Test intersection with the 2 planes perpendicular to the OBB's Y axis
	// Exactly the same thing than above.
	{
		glm::vec3 yAxis(model[1].x, model[1].y, model[1].z);
		float e = glm::dot(yAxis, delta);
		float f = glm::dot(m_direction, yAxis);

		if (fabs(f) > 0.001f)
		{

			float t1 = (e + AABBMin.y) / f;
			float t2 = (e + AABBMax.y) / f;

			if (t1 > t2)
			{
				float w = t1;
				t1 = t2;
				t2 = w;
			}

			if (t2 < tMax)
				tMax = t2;
			if (t1 > tMin)
				tMin = t1;
			if (tMin > tMax)
				return false;
		}
		else
		{
			if (-e + AABBMin.y > 0.0f || -e + AABBMax.y < 0.0f)
				return false;
		}
	}

	// Test intersection with the 2 planes perpendicular to the OBB's Z axis
	// Exactly the same thing than above.
	{
		glm::vec3 zAxis(model[2].x, model[2].y, model[2].z);
		float e = glm::dot(zAxis, delta);
		float f = glm::dot(m_direction, zAxis);

		if (fabs(f) > 0.001f)
		{

			float t1 = (e + AABBMin.z) / f;
			float t2 = (e + AABBMax.z) / f;

			if (t1 > t2)
			{
				float w = t1;
				t1 = t2;
				t2 = w;
			}

			if (t2 < tMax)
				tMax = t2;
			if (t1 > tMin)
				tMin = t1;
			if (tMin > tMax)
				return false;
		}
		else
		{
			if (-e + AABBMin.z > 0.0f || -e + AABBMax.z < 0.0f)
				return false;
		}
	}
	return true;
}

glm::vec3 Ray::GetEnd() const
{
	return m_start + m_direction * m_length;
}

glm::vec3 Ray::ClosestPointOnLine(const glm::vec3& point, const glm::vec3& a, const glm::vec3& b)
{
	const float lineLength = distance(a, b);
	const glm::vec3 vector = point - a;
	const glm::vec3 lineDirection = (b - a) / lineLength;

	// Project Vector to LineDirection to get the distance of point from a
	const float distance = dot(vector, lineDirection);
	return a + lineDirection * distance;
}

Plane::Plane(const glm::vec4& param)
{
	m_a = param.x;
	m_b = param.y;
	m_c = param.z;
	m_d = param.w;
	Normalize();
}

Plane::Plane(const glm::vec3& normal, const float distance)
{
	const auto n = glm::normalize(normal);
	m_a = n.x;
	m_b = n.y;
	m_c = n.z;
	m_d = -(n.x * (n * distance).x + n.y * (n * distance).y + n.z * (n * distance).z);
}

Plane::Plane() : m_a(0), m_b(0), m_c(0), m_d(0)
{
}

void Plane::Normalize()
{
	const float mag = glm::sqrt(m_a * m_a + m_b * m_b + m_c * m_c);
	m_a /= mag;
	m_b /= mag;
	m_c /= mag;
	m_d /= mag;
}

float Plane::CalculateTriangleDistance(const std::vector<Vertex>& vertices, const glm::uvec3& triangle) const
{
	const auto& a = vertices[triangle.x].m_position;
	const auto& b = vertices[triangle.y].m_position;
	const auto& c = vertices[triangle.z].m_position;

	const auto centroid = 
		glm::vec3(
			(a.x + b.x + c.x) / 3,
			(a.y + b.y + c.y) / 3,
			(a.z + b.z + c.z) / 3
			);
	return CalculatePointDistance(centroid);
}

float Plane::CalculateTriangleMaxDistance(const std::vector<Vertex>& vertices, const glm::uvec3& triangle) const
{
	const auto& p0 = vertices[triangle.x].m_position;
	const auto& p1 = vertices[triangle.y].m_position;
	const auto& p2 = vertices[triangle.z].m_position;

	const auto d0 = CalculatePointDistance(p0);
	const auto d1 = CalculatePointDistance(p1);
	const auto d2 = CalculatePointDistance(p2);

	return glm::max(d0, glm::max(d1, d2));
}

float Plane::CalculatePointDistance(const glm::vec3& point) const
{
	return glm::abs(m_a * point.x + m_b * point.y + m_c * point.z + m_d) /
		glm::sqrt(m_a * m_a + m_b * m_b + m_c * m_c);
}

float Plane::CalculateTriangleMinDistance(const std::vector<Vertex>& vertices, const glm::uvec3& triangle) const
{
	const auto& p0 = vertices[triangle.x].m_position;
	const auto& p1 = vertices[triangle.y].m_position;
	const auto& p2 = vertices[triangle.z].m_position;

	const auto d0 = CalculatePointDistance(p0);
	const auto d1 = CalculatePointDistance(p1);
	const auto d2 = CalculatePointDistance(p2);

	return glm::min(d0, glm::min(d1, d2));
}
