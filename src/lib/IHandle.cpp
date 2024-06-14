#include <IHandle.hpp>
using namespace evo_engine;

static std::random_device s_RandomDevice;
static std::mt19937_64 eng(s_RandomDevice());
static std::uniform_int_distribution<uint64_t> s_UniformDistribution;

Handle::Handle() : m_value(s_UniformDistribution(eng))
{
}

Handle::Handle(uint64_t value) : m_value(value)
{
}

Handle::Handle(const Handle &other) : m_value(other.m_value)
{
}