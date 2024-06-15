#include <IHandle.hpp>
using namespace evo_engine;

static std::random_device s_RandomDevice;
static std::mt19937_64 eng(s_RandomDevice());
static std::uniform_int_distribution<uint64_t> s_UniformDistribution;

Handle::Handle() : value_(s_UniformDistribution(eng))
{
}

Handle::Handle(uint64_t value) : value_(value)
{
}

Handle::Handle(const Handle &other) : value_(other.value_)
{
}