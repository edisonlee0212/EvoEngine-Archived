#pragma once
#include "ISingleton.hpp"
#include "ThreadPool.hpp"
namespace EvoEngine
{
class Jobs final : ISingleton<Jobs>
{
    ThreadPool m_workers;

  public:
    static void ResizeWorkers(size_t size);
    static ThreadPool &Workers();
    static void Initialize();
    static void ParallelFor(size_t size, const std::function<void(unsigned i)>& func);
    static void ParallelFor(size_t size, const std::function<void(unsigned i)> &func, std::vector<std::shared_future<void>>& results);
    static void ParallelFor(size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, std::vector<std::shared_future<void>>& results);
};
} // namespace EvoEngine
