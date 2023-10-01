#include "Engine/Core/Jobs.hpp"
using namespace EvoEngine;

void Jobs::ResizeWorkers(const size_t size)
{
    GetInstance().m_workers.FinishAll(true);
    GetInstance().m_workers.Resize(size);
}

ThreadPool &Jobs::Workers()
{
    return GetInstance().m_workers;
}

void Jobs::Initialize()
{
    Workers().Resize(std::thread::hardware_concurrency() - 1);
}

void Jobs::ParallelFor(const size_t size, const std::function<void(unsigned i)>& func)
{
    auto& workers = GetInstance().m_workers;
    const auto threadSize = workers.Size();
    const auto threadLoad = size / threadSize;
    const auto loadReminder = size % threadSize;
    std::vector<std::shared_future<void>> results;
    results.reserve(threadSize);
    for (int threadIndex = 0; threadIndex < threadSize; threadIndex++)
    {
        results.push_back(workers
            .Push([=](int id) {
                for (unsigned i = threadIndex * threadLoad; i < (threadIndex + 1) * threadLoad; i++)
                {
                    func(i);
                }
                if (threadIndex < loadReminder)
                {
                    const unsigned i = threadIndex + threadSize * threadLoad;
                    func(i);
                }
                })
            .share());
    }
    for (const auto& i : results) i.wait();
}

void Jobs::ParallelFor(
	const size_t size, const std::function<void(unsigned i)> &func, std::vector<std::shared_future<void>> &results)
{
    auto &workers = GetInstance().m_workers;
    const auto threadSize = workers.Size();
    const auto threadLoad = size / threadSize;
    const auto loadReminder = size % threadSize;
    results.reserve(results.size() + threadSize);
    for (int threadIndex = 0; threadIndex < threadSize; threadIndex++)
    {
        results.push_back(workers
                              .Push([=](int id) {
                                  for (unsigned i = threadIndex * threadLoad; i < (threadIndex + 1) * threadLoad; i++)
                                  {
                                      func(i);
                                  }
                                  if (threadIndex < loadReminder)
                                  {
                                      const unsigned i = threadIndex + threadSize * threadLoad;
                                      func(i);
                                  }
                              })
                              .share());
    }
}

void Jobs::ParallelFor(const size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func)
{
    auto& workers = GetInstance().m_workers;
    const auto threadSize = workers.Size();
    const auto threadLoad = size / threadSize;
    const auto loadReminder = size % threadSize;
    std::vector<std::shared_future<void>> results;
    results.reserve(threadSize);
    for (int threadIndex = 0; threadIndex < threadSize; threadIndex++)
    {
        results.push_back(workers
            .Push([=](const int id) {
                for (unsigned i = threadIndex * threadLoad; i < (threadIndex + 1) * threadLoad; i++)
                {
                    func(i, id);
                }
                if (threadIndex < loadReminder)
                {
                    const unsigned i = threadIndex + threadSize * threadLoad;
                    func(i, id);
                }
                })
            .share());
    }
    for (const auto& i : results) i.wait();
}

void Jobs::ParallelFor(const size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func,
                       std::vector<std::shared_future<void>>& results)
{
    auto& workers = GetInstance().m_workers;
    const auto threadSize = workers.Size();
    const auto threadLoad = size / threadSize;
    const auto loadReminder = size % threadSize;
    results.reserve(results.size() + threadSize);
    for (int threadIndex = 0; threadIndex < threadSize; threadIndex++)
    {
        results.push_back(workers
            .Push([=](const int id) {
                for (unsigned i = threadIndex * threadLoad; i < (threadIndex + 1) * threadLoad; i++)
                {
                    func(i, id);
                }
        if (threadIndex < loadReminder)
        {
            const unsigned i = threadIndex + threadSize * threadLoad;
            func(i, id);
        }
                })
            .share());
    }
}

void Jobs::Parallel(const std::function<void(unsigned threadIndex)>& func)
{
    auto& workers = GetInstance().m_workers;
    const auto threadSize = workers.Size();
    std::vector<std::shared_future<void>> results;
    results.reserve(threadSize);
    for (int threadIndex = 0; threadIndex < threadSize; threadIndex++)
    {
        results.push_back(workers
            .Push([=](const int id) {
                func(id);
                })
            .share());
    }
    for (const auto& i : results) i.wait();
}

void Jobs::Parallel(const std::function<void(unsigned threadIndex)>& func,
	std::vector<std::shared_future<void>>& results)
{
    auto& workers = GetInstance().m_workers;
    const auto threadSize = workers.Size();
    results.reserve(results.size() + threadSize);
    for (int threadIndex = 0; threadIndex < threadSize; threadIndex++)
    {
        results.push_back(workers
            .Push([=](const int id) {
                func(id);
                })
            .share());
    }
    for (const auto& i : results) i.wait();
}

std::shared_future<void> Jobs::AddTask(const std::function<void(unsigned threadIndex)>& func)
{
    auto& workers = GetInstance().m_workers;
    return workers.Push([=](const int id) {
        func(id);
        }).share();
}
