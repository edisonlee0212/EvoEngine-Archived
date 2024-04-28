#pragma once
#include "ISingleton.hpp"
#include "JobSystem.hpp"
namespace EvoEngine
{
	class Jobs final : ISingleton<Jobs>
	{
		JobSystem m_jobSystem;
	public:
		static size_t GetWorkerSize();
		static void Initialize(size_t workerSize);
		static void RunParallelFor(size_t size, std::function<void(unsigned i)>&& func, size_t workerSize = 0);
		static void RunParallelFor(size_t size, std::function<void(unsigned i, unsigned workerIndex)>&& func, size_t workerSize = 0);
		static JobHandle ScheduleParallelFor(size_t size, std::function<void(unsigned i)>&& func, size_t workerSize = 0);
		static JobHandle ScheduleParallelFor(size_t size, std::function<void(unsigned i, unsigned workerIndex)>&& func, size_t workerSize = 0);

		static void RunParallelFor(const std::vector<JobHandle>& dependencies, size_t size, std::function<void(unsigned i)>&& func, size_t workerSize = 0);
		static void RunParallelFor(const std::vector<JobHandle>& dependencies, size_t size, std::function<void(unsigned i, unsigned workerIndex)>&& func, size_t workerSize = 0);
		static JobHandle ScheduleParallelFor(const std::vector<JobHandle>& dependencies, size_t size, std::function<void(unsigned i)>&& func, size_t workerSize = 0);
		static JobHandle ScheduleParallelFor(const std::vector<JobHandle>& dependencies, size_t size, std::function<void(unsigned i, unsigned workerIndex)>&& func, size_t workerSize = 0);

		static JobHandle Run(const std::vector<JobHandle>& dependencies, std::function<void()>&& func);
		static JobHandle Run(std::function<void()>&& func);
		static JobHandle Combine(const std::vector<JobHandle>& dependencies);

		static void Execute(const JobHandle& jobHandle);
		static void Wait(const JobHandle &jobHandle);
	};
} // namespace EvoEngine
