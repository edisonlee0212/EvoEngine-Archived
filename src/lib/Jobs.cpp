#include "Jobs.hpp"

#include "Console.hpp"
using namespace evo_engine;

size_t Jobs::GetWorkerSize()
{
	const auto& jobs = GetInstance();
	return jobs.m_jobSystem.GetWorkerSize();
}

void Jobs::Initialize(const size_t workerSize)
{
	auto& jobs = GetInstance();
	jobs.m_jobSystem.ResizeWorker(workerSize);
}

void Jobs::RunParallelFor(const size_t size, std::function<void(unsigned i)>&& func, size_t workerSize)
{
	auto& jobs = GetInstance();
	if (workerSize == 0) workerSize = GetWorkerSize();
	const auto threadLoad = size / workerSize;
	const auto loadReminder = size % workerSize;
	std::vector<JobHandle> jobHandles;
	for (int threadIndex = 0; threadIndex < workerSize; threadIndex++)
	{
		const auto work = [func, threadIndex, workerSize, threadLoad, loadReminder]()
			{
				for (unsigned i = threadIndex * threadLoad; i < (threadIndex + 1) * threadLoad; i++)
				{
					func(i);
				}
				if (threadIndex < loadReminder)
				{
					const unsigned i = threadIndex + workerSize * threadLoad;
					func(i);
				}
			};
		jobHandles.emplace_back(jobs.m_jobSystem.PushJob({}, std::forward<std::function<void()>>(work)));
	}
	Wait(Combine(jobHandles));
}

void Jobs::RunParallelFor(const size_t size, std::function<void(unsigned i, unsigned threadIndex)>&& func, size_t workerSize)
{
	auto& jobs = GetInstance();
	if (workerSize == 0) workerSize = GetWorkerSize();
	const auto threadLoad = size / workerSize;
	const auto loadReminder = size % workerSize;
	std::vector<JobHandle> jobHandles;
	for (int threadIndex = 0; threadIndex < workerSize; threadIndex++)
	{
		const auto work = [func, threadIndex, workerSize, threadLoad, loadReminder]()
			{
				for (unsigned i = threadIndex * threadLoad; i < (threadIndex + 1) * threadLoad; i++)
				{
					func(i, threadIndex);
				}
				if (threadIndex < loadReminder)
				{
					const unsigned i = threadIndex + workerSize * threadLoad;
					func(i, threadIndex);
				}
			};
		jobHandles.emplace_back(jobs.m_jobSystem.PushJob({}, std::forward<std::function<void()>>(work)));
	}
	Wait(Combine(jobHandles));
}

JobHandle Jobs::ScheduleParallelFor(const size_t size, std::function<void(unsigned i)>&& func, size_t workerSize)
{
	auto& jobs = GetInstance();
	if (workerSize == 0) workerSize = GetWorkerSize();
	const auto threadLoad = size / workerSize;
	const auto loadReminder = size % workerSize;
	std::vector<JobHandle> jobHandles;
	for (int threadIndex = 0; threadIndex < workerSize; threadIndex++)
	{
		const auto work = [func, threadIndex, workerSize, threadLoad, loadReminder]()
			{
				for (unsigned i = threadIndex * threadLoad; i < (threadIndex + 1) * threadLoad; i++)
				{
					func(i);
				}
				if (threadIndex < loadReminder)
				{
					const unsigned i = threadIndex + workerSize * threadLoad;
					func(i);
				}
			};
		jobHandles.emplace_back(jobs.m_jobSystem.PushJob({}, std::forward<std::function<void()>>(work)));
	}
	return Combine(jobHandles);
}

JobHandle Jobs::ScheduleParallelFor(const size_t size, std::function<void(unsigned i, unsigned threadIndex)>&& func,
	size_t workerSize)
{
	auto& jobs = GetInstance();
	if (workerSize == 0) workerSize = GetWorkerSize();
	const auto threadLoad = size / workerSize;
	const auto loadReminder = size % workerSize;
	std::vector<JobHandle> jobHandles;
	for (int threadIndex = 0; threadIndex < workerSize; threadIndex++)
	{
		const auto work = [func, threadIndex, workerSize, threadLoad, loadReminder]()
			{
				for (unsigned i = threadIndex * threadLoad; i < (threadIndex + 1) * threadLoad; i++)
				{
					func(i, threadIndex);
				}
				if (threadIndex < loadReminder)
				{
					const unsigned i = threadIndex + workerSize * threadLoad;
					func(i, threadIndex);
				}
			};
		jobHandles.emplace_back(jobs.m_jobSystem.PushJob({}, std::forward<std::function<void()>>(work)));
	}
	return Combine(jobHandles);
}

void Jobs::RunParallelFor(const std::vector<JobHandle>& dependencies, const size_t size,
	std::function<void(unsigned i)>&& func, size_t workerSize)
{
	auto& jobs = GetInstance();
	if (workerSize == 0) workerSize = GetWorkerSize();
	const auto threadLoad = size / workerSize;
	const auto loadReminder = size % workerSize;
	std::vector<JobHandle> jobHandles;
	for (int threadIndex = 0; threadIndex < workerSize; threadIndex++)
	{
		const auto work = [func, threadIndex, workerSize, threadLoad, loadReminder]()
			{
				for (unsigned i = threadIndex * threadLoad; i < (threadIndex + 1) * threadLoad; i++)
				{
					func(i);
				}
				if (threadIndex < loadReminder)
				{
					const unsigned i = threadIndex + workerSize * threadLoad;
					func(i);
				}
			};
		jobHandles.emplace_back(jobs.m_jobSystem.PushJob(dependencies, std::forward<std::function<void()>>(work)));
	}
	Wait(Combine(jobHandles));
}

void Jobs::RunParallelFor(const std::vector<JobHandle>& dependencies, const size_t size,
	std::function<void(unsigned i, unsigned threadIndex)>&& func, size_t workerSize)
{
	auto& jobs = GetInstance();
	if (workerSize == 0) workerSize = GetWorkerSize();
	const auto threadLoad = size / workerSize;
	const auto loadReminder = size % workerSize;
	std::vector<JobHandle> jobHandles;
	for (int threadIndex = 0; threadIndex < workerSize; threadIndex++)
	{
		const auto work = [func, threadIndex, workerSize, threadLoad, loadReminder]()
			{
				for (unsigned i = threadIndex * threadLoad; i < (threadIndex + 1) * threadLoad; i++)
				{
					func(i, threadIndex);
				}
				if (threadIndex < loadReminder)
				{
					const unsigned i = threadIndex + workerSize * threadLoad;
					func(i, threadIndex);
				}
			};
		jobHandles.emplace_back(jobs.m_jobSystem.PushJob(dependencies, std::forward<std::function<void()>>(work)));
	}
	Wait(Combine(jobHandles));
}

JobHandle Jobs::ScheduleParallelFor(const std::vector<JobHandle>& dependencies, const size_t size,
	std::function<void(unsigned i)>&& func, size_t workerSize)
{
	auto& jobs = GetInstance();
	if (workerSize == 0) workerSize = GetWorkerSize();
	const auto threadLoad = size / workerSize;
	const auto loadReminder = size % workerSize;
	std::vector<JobHandle> jobHandles;
	for (int threadIndex = 0; threadIndex < workerSize; threadIndex++)
	{
		const auto work = [func, threadIndex, workerSize, threadLoad, loadReminder]()
			{
				for (unsigned i = threadIndex * threadLoad; i < (threadIndex + 1) * threadLoad; i++)
				{
					func(i);
				}
				if (threadIndex < loadReminder)
				{
					const unsigned i = threadIndex + workerSize * threadLoad;
					func(i);
				}
			};
		jobHandles.emplace_back(jobs.m_jobSystem.PushJob(dependencies, std::forward<std::function<void()>>(work)));
	}
	return Combine(jobHandles);
}

JobHandle Jobs::ScheduleParallelFor(const std::vector<JobHandle>& dependencies, const size_t size,
	std::function<void(unsigned i, unsigned threadIndex)>&& func, size_t workerSize)
{
	auto& jobs = GetInstance();
	if (workerSize == 0) workerSize = GetWorkerSize();
	const auto threadLoad = size / workerSize;
	const auto loadReminder = size % workerSize;
	std::vector<JobHandle> jobHandles;
	for (int threadIndex = 0; threadIndex < workerSize; threadIndex++)
	{
		const auto work = [func, threadIndex, workerSize, threadLoad, loadReminder]()
			{
				for (unsigned i = threadIndex * threadLoad; i < (threadIndex + 1) * threadLoad; i++)
				{
					func(i, threadIndex);
				}
				if (threadIndex < loadReminder)
				{
					const unsigned i = threadIndex + workerSize * threadLoad;
					func(i, threadIndex);
				}
			};
		jobHandles.emplace_back(jobs.m_jobSystem.PushJob(dependencies, std::forward<std::function<void()>>(work)));
	}
	return Combine(jobHandles);
}

JobHandle Jobs::Run(const std::vector<JobHandle>& dependencies, std::function<void()>&& func)
{
	auto& jobs = GetInstance();
	return jobs.m_jobSystem.PushJob(dependencies, std::forward<std::function<void()>>(func));
}

JobHandle Jobs::Run(std::function<void()>&& func)
{
	auto& jobs = GetInstance();
	return jobs.m_jobSystem.PushJob({}, std::forward<std::function<void()>>(func));
}

JobHandle Jobs::Combine(const std::vector<JobHandle>& dependencies)
{
	auto& jobs = GetInstance();
	return jobs.m_jobSystem.PushJob(dependencies, []() {});
}

void Jobs::Execute(const JobHandle& jobHandle)
{
	auto& jobs = GetInstance();
	if (!jobHandle.Valid()) return;
	jobs.m_jobSystem.ExecuteJob(jobHandle);
}

void Jobs::Wait(const JobHandle& jobHandle)
{
	auto& jobs = GetInstance();
	if (!jobHandle.Valid()) return;
	jobs.m_jobSystem.ExecuteJob(jobHandle);
	jobs.m_jobSystem.Wait(jobHandle);
}
