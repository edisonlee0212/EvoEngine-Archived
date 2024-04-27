#include "Engine/Core/Jobs.hpp"

#include "Console.hpp"
using namespace EvoEngine;


size_t JobHandle::GetVersion() const
{
	return m_version;
}

WorkerHandle JobHandle::GetWorkerHandle() const
{
	return m_workerHandle;
}

bool JobHandle::Valid() const
{
	return m_workerHandle >= 0;
}

size_t Jobs::Worker::GetVersion() const
{
	return m_version;
}

Jobs::Worker::Worker(const size_t threadSize, const WorkerHandle handle)
{
	m_handle = handle;
	m_idleThreadSize = m_threadSize = threadSize;
	m_threads.resize(m_threadSize);
	m_packagedWorks.resize(m_threadSize);
	m_taskAllocationSignal = std::vector<std::condition_variable>(m_threadSize);
	m_taskLock = std::vector<std::mutex>(m_threadSize);
	for(unsigned threadIndex = 0; threadIndex < m_threadSize; threadIndex++)
	{
		auto threadFunc = [this, threadIndex]()
			{
				bool newTaskAssigned = false;
				std::unique_ptr<std::function<void()>> work;
				while(true)
				{
					if(newTaskAssigned)
					{
						(*work)();
						work.reset();
						newTaskAssigned = false;
						{
							std::lock_guard statusLock(m_statusLock);
							m_idleThreadSize++;
							if (m_idleThreadSize == m_threadSize)
							{
								std::lock_guard taskFinishLock(m_taskFinishLock);
								m_taskFinishSignal.notify_one();
							}
						}
						
					}
					std::unique_lock taskAllocationLock(m_taskAllocationLock);
					m_taskAllocationSignal[threadIndex].wait(taskAllocationLock, [this, threadIndex, &newTaskAssigned, &work]()
					{
							std::lock_guard taskLock(m_taskLock[threadIndex]);
							if(m_packagedWorks[threadIndex])
							{
								work = std::move(m_packagedWorks[threadIndex]);
								newTaskAssigned = true;
							}
							return newTaskAssigned;
					});
				}
			};
		m_threads[threadIndex] = std::make_unique<std::thread>(threadFunc);
	}
	m_version = 0;
}

void Jobs::Worker::Execute()
{
	if (m_executing) return;
	m_idleThreadSize = 0;
	for(int threadIndex = 0; threadIndex < m_threadSize; threadIndex++)
	{
		std::unique_lock lock(m_taskAllocationLock);
		m_taskAllocationSignal[threadIndex].notify_one();
	}
	m_executing = true;
}

void Jobs::Worker::Wait()
{
	if (!m_executing) Execute();

	std::unique_lock lock(m_taskFinishLock);
	m_taskFinishSignal.wait(lock, [&]()
		{
			return m_idleThreadSize == m_threadSize;
		});
	m_dependencies.clear();
	auto& jobs = GetInstance();
	std::unique_lock workerManagementLock(jobs.m_workerManagementMutex);
	jobs.m_availableWorker.at(m_threads.size()).emplace(m_handle);
}

void Jobs::Worker::Wait(const size_t version)
{
	if (version != m_version) return;
	if (!m_executing) Execute();
	std::unique_lock lock(m_taskFinishLock);
	m_taskFinishSignal.wait(lock, [&]()
		{
			return m_idleThreadSize == m_threadSize;
		});
	m_dependencies.clear();
	auto& jobs = GetInstance();
	std::unique_lock workerManagementLock(jobs.m_workerManagementMutex);
	jobs.m_availableWorker.at(m_threads.size()).emplace(m_handle);
}

WorkerHandle Jobs::Worker::GetHandle() const
{
	return m_handle;
}

bool Jobs::Worker::Executing() const
{
	return m_executing;
}

void Jobs::Worker::ScheduleJob(std::function<void()>&& func)
{
	assert(m_threadSize == 1);
	m_packagedWorks[0] = std::make_unique<std::function<void()>>(std::forward<std::function<void()>>(func));
	m_executing = false;
	m_version++;
}

void Jobs::Worker::ScheduleJob(const std::vector<std::pair<std::shared_ptr<Worker>, size_t>>& dependencies, std::function<void()>&& func)
{
	if (!dependencies.empty()) {
		m_dependencies = dependencies;
	}

	assert(m_threadSize == 1);
	const auto work = [this, func]()
		{
			for (const auto& dependency : m_dependencies)
			{
				dependency.first->Wait(dependency.second);
			}
			func();
		};

	m_packagedWorks[0] = std::make_unique<std::function<void()>>(std::forward<std::function<void()>>(work));
	m_executing = false;
	m_version++;
}

void Jobs::Worker::ScheduleParallelJobs(const size_t size, std::function<void(unsigned i, unsigned threadIndex)>&& func)
{
	const auto threadLoad = size / m_threadSize;
	const auto loadReminder = size % m_threadSize;

	for (int threadIndex = 0; threadIndex < m_threadSize; threadIndex++)
	{
		const auto work = [this, func, threadIndex, threadLoad, loadReminder]()
			{
				for (unsigned i = threadIndex * threadLoad; i < (threadIndex + 1) * threadLoad; i++)
				{
					func(i, threadIndex);
				}
				if (threadIndex < loadReminder)
				{
					const unsigned i = threadIndex + m_threadSize * threadLoad;
					func(i, threadIndex);
				}
			};
		m_packagedWorks[threadIndex] = std::make_unique<std::function<void()>>(std::forward<std::function<void()>>(work));
	}
	m_executing = false;
	m_version++;
}

void Jobs::Worker::ScheduleParallelJobs(const std::vector<std::pair<std::shared_ptr<Worker>, size_t>>& dependencies, const size_t size,
	std::function<void(unsigned i, unsigned threadIndex)>&& func)
{
	const auto threadLoad = size / m_threadSize;
	const auto loadReminder = size % m_threadSize;
	if (!dependencies.empty()) {
		m_dependencies = dependencies;
	}
	for (int threadIndex = 0; threadIndex < m_threadSize; threadIndex++)
	{
		const auto work = [this, func, threadIndex, threadLoad, loadReminder]()
			{
				for (const auto& dependency : m_dependencies)
				{
					dependency.first->Wait(dependency.second);
				}
				for (unsigned i = threadIndex * threadLoad; i < (threadIndex + 1) * threadLoad; i++)
				{
					func(i, threadIndex);
				}
				if (threadIndex < loadReminder)
				{
					const unsigned i = threadIndex + m_threadSize * threadLoad;
					func(i, threadIndex);
				}
			};
		m_packagedWorks[threadIndex] = std::make_unique<std::function<void()>>(std::forward<std::function<void()>>(work));
	}
	m_executing = false;
	m_version++;
}

void Jobs::Worker::ScheduleParallelJobs(const size_t size, std::function<void(unsigned i)>&& func)
{
	const auto threadLoad = size / m_threadSize;
	const auto loadReminder = size % m_threadSize;
	for (int threadIndex = 0; threadIndex < m_threadSize; threadIndex++)
	{
		const auto work = [this, func, threadIndex, threadLoad, loadReminder]()
			{
				for (unsigned i = threadIndex * threadLoad; i < (threadIndex + 1) * threadLoad; i++)
				{
					func(i);
				}
				if (threadIndex < loadReminder)
				{
					const unsigned i = threadIndex + m_threadSize * threadLoad;
					func(i);
				}
			};
		m_packagedWorks[threadIndex] = std::make_unique<std::function<void()>>(std::forward<std::function<void()>>(work));
	}
	m_executing = false;
	m_version++;
}

void Jobs::Worker::ScheduleParallelJobs(const std::vector<std::pair<std::shared_ptr<Worker>, size_t>>& dependencies, const size_t size,
	std::function<void(unsigned i)>&& func)
{
	if (!dependencies.empty()) {
		m_dependencies = dependencies;
	}
	const auto threadLoad = size / m_threadSize;
	const auto loadReminder = size % m_threadSize;
	for (int threadIndex = 0; threadIndex < m_threadSize; threadIndex++)
	{
		const auto work = [this, func, threadIndex, threadLoad, loadReminder]()
			{
				for (const auto& dependency : m_dependencies)
				{
					dependency.first->Wait(dependency.second);
				}
				for (unsigned i = threadIndex * threadLoad; i < (threadIndex + 1) * threadLoad; i++)
				{
					func(i);
				}
				if (threadIndex < loadReminder)
				{
					const unsigned i = threadIndex + m_threadSize * threadLoad;
					func(i);
				}
			};
		m_packagedWorks[threadIndex] = std::make_unique<std::function<void()>>(std::forward<std::function<void()>>(work));
	}
	m_executing = false;
	m_version++;
}

WorkerHandle Jobs::GetAvailableWorker(size_t threadSize)
{
	std::unique_lock lock(m_workerManagementMutex);
	WorkerHandle workerHandle;
	if (m_availableWorker.at(threadSize).empty())
	{
		workerHandle = static_cast<WorkerHandle>(m_workers.size());
		const auto newWorker = std::make_shared<Worker>(threadSize, workerHandle);
		m_workers.emplace_back(newWorker);
	}
	else
	{
		workerHandle = m_availableWorker.at(threadSize).front();
		m_availableWorker.at(threadSize).pop();
	}
	return workerHandle;
}

size_t Jobs::GetDefaultThreadSize()
{
	const auto& jobs = GetInstance();
	return jobs.m_defaultThreadSize;
}

size_t Jobs::GetMaxThreadSize()
{
	const auto& jobs = GetInstance();
	return jobs.m_maxThreadSize;
}

void Jobs::Initialize(const size_t defaultThreadSize, const size_t maxThreadSize)
{
	auto& jobs = GetInstance();
	jobs.m_availableWorker.resize(maxThreadSize + 1);
	jobs.m_maxThreadSize = maxThreadSize;
	jobs.m_defaultThreadSize = defaultThreadSize;

	jobs.m_mainThreadId = std::this_thread::get_id();
}

void Jobs::RunParallelFor(const size_t size, std::function<void(unsigned i)>&& func, size_t threadSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return;
	}
	if (threadSize > jobs.m_maxThreadSize)
	{
		EVOENGINE_ERROR("Jobs: Exceed max thread size!");
	}
	if (threadSize == 0) threadSize = jobs.m_defaultThreadSize;
	const auto workerHandle = jobs.GetAvailableWorker(threadSize);
	const auto& worker = jobs.m_workers.at(workerHandle);
	worker->ScheduleParallelJobs(size, std::forward<std::function<void(unsigned i)>>(func));
	worker->Wait();
}

void Jobs::RunParallelFor(const size_t size, std::function<void(unsigned i, unsigned threadIndex)>&& func, size_t threadSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return;
	}
	if (threadSize > jobs.m_maxThreadSize)
	{
		EVOENGINE_ERROR("Jobs: Exceed max thread size!");
	}
	if (threadSize == 0) threadSize = jobs.m_defaultThreadSize;
	const auto workerHandle = jobs.GetAvailableWorker(threadSize);
	const auto& worker = jobs.m_workers.at(workerHandle);
	worker->ScheduleParallelJobs(size, std::forward<std::function<void(unsigned i, unsigned threadIndex)>>(func));
	worker->Wait();
}

JobHandle Jobs::ScheduleParallelFor(const size_t size, std::function<void(unsigned i)>&& func, size_t threadSize)
{
	auto& jobs = GetInstance();
	JobHandle retVal;
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		retVal.m_version = 0;
		retVal.m_workerHandle = -1;
		return retVal;
	}
	if (threadSize > jobs.m_maxThreadSize)
	{
		EVOENGINE_ERROR("Jobs: Exceed max thread size!");
	}
	if (threadSize == 0) threadSize = jobs.m_defaultThreadSize;
	const auto workerHandle = jobs.GetAvailableWorker(threadSize);
	const auto& worker = jobs.m_workers.at(workerHandle);
	retVal.m_workerHandle = workerHandle;
	retVal.m_version = worker->m_version;
	worker->ScheduleParallelJobs(size, std::forward<std::function<void(unsigned i)>>(func));
	return retVal;
}

JobHandle Jobs::ScheduleParallelFor(const size_t size, std::function<void(unsigned i, unsigned threadIndex)>&& func,
	size_t threadSize)
{
	auto& jobs = GetInstance();
	JobHandle retVal;
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		retVal.m_version = 0;
		retVal.m_workerHandle = -1;
		return retVal;
	}
	if (threadSize > jobs.m_maxThreadSize)
	{
		EVOENGINE_ERROR("Jobs: Exceed max thread size!");
	}
	if (threadSize == 0) threadSize = jobs.m_defaultThreadSize;
	const auto workerHandle = jobs.GetAvailableWorker(threadSize);
	const auto& worker = jobs.m_workers.at(workerHandle);
	retVal.m_workerHandle = workerHandle;
	retVal.m_version = worker->m_version;
	worker->ScheduleParallelJobs(size, std::forward<std::function<void(unsigned i, unsigned threadIndex)>>(func));
	return retVal;
}

void Jobs::RunParallelFor(const std::vector<JobHandle>& dependencies, const size_t size,
	std::function<void(unsigned i)>&& func, size_t threadSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return;
	}
	if (threadSize > jobs.m_maxThreadSize)
	{
		EVOENGINE_ERROR("Jobs: Exceed max thread size!");
	}
	if (threadSize == 0) threadSize = jobs.m_defaultThreadSize;
	const auto workerHandle = jobs.GetAvailableWorker(threadSize);
	const auto& worker = jobs.m_workers.at(workerHandle);
	std::vector<std::pair<std::shared_ptr<Worker>, size_t>> actualDependencies{};
	actualDependencies.resize(dependencies.size());
	for (int i = 0; i < dependencies.size(); i++)
	{
		actualDependencies[i] = std::make_pair(jobs.m_workers.at(dependencies.at(i).m_workerHandle), dependencies.at(i).m_version);
	}
	worker->ScheduleParallelJobs(actualDependencies, size, std::forward<std::function<void(unsigned i)>>(func));
	worker->Wait();
}

void Jobs::RunParallelFor(const std::vector<JobHandle>& dependencies, const size_t size,
	std::function<void(unsigned i, unsigned threadIndex)>&& func, size_t threadSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return;
	}
	if (threadSize > jobs.m_maxThreadSize)
	{
		EVOENGINE_ERROR("Jobs: Exceed max thread size!");
	}
	if (threadSize == 0) threadSize = jobs.m_defaultThreadSize;
	const auto workerHandle = jobs.GetAvailableWorker(threadSize);
	const auto& worker = jobs.m_workers.at(workerHandle);
	std::vector<std::pair<std::shared_ptr<Worker>, size_t>> actualDependencies{};
	actualDependencies.resize(dependencies.size());
	for (int i = 0; i < dependencies.size(); i++)
	{
		actualDependencies[i] = std::make_pair(jobs.m_workers.at(dependencies.at(i).m_workerHandle), dependencies.at(i).m_version);
	}
	worker->ScheduleParallelJobs(actualDependencies, size, std::forward<std::function<void(unsigned i, unsigned threadIndex)>>(func));
	worker->Wait();
}

JobHandle Jobs::ScheduleParallelFor(const std::vector<JobHandle>& dependencies, const size_t size,
	std::function<void(unsigned i)>&& func, size_t threadSize)
{
	auto& jobs = GetInstance();
	JobHandle retVal;
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		retVal.m_version = 0;
		retVal.m_workerHandle = -1;
		return retVal;
	}
	if (threadSize > jobs.m_maxThreadSize)
	{
		EVOENGINE_ERROR("Jobs: Exceed max thread size!");
	}
	if (threadSize == 0) threadSize = jobs.m_defaultThreadSize;
	const auto workerHandle = jobs.GetAvailableWorker(threadSize);
	const auto& worker = jobs.m_workers.at(workerHandle);
	retVal.m_workerHandle = workerHandle;
	retVal.m_version = worker->m_version;
	std::vector<std::pair<std::shared_ptr<Worker>, size_t>> actualDependencies{};
	actualDependencies.resize(dependencies.size());
	for (int i = 0; i < dependencies.size(); i++)
	{
		actualDependencies[i] = std::make_pair(jobs.m_workers.at(dependencies.at(i).m_workerHandle), dependencies.at(i).m_version);
	}
	worker->ScheduleParallelJobs(actualDependencies, size, std::forward<std::function<void(unsigned i)>>(func));
	return retVal;
}

JobHandle Jobs::ScheduleParallelFor(const std::vector<JobHandle>& dependencies, const size_t size,
	std::function<void(unsigned i, unsigned threadIndex)>&& func, size_t threadSize)
{
	auto& jobs = GetInstance();
	JobHandle retVal;
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		retVal.m_version = 0;
		retVal.m_workerHandle = -1;
		return retVal;
	}
	if (threadSize > jobs.m_maxThreadSize)
	{
		EVOENGINE_ERROR("Jobs: Exceed max thread size!");
	}
	if (threadSize == 0) threadSize = jobs.m_defaultThreadSize;
	const auto workerHandle = jobs.GetAvailableWorker(threadSize);
	const auto& worker = jobs.m_workers.at(workerHandle);
	retVal.m_workerHandle = workerHandle;
	retVal.m_version = worker->m_version;
	std::vector<std::pair<std::shared_ptr<Worker>, size_t>> actualDependencies{};
	actualDependencies.resize(dependencies.size());
	for (int i = 0; i < dependencies.size(); i++)
	{
		actualDependencies[i] = std::make_pair(jobs.m_workers.at(dependencies.at(i).m_workerHandle), dependencies.at(i).m_version);
	}
	worker->ScheduleParallelJobs(actualDependencies, size, std::forward<std::function<void(unsigned i, unsigned threadIndex)>>(func));
	return retVal;
}

JobHandle Jobs::Run(const std::vector<JobHandle>& dependencies, std::function<void()>&& func)
{
	auto& jobs = GetInstance();
	JobHandle retVal;
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		retVal.m_version = 0;
		retVal.m_workerHandle = -1;
		return retVal;
	}
	const auto workerHandle = jobs.GetAvailableWorker(1);
	const auto& worker = jobs.m_workers.at(workerHandle);
	retVal.m_workerHandle = workerHandle;
	retVal.m_version = worker->m_version;
	std::vector<std::pair<std::shared_ptr<Worker>, size_t>> actualDependencies{};
	actualDependencies.resize(dependencies.size());
	for (int i = 0; i < dependencies.size(); i++)
	{
		actualDependencies[i] = std::make_pair(jobs.m_workers.at(dependencies.at(i).m_workerHandle), dependencies.at(i).m_version);
	}
	worker->ScheduleJob(actualDependencies, std::forward<std::function<void()>>(func));
	return retVal;
}

JobHandle Jobs::Run(std::function<void()>&& func)
{
	auto& jobs = GetInstance();
	JobHandle retVal;
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		retVal.m_version = 0;
		retVal.m_workerHandle = -1;
		return retVal;
	}
	const auto workerHandle = jobs.GetAvailableWorker(1);
	const auto& worker = jobs.m_workers.at(workerHandle);
	retVal.m_workerHandle = workerHandle;
	retVal.m_version = worker->m_version;
	worker->ScheduleJob(std::forward<std::function<void()>>(func));
	return retVal;
}

JobHandle Jobs::Combine(const std::vector<JobHandle>& dependencies)
{
	auto& jobs = GetInstance();
	JobHandle retVal;
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		retVal.m_version = 0;
		retVal.m_workerHandle = -1;
		return retVal;
	}
	const auto workerHandle = jobs.GetAvailableWorker(1);
	const auto& worker = jobs.m_workers.at(workerHandle);
	retVal.m_workerHandle = workerHandle;
	retVal.m_version = worker->m_version;

	std::vector<std::pair<std::shared_ptr<Worker>, size_t>> actualDependencies{};
	actualDependencies.resize(dependencies.size());
	for(int i = 0; i < dependencies.size(); i++)
	{
		actualDependencies[i] = std::make_pair(jobs.m_workers.at(dependencies.at(i).m_workerHandle), dependencies.at(i).m_version);
	}
	worker->ScheduleJob(actualDependencies, []() {});
	return retVal;
}

void Jobs::Execute(const JobHandle& jobDependency)
{
	const auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!");
		return;
	}
	auto& worker = jobs.m_workers.at(jobDependency.m_workerHandle);
	if (worker->m_version != jobDependency.m_version)
	{
		return;
	}
	worker->Execute();
}

void Jobs::Wait(const JobHandle & jobDependency)
{
	const auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!");
		return;
	}
	auto& worker = jobs.m_workers.at(jobDependency.m_workerHandle);
	if(worker->m_version != jobDependency.m_version)
	{
		return;
	}
	worker->Wait();
}
