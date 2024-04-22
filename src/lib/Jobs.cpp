#include "Engine/Core/Jobs.hpp"

#include "Console.hpp"
using namespace EvoEngine;


size_t JobDependency::GetVersion() const
{
	return m_version;
}

WorkerHandle JobDependency::GetWorkerHandle() const
{
	return m_workerHandle;
}

bool JobDependency::Valid() const
{
	return m_workerHandle >= 0;
}

size_t Worker::GetVersion() const
{
	return m_version;
}

Worker::Worker(const size_t threadSize, const WorkerHandle handle)
{
	m_handle = handle;
	m_threads.Resize(threadSize);
	m_version = 0;
}

void Worker::Wait()
{
	if (!m_scheduled) return;
	std::lock_guard lock(m_mutex);
	
	for (const auto& i : m_tasks) i.wait();
	m_dependencies.clear();
	m_tasks.clear();

	m_scheduled = false;
	m_version++;

	auto& jobs = Jobs::GetInstance();
	std::lock_guard workerManagementLock(jobs.m_workerManagementMutex);
	jobs.m_availableWorker.at(m_threads.Size()).emplace(m_handle);

}

void Worker::Wait(const size_t version)
{
	if (!m_scheduled) return;
	if (version != m_version) return;
	std::lock_guard lock(m_mutex);
	
	
	for (const auto& i : m_tasks) i.wait();
	m_dependencies.clear();
	m_tasks.clear();

	m_scheduled = false;
	m_version++;
	auto& jobs = Jobs::GetInstance();
	std::lock_guard workerManagementLock(jobs.m_workerManagementMutex);
	jobs.m_availableWorker.at(m_threads.Size()).emplace(m_handle);
}

ThreadPool& Worker::RefThreadPool()
{
	return m_threads;
}

const ThreadPool& Worker::PeekThreadPool() const
{
	return m_threads;
}

WorkerHandle Worker::GetHandle() const
{
	return m_handle;
}

bool Worker::Scheduled() const
{
	return m_scheduled;
}

void Worker::ScheduleTask(const std::function<void()>& func)
{
	if (m_scheduled)
	{
		EVOENGINE_ERROR("Worker: Currently busy!");
		return;
	}
	m_tasks.push_back(m_threads.Push([this, func](const int id)
		{
			func();
		}).share());
	m_scheduled = true;
}

void Worker::ScheduleTask(const std::vector<std::pair<std::shared_ptr<Worker>, size_t>>& dependencies, const std::function<void()>& func)
{
	if (m_scheduled)
	{
		EVOENGINE_ERROR("Worker: Currently busy!");
		return;
	}
	if (!dependencies.empty()) {
		std::lock_guard lock(m_mutex);
		m_dependencies = dependencies;
	}
	m_tasks.push_back(m_threads.Push([this, func](const int id)
		{

			for (const auto& dependency : m_dependencies)
			{
				dependency.first->Wait(dependency.second);
			}

			func();
		}
	).share());
	m_scheduled = true;
}

void Worker::ScheduleParallelTasks(const size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, const size_t threadSize)
{
	if (m_scheduled)
	{
		EVOENGINE_ERROR("Worker: Currently busy!");
		return;
	}
	const auto threadLoad = size / threadSize;
	const auto loadReminder = size % threadSize;
	m_tasks.reserve(threadSize);
	for (int threadIndex = 0; threadIndex < threadSize; threadIndex++)
	{
		m_tasks.push_back(m_threads
			.Push([this, func, threadIndex, threadSize, threadLoad, loadReminder](const int id) {
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
	m_scheduled = true;
}

void Worker::ScheduleParallelTasks(const std::vector<std::pair<std::shared_ptr<Worker>, size_t>>& dependencies, const size_t size,
	const std::function<void(unsigned i, unsigned threadIndex)>& func, const size_t threadSize)
{
	if (m_scheduled)
	{
		EVOENGINE_ERROR("Worker: Currently busy!");
		return;
	}
	const auto threadLoad = size / threadSize;
	const auto loadReminder = size % threadSize;
	if (!dependencies.empty()) {
		std::lock_guard lock(m_mutex);
		m_dependencies = dependencies;
	}
	m_tasks.reserve(threadSize);
	for (int threadIndex = 0; threadIndex < threadSize; threadIndex++)
	{
		m_tasks.push_back(m_threads
			.Push([this, func, threadIndex, threadSize, threadLoad, loadReminder](const int id) {
				for (const auto& dependency : m_dependencies)
				{
					dependency.first->Wait(dependency.second);
				}
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
	m_scheduled = true;
}

void Worker::ScheduleParallelTasks(const size_t size, const std::function<void(unsigned i)>& func, const size_t threadSize)
{
	if (m_scheduled)
	{
		EVOENGINE_ERROR("Worker: Currently busy!");
		return;
	}
	const auto threadLoad = size / threadSize;
	const auto loadReminder = size % threadSize;
	m_tasks.reserve(threadSize);
	for (int threadIndex = 0; threadIndex < threadSize; threadIndex++)
	{
		m_tasks.push_back(m_threads
			.Push([this, func, threadIndex, threadSize, threadLoad, loadReminder](const int id) {
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
	m_scheduled = true;
}

void Worker::ScheduleParallelTasks(const std::vector<std::pair<std::shared_ptr<Worker>, size_t>>& dependencies, const size_t size,
	const std::function<void(unsigned i)>& func, const size_t threadSize)
{
	if (m_scheduled)
	{
		EVOENGINE_ERROR("Worker: Currently busy!");
		return;
	}
	if (!dependencies.empty()) {
		std::lock_guard lock(m_mutex);
		m_dependencies = dependencies;
	}
	const auto threadLoad = size / threadSize;
	const auto loadReminder = size % threadSize;
	m_tasks.reserve(threadSize);
	for (int threadIndex = 0; threadIndex < threadSize; threadIndex++)
	{
		m_tasks.push_back(m_threads
			.Push([this, func, threadIndex, threadSize, threadLoad, loadReminder](const int id) {
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
					const unsigned i = threadIndex + threadSize * threadLoad;
					func(i);
				}
				})
			.share());
	}
	m_scheduled = true;
}

WorkerHandle Jobs::GetAvailableWorker(size_t threadSize)
{
	std::lock_guard lock(m_workerManagementMutex);
	WorkerHandle workerHandle;
	if (m_availableWorker.at(threadSize).empty())
	{
		workerHandle = m_workers.size();
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

void Jobs::ParallelFor(const size_t size, const std::function<void(unsigned i)>& func, size_t threadSize)
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
	worker->ScheduleParallelTasks(size, func, threadSize);
	worker->Wait();
}

void Jobs::ParallelFor(const size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize)
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
	worker->ScheduleParallelTasks(size, func, threadSize);
	worker->Wait();
}

JobDependency Jobs::AddParallelFor(const size_t size, const std::function<void(unsigned i)>& func, size_t threadSize)
{
	auto& jobs = GetInstance();
	JobDependency retVal;
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
	worker->ScheduleParallelTasks(size, func, threadSize);
	return retVal;
}

JobDependency Jobs::AddParallelFor(const size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func,
	size_t threadSize)
{
	auto& jobs = GetInstance();
	JobDependency retVal;
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
	worker->ScheduleParallelTasks(size, func, threadSize);
	return retVal;
}

void Jobs::ParallelFor(const std::vector<JobDependency>& dependencies, const size_t size,
	const std::function<void(unsigned i)>& func, size_t threadSize)
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
	worker->ScheduleParallelTasks(actualDependencies, size, func, threadSize);
	worker->Wait();
}

void Jobs::ParallelFor(const std::vector<JobDependency>& dependencies, const size_t size,
	const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize)
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
	worker->ScheduleParallelTasks(actualDependencies, size, func, threadSize);
	worker->Wait();
}

JobDependency Jobs::AddParallelFor(const std::vector<JobDependency>& dependencies, const size_t size,
	const std::function<void(unsigned i)>& func, size_t threadSize)
{
	auto& jobs = GetInstance();
	JobDependency retVal;
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
	worker->ScheduleParallelTasks(actualDependencies, size, func, threadSize);
	return retVal;
}

JobDependency Jobs::AddParallelFor(const std::vector<JobDependency>& dependencies, const size_t size,
	const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize)
{
	auto& jobs = GetInstance();
	JobDependency retVal;
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
	worker->ScheduleParallelTasks(actualDependencies, size, func, threadSize);
	return retVal;
}

JobDependency Jobs::AddTask(const std::vector<JobDependency>& dependencies, const std::function<void()>& func)
{
	auto& jobs = GetInstance();
	JobDependency retVal;
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
	worker->ScheduleTask(actualDependencies, func);
	return retVal;
}

JobDependency Jobs::AddTask(const std::function<void()>& func)
{
	auto& jobs = GetInstance();
	JobDependency retVal;
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
	worker->ScheduleTask(func);
	return retVal;
}

JobDependency Jobs::PackTask(const std::vector<JobDependency>& dependencies)
{
	auto& jobs = GetInstance();
	JobDependency retVal;
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
	worker->ScheduleTask(actualDependencies, []() {});
	return retVal;
}

void Jobs::Wait(const JobDependency &dependency)
{
	const auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!");
		return;
	}
	auto& worker = jobs.m_workers.at(dependency.m_workerHandle);
	if(worker->m_version != dependency.m_version)
	{
		return;
	}
	worker->Wait();
}
