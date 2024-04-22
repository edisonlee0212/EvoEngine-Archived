#include "Engine/Core/Jobs.hpp"

#include "Console.hpp"
using namespace EvoEngine;

void Worker::CheckLoopHelper(std::vector<WorkerHandle>& collectedWorkers, const WorkerHandle currentWorker)
{
	collectedWorkers.emplace_back(currentWorker);
	const auto& jobs = Jobs::GetInstance();
	for(const auto& dependency : jobs.m_workers.at(currentWorker)->m_dependencies)
	{
		CheckLoopHelper(collectedWorkers, dependency);
	}
}

bool Worker::CheckLoop(const std::vector<WorkerHandle>& dependencies) const
{
	std::vector<WorkerHandle> collectedWorker = dependencies;
	for(const auto& i : collectedWorker)
	{
		CheckLoopHelper(collectedWorker, i);
	}
	
	for (const auto& i : collectedWorker)
	{
		if (i == m_handle) return true;
	}
	return false;
}

Worker::Worker(const size_t threadSize, const WorkerHandle handle)
{
	m_handle = handle;
	m_threads.Resize(threadSize);
}

void Worker::Wait()
{
	auto& jobs = Jobs::GetInstance();
	if(!m_scheduled) return;

	for (const auto& dependency : m_dependencies)
	{
		jobs.m_workers.at(dependency)->Wait();
	}
	m_dependencies.clear();
	for (const auto& i : m_tasks) i.wait();
	m_tasks.clear();

	m_scheduled = false;
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
	m_tasks.push_back(m_threads.Push([=](const int id)
		{
			func();
		}).share());
	m_scheduled = true;
}

void Worker::ScheduleTask(const std::vector<WorkerHandle>& dependencies, const std::function<void()>& func)
{
	if (m_scheduled)
	{
		EVOENGINE_ERROR("Worker: Currently busy!");
		return;
	}
#ifdef _DEBUG
	if (CheckLoop(dependencies))
	{
		EVOENGINE_ERROR("Worker: Loop!");
		return;
	}
#endif
	m_dependencies = dependencies;
	m_tasks.push_back(m_threads.Push([=](const int id)
		{
			const auto& jobs = Jobs::GetInstance();
			std::vector<std::shared_future<void>> pendingTasks;
			for (auto& i : dependencies) {
				auto& worker = jobs.m_workers.at(i);
				for (auto& task : worker->m_tasks)
				{
					pendingTasks.emplace_back(task);
				}
			}
			for (auto& task : pendingTasks)
			{
				task.wait();
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
	m_scheduled = true;
}

void Worker::ScheduleParallelTasks(const std::vector<WorkerHandle>& dependencies, const size_t size,
	const std::function<void(unsigned i, unsigned threadIndex)>& func, const size_t threadSize)
{
	if (m_scheduled)
	{
		EVOENGINE_ERROR("Worker: Currently busy!");
		return;
	}
#ifdef _DEBUG
	if (CheckLoop(dependencies))
	{
		EVOENGINE_ERROR("Worker: Loop!");
		return;
	}
#endif
	const auto threadLoad = size / threadSize;
	const auto loadReminder = size % threadSize;
	m_tasks.reserve(threadSize);
	for (int threadIndex = 0; threadIndex < threadSize; threadIndex++)
	{
		m_tasks.push_back(m_threads
			.Push([=](const int id) {
				const auto& jobs = Jobs::GetInstance();
				std::vector<std::shared_future<void>> pendingTasks;
				for (auto& i : dependencies) {
					auto& worker = jobs.m_workers.at(i);
					for (auto& task : worker->m_tasks)
					{
						pendingTasks.emplace_back(task);
					}
				}
				for (auto& task : pendingTasks)
				{
					task.wait();
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
			.Push([=](const int id) {
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

void Worker::ScheduleParallelTasks(const std::vector<WorkerHandle>& dependencies, const size_t size,
	const std::function<void(unsigned i)>& func, const size_t threadSize)
{
	if (m_scheduled)
	{
		EVOENGINE_ERROR("Worker: Currently busy!");
		return;
	}
#ifdef _DEBUG
	if (CheckLoop(dependencies))
	{
		EVOENGINE_ERROR("Worker: Loop!");
		return;
	}
#endif
	const auto threadLoad = size / threadSize;
	const auto loadReminder = size % threadSize;
	m_tasks.reserve(threadSize);
	for (int threadIndex = 0; threadIndex < threadSize; threadIndex++)
	{
		m_tasks.push_back(m_threads
			.Push([=](const int id) {
				const auto& jobs = Jobs::GetInstance();
				std::vector<std::shared_future<void>> pendingTasks;
				for (auto& i : dependencies) {
					auto& worker = jobs.m_workers.at(i);
					for (auto& task : worker->m_tasks)
					{
						pendingTasks.emplace_back(task);
					}
				}
				for (auto& task : pendingTasks)
				{
					task.wait();
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
	WorkerHandle workerHandle;
	if(jobs.m_availableWorker.at(threadSize).empty())
	{
		workerHandle = jobs.m_workers.size();
		const auto newWorker = std::make_shared<Worker>(threadSize, workerHandle);
		jobs.m_workers.emplace_back(newWorker);
	}else
	{
		workerHandle = jobs.m_availableWorker.at(threadSize).front();
		jobs.m_availableWorker.at(threadSize).pop();
	}
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
	WorkerHandle workerHandle;
	if (jobs.m_availableWorker.at(threadSize).empty())
	{
		workerHandle = jobs.m_workers.size();
		const auto newWorker = std::make_shared<Worker>(threadSize, workerHandle);
		jobs.m_workers.emplace_back(newWorker);
	}
	else
	{
		workerHandle = jobs.m_availableWorker.at(threadSize).front();
		jobs.m_availableWorker.at(threadSize).pop();
	}
	const auto& worker = jobs.m_workers.at(workerHandle);
	worker->ScheduleParallelTasks(size, func, threadSize);
	worker->Wait();
}

WorkerHandle Jobs::AddParallelFor(const size_t size, const std::function<void(unsigned i)>& func, size_t threadSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return -1;
	}
	if (threadSize > jobs.m_maxThreadSize)
	{
		EVOENGINE_ERROR("Jobs: Exceed max thread size!");
	}
	if (threadSize == 0) threadSize = jobs.m_defaultThreadSize;
	WorkerHandle workerHandle;
	if (jobs.m_availableWorker.at(threadSize).empty())
	{
		workerHandle = jobs.m_workers.size();
		const auto newWorker = std::make_shared<Worker>(threadSize, workerHandle);
		jobs.m_workers.emplace_back(newWorker);
	}
	else
	{
		workerHandle = jobs.m_availableWorker.at(threadSize).front();
		jobs.m_availableWorker.at(threadSize).pop();
	}
	const auto& worker = jobs.m_workers.at(workerHandle);
	worker->ScheduleParallelTasks(size, func, threadSize);
	return workerHandle;
}

WorkerHandle Jobs::AddParallelFor(const size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func,
	size_t threadSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return -1;
	}
	if (threadSize > jobs.m_maxThreadSize)
	{
		EVOENGINE_ERROR("Jobs: Exceed max thread size!");
	}
	if (threadSize == 0) threadSize = jobs.m_defaultThreadSize;
	WorkerHandle workerHandle;
	if (jobs.m_availableWorker.at(threadSize).empty())
	{
		workerHandle = jobs.m_workers.size();
		const auto newWorker = std::make_shared<Worker>(threadSize, workerHandle);
		jobs.m_workers.emplace_back(newWorker);
	}
	else
	{
		workerHandle = jobs.m_availableWorker.at(threadSize).front();
		jobs.m_availableWorker.at(threadSize).pop();
	}
	const auto& worker = jobs.m_workers.at(workerHandle);
	worker->ScheduleParallelTasks(size, func, threadSize);
	return workerHandle;
}

void Jobs::ParallelFor(const std::vector<WorkerHandle>& dependencies, const size_t size,
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
	WorkerHandle workerHandle;
	if (jobs.m_availableWorker.at(threadSize).empty())
	{
		workerHandle = jobs.m_workers.size();
		const auto newWorker = std::make_shared<Worker>(threadSize, workerHandle);
		jobs.m_workers.emplace_back(newWorker);
	}
	else
	{
		workerHandle = jobs.m_availableWorker.at(threadSize).front();
		jobs.m_availableWorker.at(threadSize).pop();
	}
	const auto& worker = jobs.m_workers.at(workerHandle);
	worker->ScheduleParallelTasks(dependencies, size, func, threadSize);
	worker->Wait();
}

void Jobs::ParallelFor(const std::vector<WorkerHandle>& dependencies, const size_t size,
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
	WorkerHandle workerHandle;
	if (jobs.m_availableWorker.at(threadSize).empty())
	{
		workerHandle = jobs.m_workers.size();
		const auto newWorker = std::make_shared<Worker>(threadSize, workerHandle);
		jobs.m_workers.emplace_back(newWorker);
	}
	else
	{
		workerHandle = jobs.m_availableWorker.at(threadSize).front();
		jobs.m_availableWorker.at(threadSize).pop();
	}
	const auto& worker = jobs.m_workers.at(workerHandle);
	worker->ScheduleParallelTasks(dependencies, size, func, threadSize);
	worker->Wait();
}

WorkerHandle Jobs::AddParallelFor(const std::vector<WorkerHandle>& dependencies, const size_t size,
	const std::function<void(unsigned i)>& func, size_t threadSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return -1;
	}
	if (threadSize > jobs.m_maxThreadSize)
	{
		EVOENGINE_ERROR("Jobs: Exceed max thread size!");
	}
	if (threadSize == 0) threadSize = jobs.m_defaultThreadSize;
	WorkerHandle workerHandle;
	if (jobs.m_availableWorker.at(threadSize).empty())
	{
		workerHandle = jobs.m_workers.size();
		const auto newWorker = std::make_shared<Worker>(threadSize, workerHandle);
		jobs.m_workers.emplace_back(newWorker);
	}
	else
	{
		workerHandle = jobs.m_availableWorker.at(threadSize).front();
		jobs.m_availableWorker.at(threadSize).pop();
	}
	const auto& worker = jobs.m_workers.at(workerHandle);
	worker->ScheduleParallelTasks(dependencies, size, func, threadSize);
	return workerHandle;
}

WorkerHandle Jobs::AddParallelFor(const std::vector<WorkerHandle>& dependencies, const size_t size,
	const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return -1;
	}
	if (threadSize > jobs.m_maxThreadSize)
	{
		EVOENGINE_ERROR("Jobs: Exceed max thread size!");
	}
	if (threadSize == 0) threadSize = jobs.m_defaultThreadSize;
	WorkerHandle workerHandle;
	if (jobs.m_availableWorker.at(threadSize).empty())
	{
		workerHandle = jobs.m_workers.size();
		const auto newWorker = std::make_shared<Worker>(threadSize, workerHandle);
		jobs.m_workers.emplace_back(newWorker);
	}
	else
	{
		workerHandle = jobs.m_availableWorker.at(threadSize).front();
		jobs.m_availableWorker.at(threadSize).pop();
	}
	const auto& worker = jobs.m_workers.at(workerHandle);
	worker->ScheduleParallelTasks(dependencies, size, func, threadSize);
	return workerHandle;
}

WorkerHandle Jobs::AddTask(const std::vector<WorkerHandle>& dependencies, const std::function<void()>& func)
{
	auto& jobs = GetInstance();
	if(jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return -1;
	}
	WorkerHandle workerHandle;
	if (jobs.m_availableWorker.at(1).empty())
	{
		workerHandle = jobs.m_workers.size();
		const auto newWorker = std::make_shared<Worker>(1, workerHandle);
		jobs.m_workers.emplace_back(newWorker);
	}
	else
	{
		workerHandle = jobs.m_availableWorker.at(1).front();
		jobs.m_availableWorker.at(1).pop();
	}
	const auto& worker = jobs.m_workers.at(workerHandle);
	worker->ScheduleTask(dependencies, func);
	return workerHandle;
}

WorkerHandle Jobs::AddTask(const std::function<void()>& func)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return -1;
	}
	WorkerHandle workerHandle;
	if (jobs.m_availableWorker.at(1).empty())
	{
		workerHandle = jobs.m_workers.size();
		const auto newWorker = std::make_shared<Worker>(1, workerHandle);
		jobs.m_workers.emplace_back(newWorker);
	}
	else
	{
		workerHandle = jobs.m_availableWorker.at(1).front();
		jobs.m_availableWorker.at(1).pop();
	}
	const auto& worker = jobs.m_workers.at(workerHandle);
	worker->ScheduleTask(func);
	return workerHandle;
}

WorkerHandle Jobs::PackTask(const std::vector<WorkerHandle>& dependencies)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return -1;
	}
	WorkerHandle workerHandle;
	if (jobs.m_availableWorker.at(1).empty())
	{
		workerHandle = jobs.m_workers.size();
		const auto newWorker = std::make_shared<Worker>(1, workerHandle);
		jobs.m_workers.emplace_back(newWorker);
	}
	else
	{
		workerHandle = jobs.m_availableWorker.at(1).front();
		jobs.m_availableWorker.at(1).pop();
	}
	const auto& worker = jobs.m_workers.at(workerHandle);
	worker->ScheduleTask(dependencies, [&](){});
	return workerHandle;
}

void Jobs::Wait(const WorkerHandle workerHandle)
{
	const auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!");
		return;
	}
	jobs.m_workers.at(workerHandle)->Wait();
}
