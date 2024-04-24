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
	m_threadSize = threadSize;
	m_threads.resize(m_threadSize);
	m_works.resize(m_threadSize);
	m_results.resize(m_threadSize);
	m_tasksCondition = std::vector<std::condition_variable>(m_threadSize);
	for(unsigned threadIndex = 0; threadIndex < m_threadSize; threadIndex++)
	{
		auto threadFunc = [this, threadIndex]()
			{
				bool newTaskAssigned = false;
				while(true)
				{
					if(newTaskAssigned)
					{
						(*m_works[threadIndex])();
						m_works[threadIndex].reset();
						newTaskAssigned = false;
					}
					std::unique_lock lock(this->m_conditionVariableMutex);
					this->m_tasksCondition[threadIndex].wait(lock, [this, threadIndex, &newTaskAssigned]()
					{
						newTaskAssigned = m_works[threadIndex].get();
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
	if(m_executing || !m_scheduled) return;
	std::unique_lock lock(this->m_conditionVariableMutex);
	m_executing = true;
	for (auto& i : m_tasksCondition) i.notify_one();
}

void Jobs::Worker::Wait()
{
	if (!m_scheduled) return;
	if (!m_executing) Execute();
	std::lock_guard lock(m_schedulerMutex);
	for (const auto& i : m_results) i.wait();
	m_dependencies.clear();
	
	m_scheduled = false;
	m_executing = false;
	m_version++;

	auto& jobs = Jobs::GetInstance();
	std::unique_lock workerManagementLock(jobs.m_workerManagementMutex);
	jobs.m_availableWorker.at(m_threads.size()).emplace(m_handle);

}

void Jobs::Worker::Wait(const size_t version)
{
	if (!m_scheduled) return;
	if (version != m_version) return;
	if (!m_executing) Execute();
	std::lock_guard lock(m_schedulerMutex);
	
	for (const auto& i : m_results) i.wait();
	m_dependencies.clear();

	m_scheduled = false;
	m_executing = false;
	m_version++;

	auto& jobs = Jobs::GetInstance();
	std::unique_lock workerManagementLock(jobs.m_workerManagementMutex);
	jobs.m_availableWorker.at(m_threadSize).emplace(m_handle);
}

WorkerHandle Jobs::Worker::GetHandle() const
{
	return m_handle;
}

bool Jobs::Worker::Executing() const
{
	return m_executing;
}

bool Jobs::Worker::Scheduled() const
{
	return m_scheduled;
}

void Jobs::Worker::ScheduleJob(const std::function<void()>& func)
{
	if (m_scheduled)
	{
		EVOENGINE_ERROR("Worker: Currently scheduled!");
		return;
	}
	if (m_executing)
	{
		EVOENGINE_ERROR("Worker: Currently executing!");
		return;
	}
	assert(m_threadSize == 1);
	const auto actualFunc = [this, func]()
		{
			func();
		};
	auto pck = std::make_shared<std::packaged_task<void()>>(std::forward<std::function<void()>>(actualFunc));
	m_results[0] = pck->get_future().share();
	m_works[0] = std::make_unique<std::function<void()>>(std::function([pck]() {(*pck)(); }));
	m_scheduled = true;
}

void Jobs::Worker::ScheduleJob(const std::vector<std::pair<std::shared_ptr<Worker>, size_t>>& dependencies, const std::function<void()>& func)
{
	if (m_scheduled)
	{
		EVOENGINE_ERROR("Worker: Currently scheduled!");
		return;
	}
	if (m_executing)
	{
		EVOENGINE_ERROR("Worker: Currently executing!");
		return;
	}
	if (!dependencies.empty()) {
		m_dependencies = dependencies;
	}

	assert(m_threadSize == 1);
	const auto actualFunc = [this, func]()
		{
			for (const auto& dependency : m_dependencies)
			{
				dependency.first->Wait(dependency.second);
			}
			func();
		};
	auto pck = std::make_shared<std::packaged_task<void()>>(std::forward<std::function<void()>>(actualFunc));
	m_results[0] = pck->get_future().share();
	m_works[0] = std::make_unique<std::function<void()>>(std::function([pck]() {(*pck)(); }));
	m_scheduled = true;
}

void Jobs::Worker::ScheduleParallelJobs(const size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func)
{
	if (m_scheduled)
	{
		EVOENGINE_ERROR("Worker: Currently scheduled!");
		return;
	}
	if (m_executing)
	{
		EVOENGINE_ERROR("Worker: Currently executing!");
		return;
	}
	const auto threadLoad = size / m_threadSize;
	const auto loadReminder = size % m_threadSize;

	for (int threadIndex = 0; threadIndex < m_threadSize; threadIndex++)
	{
		const auto actualFunc = [this, func, threadIndex, threadLoad, loadReminder]()
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
		auto pck = std::make_shared<std::packaged_task<void()>>(std::forward<std::function<void()>>(actualFunc));
		m_results[threadIndex] = pck->get_future().share();
		m_works[threadIndex] = std::make_unique<std::function<void()>>(std::function([pck]() {(*pck)(); }));
	}
	m_scheduled = true;
}

void Jobs::Worker::ScheduleParallelJobs(const std::vector<std::pair<std::shared_ptr<Worker>, size_t>>& dependencies, const size_t size,
	const std::function<void(unsigned i, unsigned threadIndex)>& func)
{
	if (m_scheduled)
	{
		EVOENGINE_ERROR("Worker: Currently scheduled!");
		return;
	}
	if (m_executing)
	{
		EVOENGINE_ERROR("Worker: Currently executing!");
		return;
	}
	const auto threadLoad = size / m_threadSize;
	const auto loadReminder = size % m_threadSize;
	if (!dependencies.empty()) {
		m_dependencies = dependencies;
	}
	for (int threadIndex = 0; threadIndex < m_threadSize; threadIndex++)
	{
		const auto actualFunc = [this, func, threadIndex, threadLoad, loadReminder]()
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
		auto pck = std::make_shared<std::packaged_task<void()>>(std::forward<std::function<void()>>(actualFunc));
		m_results[threadIndex] = pck->get_future().share();
		m_works[threadIndex] = std::make_unique<std::function<void()>>(std::function([pck]() {(*pck)(); }));
	}
	m_scheduled = true;
}

void Jobs::Worker::ScheduleParallelJobs(const size_t size, const std::function<void(unsigned i)>& func)
{
	if (m_scheduled)
	{
		EVOENGINE_ERROR("Worker: Currently scheduled!");
		return;
	}
	if (m_executing)
	{
		EVOENGINE_ERROR("Worker: Currently executing!");
		return;
	}
	const auto threadLoad = size / m_threadSize;
	const auto loadReminder = size % m_threadSize;
	for (int threadIndex = 0; threadIndex < m_threadSize; threadIndex++)
	{
		const auto actualFunc = [this, func, threadIndex, threadLoad, loadReminder]()
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
		auto pck = std::make_shared<std::packaged_task<void()>>(std::forward<std::function<void()>>(actualFunc));
		m_results[threadIndex] = pck->get_future().share();
		m_works[threadIndex] = std::make_unique<std::function<void()>>(std::function([pck]() {(*pck)(); }));
	}
	m_scheduled = true;
}

void Jobs::Worker::ScheduleParallelJobs(const std::vector<std::pair<std::shared_ptr<Worker>, size_t>>& dependencies, const size_t size,
	const std::function<void(unsigned i)>& func)
{
	if (m_scheduled)
	{
		EVOENGINE_ERROR("Worker: Currently scheduled!");
		return;
	}
	if (m_executing)
	{
		EVOENGINE_ERROR("Worker: Currently executing!");
		return;
	}
	if (!dependencies.empty()) {
		m_dependencies = dependencies;
	}
	const auto threadLoad = size / m_threadSize;
	const auto loadReminder = size % m_threadSize;
	for (int threadIndex = 0; threadIndex < m_threadSize; threadIndex++)
	{
		const auto actualFunc = [this, func, threadIndex, threadLoad, loadReminder]()
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
		auto pck = std::make_shared<std::packaged_task<void()>>(std::forward<std::function<void()>>(actualFunc));
		m_results[threadIndex] = pck->get_future().share();
		m_works[threadIndex] = std::make_unique<std::function<void()>>(std::function([pck]() {(*pck)(); }));
	}
	m_scheduled = true;
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

void Jobs::RunParallelFor(const size_t size, const std::function<void(unsigned i)>& func, size_t threadSize)
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
	worker->ScheduleParallelJobs(size, func);
	worker->Wait();
}

void Jobs::RunParallelFor(const size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize)
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
	worker->ScheduleParallelJobs(size, func);
	worker->Wait();
}

JobHandle Jobs::ScheduleParallelFor(const size_t size, const std::function<void(unsigned i)>& func, size_t threadSize)
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
	worker->ScheduleParallelJobs(size, func);
	return retVal;
}

JobHandle Jobs::ScheduleParallelFor(const size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func,
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
	worker->ScheduleParallelJobs(size, func);
	return retVal;
}

void Jobs::RunParallelFor(const std::vector<JobHandle>& dependencies, const size_t size,
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
	worker->ScheduleParallelJobs(actualDependencies, size, func);
	worker->Wait();
}

void Jobs::RunParallelFor(const std::vector<JobHandle>& dependencies, const size_t size,
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
	worker->ScheduleParallelJobs(actualDependencies, size, func);
	worker->Wait();
}

JobHandle Jobs::ScheduleParallelFor(const std::vector<JobHandle>& dependencies, const size_t size,
	const std::function<void(unsigned i)>& func, size_t threadSize)
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
	worker->ScheduleParallelJobs(actualDependencies, size, func);
	return retVal;
}

JobHandle Jobs::ScheduleParallelFor(const std::vector<JobHandle>& dependencies, const size_t size,
	const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize)
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
	worker->ScheduleParallelJobs(actualDependencies, size, func);
	return retVal;
}

JobHandle Jobs::Run(const std::vector<JobHandle>& dependencies, const std::function<void()>& func)
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
	worker->ScheduleJob(actualDependencies, func);
	return retVal;
}

JobHandle Jobs::Run(const std::function<void()>& func)
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
	worker->ScheduleJob(func);
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
