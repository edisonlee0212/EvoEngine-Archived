#include "Engine/Core/Jobs.hpp"

#include "Console.hpp"
using namespace EvoEngine;



int JobHandle::GetIndex() const
{
	return m_index;
}

bool JobHandle::Valid() const
{
	return m_index >= 0;
}

void JobSystemSemaphore::Reset(const size_t availability)
{
	m_availability = availability;
}

void JobSystemSemaphore::Acquire()
{
	std::unique_lock lk(m_updateMutex);
	m_cv.wait(lk, [this]() { return m_availability > 0; });
	m_availability--;
}

void JobSystemSemaphore::Release()
{
	std::unique_lock lk(m_updateMutex);
	m_availability++;
	m_cv.notify_one();
}

void Jobs::JobSystem::ReportFinish(const JobHandle& jobHandle)
{
	std::lock_guard jobManagementMutex(m_jobManagementMutex);
	const auto jobIndex = jobHandle.GetIndex();
	const auto& job = m_jobs[jobIndex];
	job->m_finished = true;
	job->m_finishedSemaphore.Release();
	for(const auto& parent : job->m_parents)
	{
		const auto& parentJob = m_jobs[parent.GetIndex()];
		bool parentJobReadyToRun = true;
		if(parentJob->m_wake)
		{
			for(const auto& childHandle : parentJob->m_children)
			{
				if(!m_jobs[childHandle.m_index]->m_finished)
				{
					parentJobReadyToRun = false;
					break;
				}
			}
		}
		if(parentJobReadyToRun)
		{
			m_availableJobPool.push(parent);
			std::unique_lock lock(this->m_jobAvailabilityMutex);
			m_jobAvailableCondition.notify_one();
		}
	}
}

void Jobs::JobSystem::InitializeWorker(const size_t workerIndex)
{
	std::shared_ptr flag(m_flags[workerIndex]); // a copy of the shared ptr to the flag
	auto threadFunc = [this, flag /* a copy of the shared ptr to the flag */]() {
		std::atomic<bool>& _flag = *flag;
		JobHandle jobHandle;
		bool isPop = false;
		while (true)
		{
			while (isPop)
			{ // if there is anything in the queue
				const auto& job = m_jobs.at(jobHandle.GetIndex());
				(*job->m_task)();
				job->m_task.reset();
				ReportFinish(jobHandle);
				if (_flag)
					return; // the thread is wanted to stop, return even if the queue is not empty yet
				isPop = m_availableJobPool.pop(jobHandle);
			}
			// the queue is empty here, wait for the next command
			std::unique_lock lock(m_jobAvailabilityMutex);
			++m_waitingThreadAmount;
			m_jobAvailableCondition.wait(lock, [this, &isPop, &jobHandle, &_flag]() {
				isPop = m_availableJobPool.pop(jobHandle);
				return isPop || _flag;
				});
			--m_waitingThreadAmount;
			if (!isPop)
				return; // if the queue is empty and this->isDone == true or *flag then return
		}
		};
	m_workers[workerIndex].reset(new std::thread(threadFunc)); // compiler may not support std::make_unique()
}

void Jobs::JobSystem::CollectDescendants(std::vector<JobHandle>& jobs, const JobHandle& walker)
{
	jobs.emplace_back(walker);
	for(const auto& i : m_jobs.at(walker.GetIndex())->m_children)
	{
		CollectDescendants(jobs, i);
	}
}

JobHandle Jobs::JobSystem::PushJob(const std::vector<JobHandle>& dependencies, std::function<void()>&& func)
{
	std::lock_guard jobManagementMutex(m_jobManagementMutex);
	std::vector<JobHandle> descendants;
	for(const auto& dependency : dependencies)
	{
		CollectDescendants(descendants, dependency);
	}
	for(const auto& jobHandle : descendants)
	{
		if(m_jobs.at(jobHandle.GetIndex())->m_wake)
		{
			EVOENGINE_ERROR("Descendants already started!");
			return {};
		}
	}
	JobHandle newJobHandle;
	if(!m_recycledJobs.empty())
	{
		newJobHandle = m_recycledJobs.front();
		m_recycledJobs.pop();
	}else
	{
		newJobHandle.m_index = m_jobs.size();
		m_jobs.emplace_back(std::make_shared<Job>());
		m_jobs.back()->m_handle = newJobHandle;
	}
	const auto& newJob = m_jobs.at(newJobHandle.GetIndex());
	newJob->m_task = std::make_unique<std::function<void()>>(std::forward< std::function<void()>>(func));
	newJob->m_wake = false;
	newJob->m_children = dependencies;
	newJob->m_recycled = false;
	newJob->m_finished = false;
	
	for(const auto& childHandle : dependencies)
	{
		const auto& childJob = m_jobs.at(childHandle.GetIndex());
		childJob->m_parents.emplace_back(newJobHandle);
	}
	return newJobHandle;
}

void Jobs::JobSystem::ExecuteJob(const JobHandle& jobHandle)
{
	std::lock_guard jobManagementMutex(m_jobManagementMutex);
	if (!m_jobs[jobHandle.GetIndex()]->m_parents.empty())
	{
		EVOENGINE_ERROR("Not root!");
		return;
	}
	std::vector<JobHandle> jobHandles;
	CollectDescendants(jobHandles, jobHandle);
	for(const auto& walker : jobHandles)
	{
		if (const auto& job = m_jobs[walker.GetIndex()]; !job->m_wake) {
			job->m_wake = true;
			if (job->m_children.empty())
			{
				m_availableJobPool.push(walker);
				std::unique_lock lock(this->m_jobAvailabilityMutex);
				m_jobAvailableCondition.notify_one();
			}
		}
	}
}

void Jobs::JobSystem::Wait(const JobHandle& jobHandle)
{
	const auto& job = m_jobs[jobHandle.GetIndex()];
	if(!job->m_parents.empty())
	{
		EVOENGINE_ERROR("Not root!");
		return;
	}
	std::vector<JobHandle> jobHandles;
	CollectDescendants(jobHandles, jobHandle);
	for (const auto& walker : jobHandles)
	{
		const auto& job = m_jobs[walker.GetIndex()];
		if (!job->m_recycled) {
			job->m_finishedSemaphore.Acquire();
			m_recycledJobs.emplace(walker);
			job->m_recycled = true;
			job->m_parents.clear();
			job->m_children.clear();
		}
	}
}

void Jobs::JobSystem::Initialize(const size_t workerSize)
{
	m_mainThreadId = std::this_thread::get_id();

	const auto oldNThreads = m_workers.size();
	if (oldNThreads <= workerSize)
	{ // if the number of threads is increased
		m_workers.resize(workerSize);
		m_flags.resize(workerSize);

		for (size_t i = oldNThreads; i < workerSize; ++i)
		{
			m_flags[i] = std::make_shared<std::atomic<bool>>(false);
			InitializeWorker(i);
		}
	}
	else
	{ // the number of threads is decreased
		for (size_t i = oldNThreads - 1; i >= workerSize; --i)
		{
			*m_flags[i] = true; // this thread will finish
			m_workers[i]->detach();
		}
		{
			// stop the detached threads that were waiting
			std::unique_lock lock(m_jobAvailabilityMutex);
			m_jobAvailableCondition.notify_all();
		}
		m_workers.resize(workerSize); // safe to delete because the threads are detached
		m_flags.resize(workerSize);   // safe to delete because the threads have copies of shared_ptr of the
		// flags, not originals
	}
}


size_t Jobs::GetWorkerSize()
{
	auto& jobs = GetInstance();
	return jobs.m_jobSystem.m_workers.size();
}

void Jobs::Initialize(const size_t workerSize)
{
	auto& jobs = GetInstance();
	jobs.m_jobSystem.Initialize(workerSize);
}

void Jobs::RunParallelFor(const size_t size, std::function<void(unsigned i)>&& func, size_t workerSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_jobSystem.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return;
	}
	if (workerSize == 0) workerSize = jobs.m_jobSystem.m_workers.size();
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
	if (jobs.m_jobSystem.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return;
	}
	if (workerSize == 0) workerSize = jobs.GetWorkerSize();
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
	if (jobs.m_jobSystem.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return {};
	}
	if (workerSize == 0) workerSize = jobs.GetWorkerSize();
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
	if (jobs.m_jobSystem.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return {};
	}
	if (workerSize == 0) workerSize = jobs.GetWorkerSize();
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
	if (jobs.m_jobSystem.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return;
	}
	if (workerSize == 0) workerSize = jobs.GetWorkerSize();
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
	if (jobs.m_jobSystem.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return;
	}
	if (workerSize == 0) workerSize = jobs.GetWorkerSize();
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
	if (jobs.m_jobSystem.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return {};
	}
	if (workerSize == 0) workerSize = jobs.GetWorkerSize();
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
	if (jobs.m_jobSystem.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return {};
	}
	if (workerSize == 0) workerSize = jobs.GetWorkerSize();
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
	if (jobs.m_jobSystem.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return {};
	}

	return jobs.m_jobSystem.PushJob(dependencies, std::forward<std::function<void()>>(func));
}

JobHandle Jobs::Run(std::function<void()>&& func)
{
	auto& jobs = GetInstance();
	if (jobs.m_jobSystem.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return {};
	}
	return jobs.m_jobSystem.PushJob({}, std::forward<std::function<void()>>(func));
}

JobHandle Jobs::Combine(const std::vector<JobHandle>& dependencies)
{
	auto& jobs = GetInstance();
	if (jobs.m_jobSystem.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return {};
	}

	return jobs.m_jobSystem.PushJob(dependencies, []() {});
}

void Jobs::Execute(const JobHandle& jobHandle)
{
	auto& jobs = GetInstance();
	if (jobs.m_jobSystem.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!");
		return;
	}
	jobs.m_jobSystem.ExecuteJob(jobHandle);
}

void Jobs::Wait(const JobHandle & jobHandle)
{
	auto& jobs = GetInstance();
	if (jobs.m_jobSystem.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!");
		return;
	}
	jobs.m_jobSystem.ExecuteJob(jobHandle);
	jobs.m_jobSystem.Wait(jobHandle);
}
