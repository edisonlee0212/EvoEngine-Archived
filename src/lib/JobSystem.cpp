#include "JobSystem.hpp"

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

void JobSystem::JobSystemSemaphore::Reset(const size_t availability)
{
	m_availability = availability;
}

void JobSystem::JobSystemSemaphore::Acquire()
{
	std::unique_lock lk(m_updateMutex);
	m_cv.wait(lk, [this]() { return m_availability > 0; });
	m_availability--;
}

void JobSystem::JobSystemSemaphore::Release()
{
	std::unique_lock lk(m_updateMutex);
	m_availability++;
	m_cv.notify_one();
}

inline void JobSystem::JobPool::Push(JobHandle const& jobHandle)
{
	std::unique_lock lock(m_mutex);
	m_queue.push(jobHandle);
}

inline bool JobSystem::JobPool::Pop(JobHandle& jobHandle)
{
	std::unique_lock lock(m_mutex);
	if (m_queue.empty())
		return false;
	jobHandle = m_queue.front();
	m_queue.pop();
	return true;
}

inline bool JobSystem::JobPool::Empty()
{
	std::unique_lock lock(m_mutex);
	return m_queue.empty();
}

void JobSystem::CheckJobAvailableHelper(const JobHandle& jobHandle)
{
	const auto& job = m_jobs.at(jobHandle.GetIndex());
	bool jobReadyToRun = job->m_wake;
	if (jobReadyToRun) {
		for (const auto& childHandle : job->m_children)
		{
			if (!m_jobs[childHandle.m_index]->m_finished)
			{
				jobReadyToRun = false;
				break;
			}
		}
	}
	if (jobReadyToRun)
	{
		m_availableJobPool.Push(jobHandle);
		std::unique_lock lock(this->m_jobAvailabilityMutex);
		m_jobAvailableCondition.notify_one();
	}
}
void JobSystem::ReportFinish(const JobHandle& jobHandle)
{
	std::lock_guard jobManagementMutex(m_jobManagementMutex);
	const auto jobIndex = jobHandle.GetIndex();
	const auto& job = m_jobs[jobIndex];
	job->m_finished = true;
	job->m_finishedSemaphore.Release();
	for (const auto& parent : job->m_parents)
	{
		CheckJobAvailableHelper(parent);
	}
}

void JobSystem::InitializeWorker(const size_t workerIndex)
{
	std::shared_ptr flag(m_flags[workerIndex]); // a copy of the shared ptr to the flag
	auto threadFunc = [this, flag /* a copy of the shared ptr to the flag */]() {
		std::atomic<bool>& flagPtr = *flag;
		JobHandle jobHandle;
		bool isPop = m_availableJobPool.Pop(jobHandle);
		while (true)
		{
			while (isPop)
			{ // if there is anything in the queue
				const auto& job = m_jobs.at(jobHandle.GetIndex());
				(*job->m_task)();
				job->m_task.reset();
				ReportFinish(jobHandle);
				//if (flagPtr)
				//	return; // the thread is wanted to stop, return even if the queue is not empty yet
				isPop = m_availableJobPool.Pop(jobHandle);
			}
			// the queue is empty here, wait for the next command
			std::unique_lock lock(m_jobAvailabilityMutex);
			++m_idleThreadAmount;
			m_jobAvailableCondition.wait(lock, [this, &isPop, &jobHandle, &flagPtr]() {
				isPop = m_availableJobPool.Pop(jobHandle);
				return isPop || m_isDone || flagPtr;
				});
			--m_idleThreadAmount;
			if (!isPop)
				return; // if the queue is empty and m_isDone == true or *flag then return
		}
		};
	m_workers[workerIndex].reset(new std::thread(threadFunc)); // compiler may not support std::make_unique()
}

bool JobSystem::MainThreadCheck() const
{
	if (m_mainThreadId == std::this_thread::get_id())
	{
		return true;
	}
	EVOENGINE_ERROR("Jobs: Not on main thread!!");
	return false;
}

void JobSystem::CollectDescendantsHelper(std::vector<JobHandle>& jobs, const JobHandle& walker)
{
	jobs.emplace_back(walker);
	for (const auto& i : m_jobs.at(walker.GetIndex())->m_children)
	{
		CollectDescendantsHelper(jobs, i);
	}
}



JobHandle JobSystem::PushJob(const std::vector<JobHandle>& dependencies, std::function<void()>&& func)
{
	if (!MainThreadCheck()) return {};
	std::lock_guard jobManagementMutex(m_jobManagementMutex);
	std::vector<JobHandle> filteredDependencies;
	for (const auto& dependency : dependencies)
	{
		if (!dependency.Valid()) continue;
		filteredDependencies.emplace_back(dependency);
	}
	std::vector<JobHandle> descendants;
	for (const auto& dependency : filteredDependencies)
	{
		CollectDescendantsHelper(descendants, dependency);
	}
	for (const auto& jobHandle : descendants)
	{
		if (m_jobs.at(jobHandle.GetIndex())->m_wake)
		{
			EVOENGINE_ERROR("Descendants already started!");
			return {};
		}
	}
	JobHandle newJobHandle;
	if (!m_recycledJobs.empty())
	{
		newJobHandle = m_recycledJobs.front();
		m_recycledJobs.pop();
	}
	else
	{
		newJobHandle.m_index = m_jobs.size();
		m_jobs.emplace_back(std::make_shared<Job>());
		m_jobs.back()->m_handle = newJobHandle;
	}
	const auto& newJob = m_jobs.at(newJobHandle.GetIndex());
	newJob->m_task = std::make_unique<std::function<void()>>(std::forward< std::function<void()>>(func));
	newJob->m_wake = false;
	newJob->m_children = filteredDependencies;
	newJob->m_recycled = false;
	newJob->m_finished = false;

	for (const auto& childHandle : filteredDependencies)
	{
		if (!childHandle.Valid()) continue;
		const auto& childJob = m_jobs.at(childHandle.GetIndex());
		childJob->m_parents.emplace_back(newJobHandle);
	}
	return newJobHandle;
}

void JobSystem::ExecuteJob(const JobHandle& jobHandle)
{
	if (!MainThreadCheck()) return;
	if (!jobHandle.Valid()) return;
	std::lock_guard jobManagementMutex(m_jobManagementMutex);
	std::vector<JobHandle> jobHandles;
	CollectDescendantsHelper(jobHandles, jobHandle);
	for (const auto& walker : jobHandles)
	{
		if (const auto& job = m_jobs[walker.GetIndex()]; !job->m_wake) {
			job->m_wake = true;
			CheckJobAvailableHelper(walker);
		}
	}
}

void JobSystem::Wait(const JobHandle& jobHandle)
{
	if (!MainThreadCheck()) return;
	if (!jobHandle.Valid()) return;
	const auto& job = m_jobs[jobHandle.GetIndex()];
	if (!job->m_parents.empty())
	{
		EVOENGINE_ERROR("Cannot wait job that's not root!");
		return;
	}
	std::vector<JobHandle> jobHandles;
	CollectDescendantsHelper(jobHandles, jobHandle);
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

void JobSystem::StopAllWorkers()
{
	if (!MainThreadCheck()) return;
	if (m_isDone) return;
	m_isDone = true; // give the waiting threads a command to finish
	{
		std::unique_lock lock(this->m_jobAvailabilityMutex);
		m_jobAvailableCondition.notify_all(); // stop all waiting threads
	}
	for (const auto& worker : m_workers)
	{ // wait for the computing threads to finish
		if (worker->joinable())
			worker->join();
	}
	// if there were no threads in the pool but some functors in the queue, the functors are not deleted by the
	// threads therefore delete them here
	m_workers.clear();
	m_flags.clear();
	m_idleThreadAmount = 0;
	m_isDone = false;
}

size_t JobSystem::IdleWorkerSize() const
{
	return m_idleThreadAmount;
}

JobSystem::JobSystem()
{
	m_mainThreadId = std::this_thread::get_id();
	ResizeWorker(1);
}

JobSystem::~JobSystem()
{
	StopAllWorkers();
}

void JobSystem::ResizeWorker(const size_t workerSize)
{
	if (!MainThreadCheck()) return;
	if (workerSize < 1)
	{
		EVOENGINE_ERROR("Worker size is zero!");
		return;
	}

	if (!m_isDone)
	{
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
}

size_t JobSystem::GetWorkerSize() const
{
	if (!MainThreadCheck()) return 0;
	return m_workers.size();
}
