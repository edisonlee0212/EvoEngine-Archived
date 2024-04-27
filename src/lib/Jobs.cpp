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

void TaskSemaphore::Reset(const size_t availability)
{
	m_availability = availability;
}

void TaskSemaphore::Acquire()
{
	std::unique_lock lk(m_updateMutex);
	m_cv.wait(lk, [this]() { return m_availability > 0; });
	m_availability--;
}

void TaskSemaphore::Release()
{
	std::unique_lock lk(m_updateMutex);
	m_availability++;
	m_cv.notify_one();
}

void Jobs::JobHolder::Wait()
{
	m_taskSemaphore.Acquire();
	m_finished = true;
	
}

void Jobs::ReportFinish(const JobHandle& jobHandle)
{
	std::lock_guard lock(m_managementMutex);
	const auto& jobHolder = m_jobHolders.at(jobHandle.GetIndex());
	for(const auto& triggerIndex : jobHolder->m_triggers)
	{
		const auto trigger = m_jobHolders.at(triggerIndex);
		for(int index = 0; index < trigger->m_dependencies.size(); index++)
		{
			if(trigger->m_dependencies.at(index) == trigger->m_index)
			{
				trigger->m_dependencies.at(index) = trigger->m_dependencies.back();
				trigger->m_dependencies.pop_back();
				break;
			}
		}
	}
}

JobHandle Jobs::PushJob(const std::vector<JobHandle>& dependencies, std::function<void()>&& func)
{
	std::lock_guard lock(m_managementMutex);
	std::shared_ptr<JobHolder> jobHolder;
	bool foundFinishedJobHolder = false;

	m_availableJobHolderSemaphore.Acquire();

	for(const auto& candidate : m_jobHolders)
	{
		std::lock_guard statusLock(candidate->m_statusLock);
		if(candidate->m_finished)
		{
			jobHolder = candidate;
			foundFinishedJobHolder = true;
		}
	}
	if(!foundFinishedJobHolder)
	{
		const int index = m_jobHolders.size();
		m_jobHolders.emplace_back();
		jobHolder = m_jobHolders.back() = std::make_shared<JobHolder>();
		jobHolder->m_index = index;

	}
	for(const auto& dependencyHandle : dependencies)
	{
		const auto& dependency = m_jobHolders.at(dependencyHandle.GetIndex());
		jobHolder->m_dependencies.emplace_back(dependencyHandle.m_index);
		dependency->m_triggers.emplace_back(jobHolder->m_index);
	}
	jobHolder->m_sleeping = true;
	jobHolder->m_finished = false;
	jobHolder->m_job = std::make_unique<std::function<void()>>(std::forward<std::function<void()>>(func));
	JobHandle retVal;
	retVal.m_index = jobHolder->m_index;
	return retVal;
}

void Jobs::WakeJob(const int jobIndex)
{
	const auto& jobHolder = m_jobHolders.at(jobIndex);
	for (const auto& dep : jobHolder->m_dependencies) WakeJob(dep);
	std::lock_guard lock(m_managementMutex);
	
	if(!jobHolder->m_sleeping) return;
	if(!jobHolder->m_job) return;
	jobHolder->m_sleeping = false;
}

std::unique_ptr<std::function<void()>> Jobs::TryPopJob(JobHandle& jobHandle)
{
	std::lock_guard lock(m_managementMutex);
	if (m_jobHolders.empty()) return {};

	for (const auto& candidate : m_jobHolders)
	{
		std::lock_guard statusLock(candidate->m_statusLock);
		if(candidate->m_finished) continue;
		if (candidate->m_sleeping) continue;
		if (!candidate->m_job) continue;
		if(candidate->m_dependencies.empty())
		{
			jobHandle.m_index = candidate->m_index;
			return std::forward<std::unique_ptr<std::function<void()>>>(candidate->m_job);
		}
	}
	return {};
}



void Jobs::SetThread(const size_t i)
{
	std::shared_ptr flag(m_flags[i]); // a copy of the shared ptr to the flag
	auto threadFunc = [this, flag /* a copy of the shared ptr to the flag */]() {
		std::atomic<bool>& _flag = *flag;
		JobHandle jobHandle;
		std::unique_ptr<std::function<void()>> job;

		bool isPop = false;
		while (true)
		{
			while (isPop)
			{ // if there is anything in the queue
				
				(*job)();
				job.reset();
				const auto& jobHolder = m_jobHolders.at(jobHandle.GetIndex());
				ReportFinish(jobHandle);
				jobHolder->m_taskSemaphore.Release();
				m_availableJobHolderSemaphore.Release();
				if (_flag)
					return; // the thread is wanted to stop, return even if the queue is not empty yet
				job = TryPopJob(jobHandle);
				isPop = job.get();
			}
			// the queue is empty here, wait for the next command
			std::unique_lock lock(m_mutex);
			++m_waitingThreadAmount;
			m_threadPoolCondition.wait(lock, [this, &job, &isPop, &jobHandle, &_flag]() {
				job = TryPopJob(jobHandle);
				isPop = job.get();
				return isPop || m_isDone || _flag;
				});
			--m_waitingThreadAmount;
			if (!isPop)
				return; // if the queue is empty and this->isDone == true or *flag then return
		}
		};
	m_threads[i].reset(new std::thread(threadFunc)); // compiler may not support std::make_unique()
}

size_t Jobs::GetWorkerSize()
{
	const auto& jobs = GetInstance();
	return jobs.m_threads.size();
}


void Jobs::Initialize(const size_t workerSize)
{
	auto& jobs = GetInstance();
	jobs.m_availableJobHolderSemaphore.Reset(128);
	jobs.m_mainThreadId = std::this_thread::get_id();

	const auto oldNThreads = jobs.m_threads.size();
	if (oldNThreads <= workerSize)
	{ // if the number of threads is increased
		jobs.m_threads.resize(workerSize);
		jobs.m_flags.resize(workerSize);

		for (size_t i = oldNThreads; i < workerSize; ++i)
		{
			jobs.m_flags[i] = std::make_shared<std::atomic<bool>>(false);
			jobs.SetThread(i);
		}
	}
	else
	{ // the number of threads is decreased
		for (size_t i = oldNThreads - 1; i >= workerSize; --i)
		{
			*jobs.m_flags[i] = true; // this thread will finish
			jobs.m_threads[i]->detach();
		}
		{
			// stop the detached threads that were waiting
			std::unique_lock lock(jobs.m_mutex);
			jobs.m_threadPoolCondition.notify_all();
		}
		jobs.m_threads.resize(workerSize); // safe to delete because the threads are detached
		jobs.m_flags.resize(workerSize);   // safe to delete because the threads have copies of shared_ptr of the
		// flags, not originals
	}

}

void Jobs::RunParallelFor(const size_t size, std::function<void(unsigned i)>&& func, size_t workerSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return;
	}
	if (workerSize == 0) workerSize = jobs.m_threads.size();
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
		jobHandles.emplace_back(jobs.PushJob({}, std::forward<std::function<void()>>(work)));
	}
	Wait(Combine(jobHandles));
}

void Jobs::RunParallelFor(const size_t size, std::function<void(unsigned i, unsigned threadIndex)>&& func, size_t workerSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return;
	}
	if (workerSize == 0) workerSize = jobs.m_threads.size();
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
		jobHandles.emplace_back(jobs.PushJob({}, std::forward<std::function<void()>>(work)));
	}
	Wait(Combine(jobHandles));
}

JobHandle Jobs::ScheduleParallelFor(const size_t size, std::function<void(unsigned i)>&& func, size_t workerSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return {};
	}
	if (workerSize == 0) workerSize = jobs.m_threads.size();
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
		jobHandles.emplace_back(jobs.PushJob({}, std::forward<std::function<void()>>(work)));
	}
	return Combine(jobHandles);
}

JobHandle Jobs::ScheduleParallelFor(const size_t size, std::function<void(unsigned i, unsigned threadIndex)>&& func,
	size_t workerSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return {};
	}
	if (workerSize == 0) workerSize = jobs.m_threads.size();
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
		jobHandles.emplace_back(jobs.PushJob({}, std::forward<std::function<void()>>(work)));
	}
	return Combine(jobHandles);
}

void Jobs::RunParallelFor(const std::vector<JobHandle>& dependencies, const size_t size,
	std::function<void(unsigned i)>&& func, size_t workerSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return;
	}
	if (workerSize == 0) workerSize = jobs.m_threads.size();
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
		jobHandles.emplace_back(jobs.PushJob(dependencies, std::forward<std::function<void()>>(work)));
	}
	Wait(Combine(jobHandles));
}

void Jobs::RunParallelFor(const std::vector<JobHandle>& dependencies, const size_t size,
	std::function<void(unsigned i, unsigned threadIndex)>&& func, size_t workerSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return;
	}
	if (workerSize == 0) workerSize = jobs.m_threads.size();
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
		jobHandles.emplace_back(jobs.PushJob(dependencies, std::forward<std::function<void()>>(work)));
	}
	Wait(Combine(jobHandles));
}

JobHandle Jobs::ScheduleParallelFor(const std::vector<JobHandle>& dependencies, const size_t size,
	std::function<void(unsigned i)>&& func, size_t workerSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return {};
	}
	if (workerSize == 0) workerSize = jobs.m_threads.size();
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
		jobHandles.emplace_back(jobs.PushJob(dependencies, std::forward<std::function<void()>>(work)));
	}
	return Combine(jobHandles);
}

JobHandle Jobs::ScheduleParallelFor(const std::vector<JobHandle>& dependencies, const size_t size,
	std::function<void(unsigned i, unsigned threadIndex)>&& func, size_t workerSize)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return {};
	}
	if (workerSize == 0) workerSize = jobs.m_threads.size();
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
		jobHandles.emplace_back(jobs.PushJob(dependencies, std::forward<std::function<void()>>(work)));
	}
	return Combine(jobHandles);
}

JobHandle Jobs::Run(const std::vector<JobHandle>& dependencies, std::function<void()>&& func)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return {};
	}

	return jobs.PushJob(dependencies, std::forward<std::function<void()>>(func));
}

JobHandle Jobs::Run(std::function<void()>&& func)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return {};
	}
	return jobs.PushJob({}, std::forward<std::function<void()>>(func));
}

JobHandle Jobs::Combine(const std::vector<JobHandle>& dependencies)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!!");
		return {};
	}

	return jobs.PushJob({}, []() {});
}

void Jobs::Execute(const JobHandle& jobHandle)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!");
		return;
	}
	jobs.WakeJob(jobHandle.GetIndex());
	std::unique_lock lock(jobs.m_mutex);
	jobs.m_threadPoolCondition.notify_one();
}

void Jobs::Wait(const JobHandle & jobHandle)
{
	auto& jobs = GetInstance();
	if (jobs.m_mainThreadId != std::this_thread::get_id())
	{
		EVOENGINE_ERROR("Jobs: Not on main thread!");
		return;
	}
	Execute(jobHandle);
	jobs.m_jobHolders.at(jobHandle.GetIndex())->Wait();
}
