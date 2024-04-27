#pragma once
#include "ISingleton.hpp"
#include "ThreadPool.hpp"
namespace EvoEngine
{
	class JobHandle
	{
		int m_index = -1;
		size_t m_version = 0;
		friend class Jobs;
	public:
		[[nodiscard]] size_t GetVersion() const;
		[[nodiscard]] int GetIndex() const;
		[[nodiscard]] bool Valid() const;
	};

	class TaskSemaphore {
		size_t m_availability = 0;
		std::mutex m_updateMutex;
		std::condition_variable m_cv;

	public:
		void Reset(size_t availability = 0);
		void Acquire();
		void Release();
	};

	class Jobs final : ISingleton<Jobs>
	{
		class JobHolder
		{
			bool m_sleeping = true;
			bool m_finished = false;
			TaskSemaphore m_taskSemaphore;
			std::mutex m_statusLock;
			std::vector<JobHandle> m_dependencies;
			std::unique_ptr<std::function<void()>> m_job;
			size_t m_version = 0;
			friend class Jobs;
			int m_index;
		public:
			void Wait();
		};
		std::queue<int> m_availableTaskHolders;
		std::mutex m_managementMutex;
		std::thread::id m_mainThreadId;
		std::vector<std::unique_ptr<std::thread>> m_threads;
		std::vector<std::shared_ptr<std::atomic<bool>>> m_flags;
		std::atomic<bool> m_isDone;
		std::atomic<bool> m_isStop;
		std::atomic<int> m_waitingThreadAmount; // how many threads are waiting

		std::mutex m_mutex;
		std::condition_variable m_threadPoolCondition;
		void SetThread(size_t i);
		TaskSemaphore m_availableJobHolderSemaphore;
		std::vector<std::shared_ptr<JobHolder>> m_jobHolders;
		JobHandle PushJob(const std::vector<JobHandle>& dependencies, std::function<void()>&& func);

		std::shared_ptr<JobHolder> GetJobHolder(const JobHandle& jobHandle);

		void WakeJob(const JobHandle& jobHandle);
		std::unique_ptr<std::function<void()>> TryPopJob(JobHandle& jobHandle);

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
