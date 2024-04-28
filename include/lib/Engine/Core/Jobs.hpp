#pragma once
#include "ISingleton.hpp"
#include "ThreadPool.hpp"
namespace EvoEngine
{
	class JobHandle
	{
		int m_index = -1;
		friend class Jobs;
	public:
		[[nodiscard]] int GetIndex() const;
		[[nodiscard]] bool Valid() const;
	};

	class JobSystemSemaphore {
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
		class JobSystem
		{
			struct Job
			{
				std::vector<JobHandle> m_parents;
				std::vector<JobHandle> m_children;
				JobHandle m_handle;
				JobSystemSemaphore m_finishedSemaphore;
				bool m_recycled = false;
				bool m_finished = false;
				bool m_wake = false;
				std::unique_ptr<std::function<void()>> m_task;
			};

			std::vector<std::shared_ptr<Job>> m_jobs;
			std::queue<JobHandle> m_recycledJobs;
			EvoEngine::JobSystem::ThreadQueue<JobHandle> m_availableJobPool;

			std::mutex m_jobManagementMutex;
			void ReportFinish(const JobHandle& jobHandle);
			void InitializeWorker(size_t workerIndex);

			void CollectDescendants(std::vector<JobHandle>& jobs, const JobHandle& walker);
		public:
			JobHandle PushJob(const std::vector<JobHandle>& dependencies, std::function<void()>&& func);
			void ExecuteJob(const JobHandle& jobHandle);
			void Wait(const JobHandle& jobHandle);
			std::atomic<int> m_waitingThreadAmount; // how many threads are waiting
			std::vector<std::unique_ptr<std::thread>> m_workers;
			std::vector<std::shared_ptr<std::atomic<bool>>> m_flags;

			std::mutex m_jobAvailabilityMutex;
			std::condition_variable m_jobAvailableCondition;
			void Initialize(size_t workerSize);
			std::thread::id m_mainThreadId;
		};
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
