#pragma once
namespace evo_engine
{
	class JobHandle
	{
		int m_index = -1;
		friend class JobSystem;
	public:
		[[nodiscard]] int GetIndex() const;
		[[nodiscard]] bool Valid() const;
	};

	class JobSystem
	{
		class JobSystemSemaphore {
			size_t m_availability = 0;
			std::mutex m_updateMutex;
			std::condition_variable m_cv;

		public:
			void Reset(size_t availability = 0);
			void Acquire();
			void Release();
		};
		struct Job
		{
			std::vector<JobHandle> m_parents;
			std::vector<JobHandle> m_children;
			JobHandle m_handle;
			JobSystemSemaphore m_finishedSemaphore;
			bool m_recycled = false;
			bool m_finished = false;
			bool m_wake = false;
			std::function<void()> m_task;
		};
		class JobPool {
		public:
			void Push(std::pair<JobHandle, std::function<void()>>&& job);
			// deletes the retrieved element, do not use for non integral types
			[[nodiscard]] bool Pop(std::pair<JobHandle, std::function<void()>>& job);
			[[nodiscard]] bool Empty();
		private:
			std::queue<std::pair<JobHandle, std::function<void()>>> m_queue;
			std::mutex m_mutex;
		};
		std::vector<std::shared_ptr<Job>> m_jobs;
		std::queue<JobHandle> m_recycledJobs;
		JobPool m_availableJobPool;

		[[nodiscard]] bool MainThreadCheck() const;
		void CollectDescendantsHelper(std::vector<JobHandle>& jobs, const JobHandle& walker);
		void CheckJobAvailableHelper(const JobHandle& jobHandle);
		void ReportFinish(const JobHandle& jobHandle);
		void InitializeWorker(size_t workerIndex);
		std::atomic<int> m_idleThreadAmount; // how many threads are waiting
		std::vector<std::unique_ptr<std::thread>> m_workers;
		std::vector<std::shared_ptr<std::atomic<bool>>> m_flags;
		std::atomic<bool> m_isDone;

		std::mutex m_jobManagementMutex;
		std::mutex m_jobAvailabilityMutex;
		std::condition_variable m_jobAvailableCondition;

		std::thread::id m_mainThreadId;
	public:
		void StopAllWorkers();
		[[nodiscard]] size_t IdleWorkerSize() const;
		JobSystem();
		~JobSystem();
		void ResizeWorker(size_t workerSize);
		[[nodiscard]] size_t GetWorkerSize() const;
		[[nodiscard]] JobHandle PushJob(const std::vector<JobHandle>& dependencies, std::function<void()>&& func);
		void ExecuteJob(const JobHandle& jobHandle);
		void Wait(const JobHandle& jobHandle);
	};
}
