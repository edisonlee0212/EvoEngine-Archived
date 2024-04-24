#pragma once
#include "ISingleton.hpp"
namespace EvoEngine
{
	typedef int WorkerHandle;

	class JobHandle
	{
		WorkerHandle m_workerHandle = -1;
		size_t m_version = 0;
		friend class Jobs;
	public:
		[[nodiscard]] size_t GetVersion() const;
		[[nodiscard]] WorkerHandle GetWorkerHandle() const;
		[[nodiscard]] bool Valid() const;
	};


	class Jobs final : ISingleton<Jobs>
	{
		class Worker
		{
			std::vector<std::condition_variable> m_tasksCondition;
			std::vector<std::unique_ptr<std::thread>> m_threads;

			WorkerHandle m_handle = -1;
			bool m_executing = false;
			bool m_scheduled = false;
			std::vector<std::pair<std::shared_ptr<Worker>, size_t>> m_dependencies;

			std::vector<std::unique_ptr<std::function<void()>>> m_works;
			std::vector<std::shared_future<void>> m_results;

			friend class Jobs;
			std::mutex m_schedulerMutex{};
			std::mutex m_conditionVariableMutex{};
			size_t m_threadSize = 0;

			size_t m_version = 0;
		public:
			[[nodiscard]] size_t GetVersion() const;
			Worker(size_t threadSize, WorkerHandle handle);

			void Execute();
			void Wait();
			void Wait(size_t version);

			[[nodiscard]] WorkerHandle GetHandle() const;
			[[nodiscard]] bool Executing() const;
			[[nodiscard]] bool Scheduled() const;

			void ScheduleJob(const std::function<void()>& func);
			void ScheduleJob(const std::vector<std::pair<std::shared_ptr<Worker>, size_t>>& dependencies, const std::function<void()>& func);
			void ScheduleParallelJobs(size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func);
			void ScheduleParallelJobs(const std::vector<std::pair<std::shared_ptr<Worker>, size_t>>& dependencies, size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func);

			void ScheduleParallelJobs(size_t size, const std::function<void(unsigned i)>& func);
			void ScheduleParallelJobs(const std::vector<std::pair<std::shared_ptr<Worker>, size_t>>& dependencies, size_t size, const std::function<void(unsigned i)>& func);
		};
		std::vector<std::shared_ptr<Worker>> m_workers;
		std::vector<std::queue<WorkerHandle>> m_availableWorker;

		friend class Worker;

		size_t m_maxThreadSize = 16;
		size_t m_defaultThreadSize = 8;

		std::thread::id m_mainThreadId;
		std::mutex m_workerManagementMutex{};

		WorkerHandle GetAvailableWorker(size_t threadSize);

	public:
		static size_t GetDefaultThreadSize();
		static size_t GetMaxThreadSize();
		static void Initialize(size_t defaultThreadSize, size_t maxThreadSize);
		static void RunParallelFor(size_t size, const std::function<void(unsigned i)>& func, size_t threadSize = 0);
		static void RunParallelFor(size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize = 0);
		static JobHandle ScheduleParallelFor(size_t size, const std::function<void(unsigned i)>& func, size_t threadSize = 0);
		static JobHandle ScheduleParallelFor(size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize = 0);

		static void RunParallelFor(const std::vector<JobHandle>& dependencies, size_t size, const std::function<void(unsigned i)>& func, size_t threadSize = 0);
		static void RunParallelFor(const std::vector<JobHandle>& dependencies, size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize = 0);
		static JobHandle ScheduleParallelFor(const std::vector<JobHandle>& dependencies, size_t size, const std::function<void(unsigned i)>& func, size_t threadSize = 0);
		static JobHandle ScheduleParallelFor(const std::vector<JobHandle>& dependencies, size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize = 0);

		static JobHandle Run(const std::vector<JobHandle>& dependencies, const std::function<void()>& func);
		static JobHandle Run(const std::function<void()>& func);
		static JobHandle Combine(const std::vector<JobHandle>& dependencies);

		static void Execute(const JobHandle& jobDependency);
		static void Wait(const JobHandle &jobDependency);
	};
} // namespace EvoEngine
