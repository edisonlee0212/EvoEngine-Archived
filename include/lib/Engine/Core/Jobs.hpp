#pragma once
#include "ISingleton.hpp"
#include "ThreadPool.hpp"
namespace EvoEngine
{
	class Worker;
	typedef int WorkerHandle;

	class JobDependency
	{
		WorkerHandle m_workerHandle = -1;
		size_t m_version = 0;
		friend class Jobs;
	public:
		size_t GetVersion() const;
		WorkerHandle GetWorkerHandle() const;
		bool Valid() const;
	};

	class Worker
	{
		ThreadPool m_threads;
		WorkerHandle m_handle = -1;
		bool m_scheduled = false;
		std::vector<std::pair<std::shared_ptr<Worker>, size_t>> m_dependencies;
		std::vector<std::shared_future<void>> m_tasks;

		friend class Jobs;
		std::mutex m_mutex{};

		size_t m_version = 0;
	public:
		size_t GetVersion() const;
		Worker(size_t threadSize, WorkerHandle handle);
		void Wait();
		void Wait(size_t version);

		ThreadPool& RefThreadPool();
		const ThreadPool& PeekThreadPool() const;
		WorkerHandle GetHandle() const;
		bool Scheduled() const;

		void ScheduleTask(const std::function<void()>& func);
		void ScheduleTask(const std::vector<std::pair<std::shared_ptr<Worker>, size_t>>& dependencies, const std::function<void()>& func);
		void ScheduleParallelTasks(size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize);
		void ScheduleParallelTasks(const std::vector<std::pair<std::shared_ptr<Worker>, size_t>>& dependencies, size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize);

		void ScheduleParallelTasks(size_t size, const std::function<void(unsigned i)>& func, size_t threadSize);
		void ScheduleParallelTasks(const std::vector<std::pair<std::shared_ptr<Worker>, size_t>>& dependencies, size_t size, const std::function<void(unsigned i)>& func, size_t threadSize);
	};

	class Jobs final : ISingleton<Jobs>
	{
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
		static void ParallelFor(size_t size, const std::function<void(unsigned i)>& func, size_t threadSize = 0);
		static void ParallelFor(size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize = 0);
		static JobDependency AddParallelFor(size_t size, const std::function<void(unsigned i)>& func, size_t threadSize = 0);
		static JobDependency AddParallelFor(size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize = 0);

		static void ParallelFor(const std::vector<JobDependency>& dependencies, size_t size, const std::function<void(unsigned i)>& func, size_t threadSize = 0);
		static void ParallelFor(const std::vector<JobDependency>& dependencies, size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize = 0);
		static JobDependency AddParallelFor(const std::vector<JobDependency>& dependencies, size_t size, const std::function<void(unsigned i)>& func, size_t threadSize = 0);
		static JobDependency AddParallelFor(const std::vector<JobDependency>& dependencies, size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize = 0);

		static JobDependency AddTask(const std::vector<JobDependency>& dependencies, const std::function<void()>& func);
		static JobDependency AddTask(const std::function<void()>& func);
		static JobDependency PackTask(const std::vector<JobDependency>& dependencies);
		static void Wait(const JobDependency &jobDependency);
	};
} // namespace EvoEngine
