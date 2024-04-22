#pragma once
#include "ISingleton.hpp"
#include "ThreadPool.hpp"
namespace EvoEngine
{
	typedef int WorkerHandle;

	struct WorkerDependency
	{
		WorkerHandle m_workerHandle = -1;
		size_t m_version = 0;
	};

	class Worker
	{
		ThreadPool m_threads;
		WorkerHandle m_handle = -1;
		bool m_scheduled = false;
		std::vector<WorkerDependency> m_dependencies;
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
		void ScheduleTask(const std::vector<WorkerHandle>& dependencies, const std::function<void()>& func);
		void ScheduleParallelTasks(size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize);
		void ScheduleParallelTasks(const std::vector<WorkerHandle>& dependencies, size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize);

		void ScheduleParallelTasks(size_t size, const std::function<void(unsigned i)>& func, size_t threadSize);
		void ScheduleParallelTasks(const std::vector<WorkerHandle>& dependencies, size_t size, const std::function<void(unsigned i)>& func, size_t threadSize);
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
		static WorkerHandle AddParallelFor(size_t size, const std::function<void(unsigned i)>& func, size_t threadSize = 0);
		static WorkerHandle AddParallelFor(size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize = 0);

		static void ParallelFor(const std::vector<WorkerHandle>& dependencies, size_t size, const std::function<void(unsigned i)>& func, size_t threadSize = 0);
		static void ParallelFor(const std::vector<WorkerHandle>& dependencies, size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize = 0);
		static WorkerHandle AddParallelFor(const std::vector<WorkerHandle>& dependencies, size_t size, const std::function<void(unsigned i)>& func, size_t threadSize = 0);
		static WorkerHandle AddParallelFor(const std::vector<WorkerHandle>& dependencies, size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t threadSize = 0);

		static WorkerHandle AddTask(const std::vector<WorkerHandle>& dependencies, const std::function<void()>& func);
		static WorkerHandle AddTask(const std::function<void()>& func);
		static WorkerHandle PackTask(const std::vector<WorkerHandle>& dependencies);
		static void Wait(WorkerHandle workerHandle);
	};
} // namespace EvoEngine
