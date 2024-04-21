#pragma once
#include "ISingleton.hpp"
#include "ThreadPool.hpp"
namespace EvoEngine
{
	typedef int WorkerHandle;
	class Worker
	{
		ThreadPool m_threads;
		WorkerHandle m_handle = -1;
		bool m_scheduled = false;
		std::vector<WorkerHandle> m_dependencies;
		std::vector<std::shared_future<void>> m_tasks;

		static void CheckLoopHelper(std::vector<WorkerHandle>& collectedWorkers, WorkerHandle currentWorker);
		bool CheckLoop(const std::vector<WorkerHandle>& dependencies) const;

		friend class Jobs;
	public:
		void Wait();
		ThreadPool& RefThreadPool();
		const ThreadPool& PeekThreadPool() const;
		WorkerHandle GetHandle() const;
		bool Scheduled() const;

		void ScheduleTask(const std::function<void()>& func);
		void ScheduleTask(const std::vector<WorkerHandle>& dependencies, const std::function<void()>& func);
		void ScheduleParallelTasks(size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t workerSize = 10);
		void ScheduleParallelTasks(const std::vector<WorkerHandle>& dependencies, size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t workerSize = 10);

		void ScheduleParallelTasks(size_t size, const std::function<void(unsigned i)>& func, size_t workerSize = 10);
		void ScheduleParallelTasks(const std::vector<WorkerHandle>& dependencies, size_t size, const std::function<void(unsigned i)>& func, size_t workerSize = 10);
	};

	class Jobs final : ISingleton<Jobs>
	{
		std::vector<std::shared_ptr<Worker>> m_workers;
		std::queue<WorkerHandle> m_availableWorker;

		friend class Worker;
	public:
		static void ParallelFor(size_t size, const std::function<void(unsigned i)>& func, size_t workerSize = 10);
		static void ParallelFor(size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t workerSize = 10);
		static WorkerHandle AddParallelFor(size_t size, const std::function<void(unsigned i)>& func, size_t workerSize = 10);
		static WorkerHandle AddParallelFor(size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t workerSize = 10);

		static void ParallelFor(const std::vector<WorkerHandle>& dependencies, size_t size, const std::function<void(unsigned i)>& func, size_t workerSize = 10);
		static void ParallelFor(const std::vector<WorkerHandle>& dependencies, size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t workerSize = 10);
		static WorkerHandle AddParallelFor(const std::vector<WorkerHandle>& dependencies, size_t size, const std::function<void(unsigned i)>& func, size_t workerSize = 10);
		static WorkerHandle AddParallelFor(const std::vector<WorkerHandle>& dependencies, size_t size, const std::function<void(unsigned i, unsigned threadIndex)>& func, size_t workerSize = 10);

		static WorkerHandle AddTask(const std::function<void()>& func);

		static void Wait(WorkerHandle workerHandle);
	};
} // namespace EvoEngine
