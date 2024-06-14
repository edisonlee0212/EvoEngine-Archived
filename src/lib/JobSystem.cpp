#include "JobSystem.hpp"

#include "Console.hpp"

using namespace evo_engine;

int JobHandle::GetIndex() const {
  return index_;
}

bool JobHandle::Valid() const {
  return index_ >= 0;
}

void JobSystem::JobSystemSemaphore::Reset(const size_t availability) {
  availability_ = availability;
}

void JobSystem::JobSystemSemaphore::Acquire() {
  std::unique_lock lk(update_mutex_);
  cv_.wait(lk, [this]() {
    return availability_ > 0;
  });
  availability_--;
}

void JobSystem::JobSystemSemaphore::Release() {
  std::unique_lock lk(update_mutex_);
  availability_++;
  cv_.notify_one();
}

inline void JobSystem::JobPool::Push(std::pair<JobHandle, std::function<void()>>&& job) {
  std::unique_lock lock(pool_mutex_);
  job_queue_.push(std::forward<std::pair<JobHandle, std::function<void()>>>(job));
}

inline bool JobSystem::JobPool::Pop(std::pair<JobHandle, std::function<void()>>& job) {
  std::unique_lock lock(pool_mutex_);
  if (job_queue_.empty())
    return false;
  job = std::forward<std::pair<JobHandle, std::function<void()>>>(job_queue_.front());
  job_queue_.pop();
  return true;
}

inline bool JobSystem::JobPool::Empty() {
  std::unique_lock lock(pool_mutex_);
  return job_queue_.empty();
}

void JobSystem::CheckJobAvailableHelper(const JobHandle& job_handle) {
  const auto& job = jobs_.at(job_handle.GetIndex());
  bool jobReadyToRun = job->wake;
  if (jobReadyToRun) {
    for (const auto& childHandle : job->children) {
      if (!jobs_[childHandle.index_]->finished) {
        jobReadyToRun = false;
        break;
      }
    }
  }
  if (jobReadyToRun) {
    available_job_pool_.Push(std::make_pair(job_handle, std::forward<std::function<void()>>(job->task)));
    std::unique_lock lock(this->job_availability_mutex_);
    job_available_condition_.notify_one();
  }
}
void JobSystem::ReportFinish(const JobHandle& job_handle) {
  std::lock_guard jobManagementMutex(job_management_mutex_);
  const auto jobIndex = job_handle.GetIndex();
  const auto& job = jobs_[jobIndex];
  job->finished = true;
  job->finished_semaphore.Release();
  for (const auto& parent : job->parents) {
    CheckJobAvailableHelper(parent);
  }
}

void JobSystem::InitializeWorker(const size_t worker_index) {
  std::shared_ptr flag(flags_[worker_index]);  // a copy of the shared ptr to the flag
  auto threadFunc = [this, flag /* a copy of the shared ptr to the flag */]() {
    std::atomic<bool>& flagPtr = *flag;
    std::pair<JobHandle, std::function<void()>> task;
    bool isPop = available_job_pool_.Pop(task);
    while (true) {
      while (isPop) {  // if there is anything in the queue
        task.second();
        ReportFinish(task.first);
        // if (flagPtr)
        //	return; // the thread is wanted to stop, return even if the queue is not empty yet
        isPop = available_job_pool_.Pop(task);
        if (!isPop)
          task = {};
      }
      // the queue is empty here, wait for the next command
      std::unique_lock lock(job_availability_mutex_);
      ++idle_thread_amount_;
      job_available_condition_.wait(lock, [this, &isPop, &task, &flagPtr]() {
        isPop = available_job_pool_.Pop(task);
        return isPop || is_done_ || flagPtr;
      });
      --idle_thread_amount_;
      if (!isPop)
        return;  // if the queue is empty and is_done_ == true or *flag then return
    }
  };
  workers_[worker_index].reset(new std::thread(threadFunc));  // compiler may not support std::make_unique()
}

bool JobSystem::MainThreadCheck() const {
  if (main_thread_id_ == std::this_thread::get_id()) {
    return true;
  }
  EVOENGINE_ERROR("Jobs: Not on main thread!!");
  return false;
}

void JobSystem::CollectDescendantsHelper(std::vector<JobHandle>& jobs, const JobHandle& walker) {
  jobs.emplace_back(walker);
  for (const auto& i : jobs_.at(walker.GetIndex())->children) {
    CollectDescendantsHelper(jobs, i);
  }
}

JobHandle JobSystem::PushJob(const std::vector<JobHandle>& dependencies, std::function<void()>&& func) {
  if (!MainThreadCheck())
    return {};
  std::lock_guard jobManagementMutex(job_management_mutex_);
  std::vector<JobHandle> filteredDependencies;
  for (const auto& dependency : dependencies) {
    if (!dependency.Valid())
      continue;
    filteredDependencies.emplace_back(dependency);
  }
  std::vector<JobHandle> descendants;
  for (const auto& dependency : filteredDependencies) {
    CollectDescendantsHelper(descendants, dependency);
  }
  JobHandle newJobHandle;
  if (!recycled_jobs_.empty()) {
    newJobHandle = recycled_jobs_.front();
    recycled_jobs_.pop();
  } else {
    newJobHandle.index_ = jobs_.size();
    jobs_.emplace_back(std::make_shared<Job>());
    jobs_.back()->job_handle = newJobHandle;
  }
  const auto& newJob = jobs_.at(newJobHandle.GetIndex());
  newJob->task = std::forward<std::function<void()>>(func);
  newJob->wake = false;
  newJob->children = filteredDependencies;
  newJob->recycled = false;
  newJob->finished = false;

  for (const auto& childHandle : filteredDependencies) {
    if (!childHandle.Valid())
      continue;
    const auto& childJob = jobs_.at(childHandle.GetIndex());
    childJob->parents.emplace_back(newJobHandle);
  }
  return newJobHandle;
}

void JobSystem::ExecuteJob(const JobHandle& job_handle) {
  if (!MainThreadCheck())
    return;
  if (!job_handle.Valid())
    return;
  std::lock_guard jobManagementMutex(job_management_mutex_);
  std::vector<JobHandle> jobHandles;
  CollectDescendantsHelper(jobHandles, job_handle);
  for (const auto& walker : jobHandles) {
    if (const auto& job = jobs_[walker.GetIndex()]; !job->wake) {
      job->wake = true;
      CheckJobAvailableHelper(walker);
    }
  }
}

void JobSystem::Wait(const JobHandle& job_handle) {
  if (!MainThreadCheck())
    return;
  if (!job_handle.Valid())
    return;
  const auto& job = jobs_[job_handle.GetIndex()];
  if (!job->parents.empty()) {
    EVOENGINE_ERROR("Cannot wait job that's not root!");
    return;
  }
  std::vector<JobHandle> jobHandles;
  CollectDescendantsHelper(jobHandles, job_handle);
  for (const auto& walker : jobHandles) {
    const auto& job = jobs_[walker.GetIndex()];
    if (!job->recycled) {
      job->finished_semaphore.Acquire();
      recycled_jobs_.emplace(walker);
      job->recycled = true;
      job->parents.clear();
      job->children.clear();
    }
  }
}

void JobSystem::StopAllWorkers() {
  if (!MainThreadCheck())
    return;
  if (is_done_)
    return;
  is_done_ = true;  // give the waiting threads a command to finish
  {
    std::unique_lock lock(this->job_availability_mutex_);
    job_available_condition_.notify_all();  // stop all waiting threads
  }
  for (const auto& worker : workers_) {  // wait for the computing threads to finish
    if (worker->joinable())
      worker->join();
  }
  // if there were no threads in the pool but some functors in the queue, the functors are not deleted by the
  // threads therefore delete them here
  workers_.clear();
  flags_.clear();
  idle_thread_amount_ = 0;
  is_done_ = false;
}

size_t JobSystem::IdleWorkerSize() const {
  return idle_thread_amount_;
}

JobSystem::JobSystem() {
  main_thread_id_ = std::this_thread::get_id();
  ResizeWorker(1);
}

JobSystem::~JobSystem() {
  StopAllWorkers();
}

void JobSystem::ResizeWorker(const size_t worker_size) {
  if (!MainThreadCheck())
    return;
  if (worker_size < 1) {
    EVOENGINE_ERROR("Worker size is zero!");
    return;
  }

  if (!is_done_) {
    const auto oldNThreads = workers_.size();
    if (oldNThreads <= worker_size) {  // if the number of threads is increased
      workers_.resize(worker_size);
      flags_.resize(worker_size);

      for (size_t i = oldNThreads; i < worker_size; ++i) {
        flags_[i] = std::make_shared<std::atomic<bool>>(false);
        InitializeWorker(i);
      }
    } else {  // the number of threads is decreased
      for (size_t i = oldNThreads - 1; i >= worker_size; --i) {
        *flags_[i] = true;  // this thread will finish
        workers_[i]->detach();
      }
      {
        // stop the detached threads that were waiting
        std::unique_lock lock(job_availability_mutex_);
        job_available_condition_.notify_all();
      }
      workers_.resize(worker_size);  // safe to delete because the threads are detached
      flags_.resize(worker_size);    // safe to delete because the threads have copies of shared_ptr of the
                                     // flags, not originals
    }
  }
}

size_t JobSystem::GetWorkerSize() const {
  if (!MainThreadCheck())
    return 0;
  return workers_.size();
}
