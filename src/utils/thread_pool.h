#ifndef THREAD_POOL_H
#define THREAD_POOL_H
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <queue>
#include <type_traits>
#include <utility>
#include <vector>

// C++11 版的 线程池
namespace threads
{
    class ThreadsGuard
    {
    public:
        ThreadsGuard(std::vector<std::thread>& v)
            : threads_(v)
        {
            
        }

        ~ThreadsGuard()
        {
            for (size_t i = 0; i != threads_.size(); ++i)
            {
                if (threads_[i].joinable())
                {
                    threads_[i].join();
                }
            }
        }
    private:
        ThreadsGuard(ThreadsGuard&& tg) = delete;
        ThreadsGuard& operator = (ThreadsGuard&& tg) = delete;

        ThreadsGuard(const ThreadsGuard&) = delete;
        ThreadsGuard& operator = (const ThreadsGuard&) = delete;
    private:
        std::vector<std::thread>& threads_;
    };


    class ThreadPool
    {
    public:
        typedef std::function<void()> task_type;

    public:
        explicit ThreadPool(int n = 0);

        ~ThreadPool()
        {
            stop();
            cond_.notify_all();
        }

        void stop()
        {
            stop_.store(true, std::memory_order_release);
        }

        template<class Function, class... Args>

        /*add(): 模板函数，接受一个函数和任意数量的参数，并将该函数包装为一个任务添加到任务队列中。
        返回一个std::future对象，该对象持有函数的返回值（如果有的话）*/
        std::future<typename std::result_of<Function(Args...)>::type> add(Function&&, Args&&...);

    private:
        ThreadPool(ThreadPool&&) = delete;
        ThreadPool& operator = (ThreadPool&&) = delete;
        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator = (const ThreadPool&) = delete;

    private:
        std::atomic<bool> stop_;
        std::mutex mtx_;                    //互斥锁
        std::condition_variable cond_;      //条件变量

        std::queue<task_type> tasks_;       //任务队列
        std::vector<std::thread> threads_;  //线程容器
        threads::ThreadsGuard tg_;
    };


    inline ThreadPool::ThreadPool(int n)
        : stop_(false)
        , tg_(threads_)
    {
        int nthreads = n;
        if (nthreads <= 0)
        {
            nthreads = std::thread::hardware_concurrency(); //系统硬件的并发线程数
            nthreads = (nthreads == 0 ? 2 : nthreads);  //如果并发线程数为0，则默认使用2个线程。
        }

        for (int i = 0; i != nthreads; ++i)  //创建线程
        {
            threads_.push_back(std::thread(
                [this]{ //每个线程执行一个 lambda 函数，该函数负责从任务队列中取出任务并执行它。
                    
                    //检查 stop_ 是否为 true。如果是，则线程返回。
                    while (!stop_.load(std::memory_order_acquire))
                    {
                        //等待任务队列不为空或 stop_ 为 true。
                        //这里使用条件变量 cond_ 进行等待，并使用互斥锁 mtx_ 进行保护。
                        task_type task;
                        {
                            std::unique_lock<std::mutex> ulk(this->mtx_);
                            this->cond_.wait(ulk, [this]{ return stop_.load(std::memory_order_acquire) || !this->tasks_.empty(); });
                            
                            //如果 stop_ 为 true，线程返回。
                            if (stop_.load(std::memory_order_acquire))
                                return;
                            //从任务队列中取出一个任务并执行它。
                            task = std::move(this->tasks_.front());
                            this->tasks_.pop();
                        }
                        task();
                    }
                }
            ));
        }
    }




    
    template<class Function, class... Args>
    
    //参数：该函数接受一个可调用对象（如函数、lambda表达式等）和任意数量的参数。
    //返回类型：返回一个 std::future 对象，该对象表示异步计算的结果。
    std::future<typename std::result_of<Function(Args...)>::type>
        ThreadPool::add(Function&& fcn, Args&&... args)
    {
        
        typedef typename std::result_of<Function(Args...)>::type return_type;

        //使用 std::packaged_task 封装传入的函数和参数，创建一个任务对象。
        typedef std::packaged_task<return_type()> task;

        //使用 std::make_shared 创建一个共享指针，指向任务对象。
        auto t = std::make_shared<task>(std::bind(std::forward<Function>(fcn), std::forward<Args>(args)...));
        
        //添加任务到队列
        auto ret = t->get_future();
        {
            //使用互斥锁保护代码块，确保在多线程环境中对任务队列的操作是安全的。
            std::lock_guard<std::mutex> lg(mtx_);
            //检查 stop_ 是否为 true。如果是，则抛出一个运行时错误。
            if (stop_.load(std::memory_order_acquire))
                throw std::runtime_error("thread pool has stopped");

            /*将任务添加到任务队列中。这里使用了一个 lambda 函数来执行任务，
              该 lambda 函数接受任务对象作为参数。*/
            tasks_.emplace([t]{(*t)(); });
        }
        //通知等待的线程
        cond_.notify_one();
        //返回结果
        return ret;
    }
}

#endif  /* THREAD_POOL_H */