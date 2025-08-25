#define DllExport __declspec(dllexport)
extern "C"
{
    DllExport int server_get_interface_version();

    DllExport int terminate_server(void *ctx);
    DllExport void* create_server_context();
    DllExport int server_main(void* ctx, int argc, char** argv);

    DllExport size_t server_get_output(void* ctx, int id_task, char* dst, size_t dst_size, unsigned long timeout);
    DllExport bool server_get_output_done(void* ctx, int id_task);

    DllExport int server_post_completions(void* ctx, const char* req_body);
    DllExport int server_post_embeddings(void* ctx, const char* req_body);
    DllExport int server_post_reranking(void* ctx, const char* req_body);
    DllExport int server_post_tokenize(void* ctx, const char* req_body);

    DllExport void server_stop_output(void* ctx, int id_task);
    DllExport bool server_wait_for_ready(void* ctx, unsigned long timeout);

    DllExport int server_save_slot(void* ctx, const char* req_body, int id_slot);
    DllExport int server_restore_slot(void* ctx, const char* req_body, int id_slot);
    DllExport int server_erase_slot(void* ctx, int id_slot);

    DllExport void server_flush_log(char* log_file);
}

#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>

#include <cpp-httplib/httplib.h>

#define OUTPUT_BUFFER_HISTORY_LIMIT 20

struct OutputBuffer
{
    char* buffer = nullptr;
    size_t buffer_size = 0;

    std::timed_mutex buffer_mutex;
    std::condition_variable_any buffer_cv;
    bool output_ready = false;

    OutputBuffer()
    {
        buffer = nullptr;
        buffer_size = 0;

        output_ready = false;
    }

    ~OutputBuffer()
    {
        free(buffer);
    }

    void reset()
    {
        std::unique_lock<std::timed_mutex> lck(buffer_mutex);

        if (buffer)
        {
            memset(buffer, 0, buffer_size);
        }
    }

    bool append(const char* data, size_t size)
    {
        bool ret = false;
        if (!buffer_mutex.try_lock_for(std::chrono::milliseconds(200)))
        {
            return ret;
        }

        char* end_of_buffer = allocate(size);
        if (nullptr != end_of_buffer)
        {
            memcpy(end_of_buffer, data, size);

            output_ready = true;
            buffer_cv.notify_one();

            ret = true;
        }

        buffer_mutex.unlock();

        return ret;
    }

    size_t copy_output(char* dst, size_t dst_size, unsigned long timeout = 100)
    {
        bool ret = false;
        std::unique_lock<std::timed_mutex> lck(buffer_mutex);

        while (!output_ready)
        {
            if (buffer_cv.wait_for(lck, std::chrono::milliseconds(timeout)) == std::cv_status::timeout)
            {
                return 0;
            }
        }

        size_t len = 0;
        bool copied = false;

        if (nullptr != buffer)
        {
            len = strlen(buffer) + 1;
            if (dst != nullptr && len <= dst_size)
            {
                memcpy(dst, buffer, len);
                memset(buffer, 0, buffer_size);
                copied = true;
                output_ready = false;
            }
        }

        if (!copied && len > 0)
        {
            output_ready = true;
        }

        return len;
    }

    char* allocate(size_t size)
    {
        if (size == 0)
            return buffer;

        if (nullptr == buffer)
        {
            buffer_size = 2 * size;
            buffer = (char*)malloc(buffer_size);
            memset(buffer, 0, buffer_size);
            return buffer;
        }
        else
        {
            size_t len = strlen(buffer);
            size_t new_size = len + size + 1;

            if (new_size > buffer_size)
            {
                new_size = 2 * (len + size);
                char* temp = (char*)malloc(new_size);
                memset(temp, 0, new_size);
                memcpy(temp, buffer, len);
                free(buffer);
                buffer = temp;
                buffer_size = new_size;
            }

            return buffer + len;
        }
    }

    bool write(const char* data, size_t size)
    {
        bool ret = append(data, size);

        return ret;
    }
};


struct AmdChatTask
{
    OutputBuffer output_buffer;
    int id_task;
    bool closed;

    httplib::Request req;
    httplib::Response res;
    httplib::DataSink sink;
    httplib::Server::Handler handler;

    void run()
    {
        sink.write = [&](const char * data, size_t size) {
            return output_buffer.write(data, size) && !closed;
        };
        sink.done = [&]() {
            closed = true;
        };
        sink.is_writable = [&]() {
            return !closed;
        };
        req.is_connection_closed = [&]() {
            return closed;
        };
        if (handler) {
            handler(req, res);
        }
        if (res.content_provider_) {
            std::thread t([this]() {
                res.content_provider_(0, 0, sink);
                sink.done();
            });
            t.detach();
        } else {
            sink.write(res.body.c_str(), res.body.size());
            sink.done();
        }
    }

    AmdChatTask()
    {
        id_task = 0;
        closed = false;
    }
    ~AmdChatTask()
    {
    }
};

class AmdChatTaskHandler
{
public:
    AmdChatTaskHandler()
    {
    }

    ~AmdChatTaskHandler()
    {
        for (AmdChatTask *task : tasks)
        {
            delete task;
        }
    }

    AmdChatTask* run_task(const char * req_body, httplib::Server::Handler handler) {
        AmdChatTask *task = new_task(req_body, handler);

        if (task) {
            std::unique_lock<std::mutex> lock(handler_mutex);
            task->run();
        }

        return task;
    }

    AmdChatTask* find_task_by_id(int id_task) {
        std::unique_lock<std::mutex> lock(tasks_mutex);
        for (AmdChatTask* task : tasks)
        {
            if (task->id_task == id_task)
            {
                return task;
            }
        }
        return nullptr;
    }
private:
    int task_counter = 0;
    std::vector<AmdChatTask*> tasks;
    std::mutex tasks_mutex;
    std::mutex handler_mutex;

    AmdChatTask * new_task(const char * req_body, httplib::Server::Handler handler) {
        std::unique_lock<std::mutex> lock(tasks_mutex);
        AmdChatTask *                task = new AmdChatTask();

        task->id_task  = task_counter++;
        task->req.body = req_body;
        task->handler  = handler;

        tasks.push_back(task);

        while (tasks.size() > OUTPUT_BUFFER_HISTORY_LIMIT) {
            AmdChatTask * task = tasks.front();
            if (task->closed) {
                tasks.erase(tasks.begin());
                delete task;
            } else {
                break;
            }
        }

        return task;
    }
};
