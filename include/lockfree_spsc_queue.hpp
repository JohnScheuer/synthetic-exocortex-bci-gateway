#ifndef LOCKFREE_SPSC_QUEUE_HPP
#define LOCKFREE_SPSC_QUEUE_HPP

#include <vector>
#include <atomic>
#include <cstddef>
#include <cassert>

template<typename T, size_t Capacity>
class LockFreeSPSCQueue {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

    LockFreeSPSCQueue() : head_(0), tail_(0) {}

    bool enqueue(const T& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        if (current_tail - head_.load(std::memory_order_acquire) == Capacity) {
            return false;
        }
        buffer_[current_tail & kMask] = item;
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    bool dequeue(T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        item = buffer_[current_head & kMask];
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    size_t size() const {
        return tail_.load(std::memory_order_relaxed) - head_.load(std::memory_order_relaxed);
    }

private:
    static constexpr size_t kMask = Capacity - 1;
    T buffer_[Capacity];
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
};

#endif // LOCKFREE_SPSC_QUEUE_HPP
