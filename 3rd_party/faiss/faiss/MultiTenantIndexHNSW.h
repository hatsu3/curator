#ifndef MULTI_TENANT_INDEX_HNSW_H
#define MULTI_TENANT_INDEX_HNSW_H

#include <stdint.h>
#include <unordered_map>

#include <faiss/MetricType.h>
#include <faiss/MultiTenantIndex.h>
#include <faiss/hnswlib/hnswlib.h>
#include <thread>

namespace faiss {

template <class Function>
inline void ParallelFor(
        size_t start,
        size_t end,
        size_t numThreads,
        Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(
                                lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value
                         * that size_t can fit, because fetch_add returns the
                         * previous value before the increment (what will result
                         * in overflow and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

class PermissionCheck : public hnswlib::BaseFilterFunctor {
    tid_t tenant;
    const AccessMap& access_map;

public:
    PermissionCheck(tid_t tenant, const AccessMap& access_map) : tenant(tenant), access_map(access_map) {}
    bool operator()(hnswlib::labeltype label_id) {
        return access_map.at(label_id).find(tenant) != access_map.at(label_id).end();
    }
};

struct MultiTenantIndexHNSW : MultiTenantIndex {

    size_t d, M, ef_construction, ef, max_elements;
    
    AccessMap access_map;
    std::unordered_map<idx_t, tid_t> vector_owners;

    hnswlib::L2Space* space;
    hnswlib::HierarchicalNSW<float>* index;

    MultiTenantIndexHNSW(size_t d, size_t M, size_t ef_construction, size_t ef, size_t max_elements);

    ~MultiTenantIndexHNSW();

    void train(idx_t n, const float* x, tid_t tid) override;

    void add_vector_with_ids(
            idx_t n,
            const float* x,
            const idx_t* xids,
            tid_t tid) override;

    void grant_access(idx_t xid, tid_t tid) override;

    bool remove_vector(idx_t xid, tid_t tid) override;

    bool revoke_access(idx_t xid, tid_t tid) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            tid_t tid,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void add_vector(idx_t n, const float* x, tid_t tid) override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            tid_t tid,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    void assign(idx_t n, const float* x, tid_t tid, idx_t* labels, idx_t k = 1)
            const override;

    void reset() override;

    void reconstruct(idx_t key, float* recons) const override;

    void reconstruct_batch(idx_t n, const idx_t* keys, float* recons)
            const override;

    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    void search_and_reconstruct(
            idx_t n,
            const float* x,
            idx_t k,
            tid_t tid,
            float* distances,
            idx_t* labels,
            float* recons,
            const SearchParameters* params = nullptr) const override;

    void compute_residual(const float* x, float* residual, idx_t key)
            const override;

    void compute_residual_n(
            idx_t n,
            const float* xs,
            float* residuals,
            const idx_t* keys) const override;

    DistanceComputer* get_distance_computer() const override;

    size_t sa_code_size() const override;

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    void merge_from(MultiTenantIndex& otherIndex, idx_t add_id = 0) override;

    void check_compatible_for_merge(const MultiTenantIndex& otherIndex) const override;
};

} // namespace faiss

#endif