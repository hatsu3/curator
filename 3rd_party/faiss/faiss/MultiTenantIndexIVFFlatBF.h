// -*- c++ -*-

#ifndef FAISS_MULTI_TENANT_INDEX_IVF_FLAT_BF_H
#define FAISS_MULTI_TENANT_INDEX_IVF_FLAT_BF_H

#include <stdint.h>
#include <unordered_map>

#include <faiss/MultiTenantIndexIVFFlat.h>
#include <faiss/BloomFilter.h>

namespace faiss {

/** Inverted file with stored vectors. Here the inverted file
 * pre-selects the vectors to be searched, but they are not otherwise
 * encoded, the code array just contains the raw float entries.
 */
struct MultiTenantIndexIVFFlatBF : MultiTenantIndexIVFFlat {
    std::vector<bloom_filter> ivf_bfs;
    std::vector<std::unordered_map<tid_t, size_t>> tenant_nvecs;

    bloom_parameters bf_params;
    size_t bf_capacity;
    float bf_false_pos;
    float gamma;

    MultiTenantIndexIVFFlatBF(
            Index* quantizer,
            size_t d,
            size_t nlist_,
            MetricType = METRIC_L2,
            size_t bf_capacity = 1000,
            float bf_false_pos = 0.01,
            float gamma = 5.0);

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

    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            tid_t tid,
            const idx_t* keys,
            const float* coarse_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params,
            IndexIVFStats* ivf_stats) const override;

    MultiTenantIndexIVFFlatBF();
};

} // namespace faiss

#endif
