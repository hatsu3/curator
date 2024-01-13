// -*- c++ -*-

#ifndef FAISS_MULTI_TENANT_INDEX_IVF_FLAT_SEP_H
#define FAISS_MULTI_TENANT_INDEX_IVF_FLAT_SEP_H

#include <stdint.h>
#include <unordered_map>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/MultiTenantIndexIVFFlat.h>

namespace faiss {

struct MultiTenantIndexIVFFlatSep : MultiTenantIndexIVFFlat {
    size_t nlist;

    // index for each tenant
    std::unordered_map<tid_t, IndexIVFFlat*> indexes;
    // quantizer for each tenant
    std::unordered_map<tid_t, IndexFlatL2*> quantizers;
    // tenants that have access to each vector
    std::unordered_map<tid_t, std::unordered_set<idx_t>> xid_to_tids;
    // the owner of each vector
    std::unordered_map<idx_t, tid_t> xid_to_owner;

    MultiTenantIndexIVFFlatSep(Index* quantizer, size_t d, size_t nlist);

    ~MultiTenantIndexIVFFlatSep() override;

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
};

} // namespace faiss

#endif
