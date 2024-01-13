// -*- c++ -*-

#include <faiss/MultiTenantIndexIVFFlatSep.h>

#include <cinttypes>
#include <cstdio>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>

namespace faiss {

MultiTenantIndexIVFFlatSep::MultiTenantIndexIVFFlatSep(Index* quantizer, size_t d, size_t nlist)
    : MultiTenantIndexIVFFlat(quantizer, d, nlist, METRIC_L2), nlist(nlist) {}

MultiTenantIndexIVFFlatSep::~MultiTenantIndexIVFFlatSep() {
    for (auto& kv : quantizers) {
        delete kv.second;
    }
    for (auto& kv : indexes) {
        delete kv.second;
    }
}

void MultiTenantIndexIVFFlatSep::train(idx_t n, const float* x, tid_t tid) {
    if (indexes.find(tid) == indexes.end()) {
        quantizers.emplace(tid, new IndexFlatL2(d));
        
        size_t new_nlist = this->nlist;
        if (n < new_nlist) {
            printf("WARNING: The number of training vectors (%" PRId64 ") is less than the number of clusters (%" PRId64 "), "
                   "setting the number of clusters to %" PRId64 "\n",
                   n, nlist, n);
            new_nlist = n;
        }
        auto index = new IndexIVFFlat(quantizers[tid], d, new_nlist, METRIC_L2);
        index->set_direct_map_type(DirectMap::Hashtable);
        indexes.emplace(tid, index);
    }
    indexes[tid]->train(n, x);
}

void MultiTenantIndexIVFFlatSep::add_vector_with_ids(
        idx_t n,
        const float* x,
        const idx_t* xids,
        tid_t tid) {
        
    for (idx_t i = 0; i < n; i++) {
        xid_to_tids.emplace(xids[i], std::unordered_set<tid_t>{tid});
        xid_to_owner.emplace(xids[i], tid);
    }

    indexes.at(tid)->add_with_ids(n, x, xids);
}

void MultiTenantIndexIVFFlatSep::grant_access(idx_t xid, tid_t tid) {
    // update the list of tenants that have access to the vector
    xid_to_tids.at(xid).insert(tid);

    // insert the vector into the tenant's index
    auto index = indexes.at(xid_to_owner.at(xid));
    idx_t lo = index->direct_map.get(xid);
    const uint8_t* x = index->invlists->get_single_code(lo_listno(lo), lo_offset(lo));
    indexes.at(tid)->add_with_ids(1, reinterpret_cast<const float*>(x), &xid);
}

bool MultiTenantIndexIVFFlatSep::remove_vector(idx_t xid, tid_t tid) {
    // check if the tenant is the owner of the vector
    if (xid_to_owner.at(xid) != tid) {
        return false;
    }
    xid_to_owner.erase(xid);

    // remove the vector from indexes of all tenants that have access to it
    for (tid_t tenant : xid_to_tids.at(xid)) {
        IDSelectorArray selector(1, &xid);
        indexes.at(tenant)->remove_ids(selector);
    }
    
    xid_to_tids.erase(xid);

    return true;
}

bool MultiTenantIndexIVFFlatSep::revoke_access(idx_t xid, tid_t tid) {
    // check if the tenant has access to the vector
    auto& tids = xid_to_tids.at(xid);
    if (tids.find(tid) == tids.end()) {
        return false;
    }
    tids.erase(tid);
    
    // remove the vector from the tenant's index
    IDSelectorArray selector(1, &xid);
    indexes.at(tid)->remove_ids(selector);
    
    return true;
}

void MultiTenantIndexIVFFlatSep::search(
        idx_t n,
        const float* x,
        idx_t k,
        tid_t tid,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    
    indexes.at(tid)->search(n, x, k, distances, labels, params);
}

}  // namespace faiss