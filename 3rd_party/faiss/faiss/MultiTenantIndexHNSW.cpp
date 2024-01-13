#include <faiss/MultiTenantIndexHNSW.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {

MultiTenantIndexHNSW::MultiTenantIndexHNSW(
        size_t d,
        size_t M,
        size_t ef_construction,
        size_t ef,
        size_t max_elements)
        : d(d),
          M(M),
          ef_construction(ef_construction),
          ef(ef),
          max_elements(max_elements) {
    space = new hnswlib::L2Space(d);
    index = new hnswlib::HierarchicalNSW<float>(
        space, max_elements, M, ef_construction, 100, 
        /*allow_replace_deleted=*/true
    );
    index->setEf(ef);
}

MultiTenantIndexHNSW::~MultiTenantIndexHNSW() {
    delete space;
    delete index;
}

void MultiTenantIndexHNSW::train(idx_t n, const float* x, tid_t tid) {
    FAISS_THROW_MSG("HNSW does not support training");
}

void MultiTenantIndexHNSW::add_vector_with_ids(
        idx_t n,
        const float* x,
        const idx_t* xids,
        tid_t tid) {
    
    for (size_t i = 0; i < n; i++) {
        idx_t label = xids[i];
        
        // add the vector to the index
        index->addPoint((void*)(x + i * d), label, /*replace_deleted=*/true);
        
        // update the access map
        auto it = access_map.find(label);
        FAISS_THROW_IF_NOT_MSG(it == access_map.end(), "Vector already exists");
        access_map[label].insert(tid);
        vector_owners[label] = tid;
    }
}

void MultiTenantIndexHNSW::grant_access(idx_t xid, tid_t tid) {
    // update the access map
    auto it = access_map.find(xid);
    FAISS_THROW_IF_NOT_MSG(it != access_map.end(), "Vector not found");
    it->second.insert(tid);
}

bool MultiTenantIndexHNSW::remove_vector(idx_t xid, tid_t tid) {
    // check if the vector is owned by the tenant
    auto it = vector_owners.find(xid);
    FAISS_THROW_IF_NOT_MSG(it != vector_owners.end(), "Vector not found");
    if (it->second != tid) {
        return false;
    }

    // update the access map
    vector_owners.erase(it);
    access_map.erase(xid);

    // remove the vector from the index
    index->markDelete(xid);

    return true;
}

bool MultiTenantIndexHNSW::revoke_access(idx_t xid, tid_t tid) {
    auto it = access_map.find(xid);
    FAISS_THROW_IF_NOT_MSG(it != access_map.end(), "Vector not found");
    return it->second.erase(tid);
}

void MultiTenantIndexHNSW::search(
        idx_t n,
        const float* x,
        idx_t k,
        tid_t tid,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    
    PermissionCheck checker(tid, access_map);
    size_t num_threads = getenv("OMP_NUM_THREADS") ? atoi(getenv("OMP_NUM_THREADS")) : 1;

    ParallelFor(0, n, num_threads, [&](size_t i, size_t threadId) {
        std::vector<std::pair<float, hnswlib::labeltype>> result = index->searchKnnCloserFirst(
                (void*)(x + i * d), k, &checker);
        
        for (size_t j = 0; j < k; j++) {
            distances[i * k + j] = result[j].first;
            labels[i * k + j] = result[j].second;
        }
    });
}

void MultiTenantIndexHNSW::add_vector(idx_t n, const float* x, tid_t tid) {
    FAISS_THROW_MSG("Not implemented");
}

void MultiTenantIndexHNSW::range_search(
        idx_t n,
        const float* x,
        float radius,
        tid_t tid,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    FAISS_THROW_MSG("Not implemented");
}

void MultiTenantIndexHNSW::assign(idx_t n, const float* x, tid_t tid, idx_t* labels, idx_t k) const {
    FAISS_THROW_MSG("Not implemented");
}

void MultiTenantIndexHNSW::reset() {
    FAISS_THROW_MSG("Not implemented");
}

void MultiTenantIndexHNSW::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_MSG("Not implemented");
}

void MultiTenantIndexHNSW::reconstruct_batch(idx_t n, const idx_t* keys, float* recons) const {
    FAISS_THROW_MSG("Not implemented");
}

void MultiTenantIndexHNSW::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    FAISS_THROW_MSG("Not implemented");
}

void MultiTenantIndexHNSW::search_and_reconstruct(
        idx_t n,
        const float* x,
        idx_t k,
        tid_t tid,
        float* distances,
        idx_t* labels,
        float* recons,
        const SearchParameters* params) const {
    FAISS_THROW_MSG("Not implemented");
}

void MultiTenantIndexHNSW::compute_residual(const float* x, float* residual, idx_t key) const {
    FAISS_THROW_MSG("Not implemented");
}

void MultiTenantIndexHNSW::compute_residual_n(
        idx_t n,
        const float* xs,
        float* residuals,
        const idx_t* keys) const {
    FAISS_THROW_MSG("Not implemented");
}

DistanceComputer* MultiTenantIndexHNSW::get_distance_computer() const {
    FAISS_THROW_MSG("Not implemented");
}

size_t MultiTenantIndexHNSW::sa_code_size() const {
    FAISS_THROW_MSG("Not implemented");
}

void MultiTenantIndexHNSW::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_MSG("Not implemented");
}

void MultiTenantIndexHNSW::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    FAISS_THROW_MSG("Not implemented");
}

void MultiTenantIndexHNSW::merge_from(MultiTenantIndex& otherIndex, idx_t add_id) {
    FAISS_THROW_MSG("Not implemented");
}

void MultiTenantIndexHNSW::check_compatible_for_merge(const MultiTenantIndex& otherIndex) const {
    FAISS_THROW_MSG("Not implemented");
}

} // namespace faiss