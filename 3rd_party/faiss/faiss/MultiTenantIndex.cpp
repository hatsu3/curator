// -*- c++ -*-

#include <faiss/MultiTenantIndex.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <cstring>

namespace faiss {

MultiTenantIndex::~MultiTenantIndex() {}

void MultiTenantIndex::train(idx_t /*n*/, const float* /*x*/, tid_t /*tid*/) {
    // does nothing by default
}

void MultiTenantIndex::range_search(
        idx_t,
        const float*,
        float,
        tid_t,
        RangeSearchResult*,
        const SearchParameters* params) const {
    FAISS_THROW_MSG("range search not implemented");
}

void MultiTenantIndex::assign(idx_t n, const float* x, tid_t tid, idx_t* labels, idx_t k) const {
    std::vector<float> distances(n * k);
    search(n, x, k, tid, distances.data(), labels);
}

void MultiTenantIndex::add_vector_with_ids(
        idx_t /*n*/,
        const float* /*x*/,
        const idx_t* /*xids*/,
        tid_t /*tid*/) {
    FAISS_THROW_MSG("add_vector_with_ids not implemented for this type of index");
}

bool MultiTenantIndex::remove_vector(idx_t xid, tid_t tid) {
    FAISS_THROW_MSG("remove_vector not implemented for this type of index");    
}

bool MultiTenantIndex::revoke_access(idx_t xid, tid_t tid) {
    FAISS_THROW_MSG("revoke_access not implemented for this type of index");
}

void MultiTenantIndex::reconstruct(idx_t, float*) const {
    FAISS_THROW_MSG("reconstruct not implemented for this type of index");
}

void MultiTenantIndex::reconstruct_batch(idx_t n, const idx_t* keys, float* recons) const {
    std::mutex exception_mutex;
    std::string exception_string;
#pragma omp parallel for if (n > 1000)
    for (idx_t i = 0; i < n; i++) {
        try {
            reconstruct(keys[i], &recons[i * d]);
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(exception_mutex);
            exception_string = e.what();
        }
    }
    if (!exception_string.empty()) {
        FAISS_THROW_MSG(exception_string.c_str());
    }
}

void MultiTenantIndex::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
#pragma omp parallel for if (ni > 1000)
    for (idx_t i = 0; i < ni; i++) {
        reconstruct(i0 + i, recons + i * d);
    }
}

void MultiTenantIndex::search_and_reconstruct(
        idx_t n,
        const float* x,
        idx_t k,
        tid_t tid,
        float* distances,
        idx_t* labels,
        float* recons,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);

    search(n, x, k, tid, distances, labels, params);
    for (idx_t i = 0; i < n; ++i) {
        for (idx_t j = 0; j < k; ++j) {
            idx_t ij = i * k + j;
            idx_t key = labels[ij];
            float* reconstructed = recons + ij * d;
            if (key < 0) {
                // Fill with NaNs
                memset(reconstructed, -1, sizeof(*reconstructed) * d);
            } else {
                reconstruct(key, reconstructed);
            }
        }
    }
}

void MultiTenantIndex::compute_residual(const float* x, float* residual, idx_t key) const {
    reconstruct(key, residual);
    for (size_t i = 0; i < d; i++) {
        residual[i] = x[i] - residual[i];
    }
}

void MultiTenantIndex::compute_residual_n(
        idx_t n,
        const float* xs,
        float* residuals,
        const idx_t* keys) const {
#pragma omp parallel for
    for (idx_t i = 0; i < n; ++i) {
        compute_residual(&xs[i * d], &residuals[i * d], keys[i]);
    }
}

size_t MultiTenantIndex::sa_code_size() const {
    FAISS_THROW_MSG("standalone codec not implemented for this type of index");
}

void MultiTenantIndex::sa_encode(idx_t, const float*, uint8_t*) const {
    FAISS_THROW_MSG("standalone codec not implemented for this type of index");
}

void MultiTenantIndex::sa_decode(idx_t, const uint8_t*, float*) const {
    FAISS_THROW_MSG("standalone codec not implemented for this type of index");
}

namespace {

// storage that explicitly reconstructs vectors before computing distances
struct GenericDistanceComputer : DistanceComputer {
    size_t d;
    const MultiTenantIndex& storage;
    std::vector<float> buf;
    const float* q;

    explicit GenericDistanceComputer(const MultiTenantIndex& storage) : storage(storage) {
        d = storage.d;
        buf.resize(d * 2);
    }

    float operator()(idx_t i) override {
        storage.reconstruct(i, buf.data());
        return fvec_L2sqr(q, buf.data(), d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        storage.reconstruct(i, buf.data());
        storage.reconstruct(j, buf.data() + d);
        return fvec_L2sqr(buf.data() + d, buf.data(), d);
    }

    void set_query(const float* x) override {
        q = x;
    }
};

} // namespace

DistanceComputer* MultiTenantIndex::get_distance_computer() const {
    if (metric_type == METRIC_L2) {
        return new GenericDistanceComputer(*this);
    } else {
        FAISS_THROW_MSG("get_distance_computer() not implemented");
    }
}

void MultiTenantIndex::merge_from(MultiTenantIndex& /* otherIndex */, idx_t /* add_id */) {
    FAISS_THROW_MSG("merge_from() not implemented");
}

void MultiTenantIndex::check_compatible_for_merge(const MultiTenantIndex& /* otherIndex */) const {
    FAISS_THROW_MSG("check_compatible_for_merge() not implemented");
}

} // namespace faiss
