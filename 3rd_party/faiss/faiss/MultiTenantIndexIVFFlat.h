// -*- c++ -*-

#ifndef FAISS_MULTI_TENANT_INDEX_IVF_FLAT_H
#define FAISS_MULTI_TENANT_INDEX_IVF_FLAT_H

#include <stdint.h>
#include <unordered_map>

#include <faiss/MultiTenantIndexIVF.h>

namespace faiss {

/** Inverted file with stored vectors. Here the inverted file
 * pre-selects the vectors to be searched, but they are not otherwise
 * encoded, the code array just contains the raw float entries.
 */
struct MultiTenantIndexIVFFlat : MultiTenantIndexIVF {
    MultiTenantIndexIVFFlat(
            Index* quantizer,
            size_t d,
            size_t nlist_,
            MetricType = METRIC_L2);

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx) override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    MultiTenantIndexIVFFlat();
};

} // namespace faiss

#endif
