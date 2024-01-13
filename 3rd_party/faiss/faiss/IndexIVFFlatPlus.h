// -*- c++ -*-

#ifndef FAISS_INDEX_IVF_FLAT_PLUS_H
#define FAISS_INDEX_IVF_FLAT_PLUS_H

#include <faiss/IndexIVFFlat.h>

namespace faiss {

struct IndexIVFFlatPlus : IndexIVFFlat {
    IndexIVFFlatPlus(
            Index* quantizer,
            size_t d,
            size_t nlist_,
            MetricType = METRIC_L2);

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
};

}  // namespace faiss

#endif
