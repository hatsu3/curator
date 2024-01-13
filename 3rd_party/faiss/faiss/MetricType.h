/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_METRIC_TYPE_H
#define FAISS_METRIC_TYPE_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <faiss/impl/platform_macros.h>

namespace faiss {

/// The metric space for vector comparison for Faiss indices and algorithms.
///
/// Most algorithms support both inner product and L2, with the flat
/// (brute-force) indices supporting additional metric types for vector
/// comparison.
enum MetricType {
    METRIC_INNER_PRODUCT = 0, ///< maximum inner product search
    METRIC_L2 = 1,            ///< squared L2 search
    METRIC_L1,                ///< L1 (aka cityblock)
    METRIC_Linf,              ///< infinity distance
    METRIC_Lp,                ///< L_p distance, p is given by a faiss::Index
                              /// metric_arg

    /// some additional metrics defined in scipy.spatial.distance
    METRIC_Canberra = 20,
    METRIC_BrayCurtis,
    METRIC_JensenShannon,
    METRIC_Jaccard, ///< defined as: sum_i(min(a_i, b_i)) / sum_i(max(a_i, b_i))
                    ///< where a_i, b_i > 0
};

/// all vector indices are this type
using idx_t = int64_t;

/// this function is used to distinguish between min and max indexes since
/// we need to support similarity and dis-similarity metrics in a flexible way
constexpr bool is_similarity_metric(MetricType metric_type) {
    return ((metric_type == METRIC_INNER_PRODUCT) ||
            (metric_type == METRIC_Jaccard));
}

/// multi-tenancy-related types
// tenant id, -1 means any tenant
using tid_t = int64_t;

struct AccessMap {
    /* For each bucket, we maintain a mapping between tenant ID and a list 
     * of vector IDs (corresponding to the vectors accessible to the tenant). 
     * We also maintain a mapping between vector ID and the ID of the bucket 
     * that contains the video.
     */

    // Maps a vector to the bucket that contains it
    std::unordered_map<idx_t, idx_t> vid_to_bid;

    // For each bucket, maps a tenant to the list of vectors that are accessible to it
    std::vector<std::unordered_map<tid_t, std::unordered_set<idx_t>>> tid_to_vids;
};

} // namespace faiss

#endif