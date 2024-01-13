/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>

namespace faiss {

/***********************************************************************
 * IDSelectorRange
 ***********************************************************************/

IDSelectorRange::IDSelectorRange(idx_t imin, idx_t imax, bool assume_sorted)
        : imin(imin), imax(imax), assume_sorted(assume_sorted) {}

bool IDSelectorRange::is_member(idx_t id) const {
    return id >= imin && id < imax;
}

void IDSelectorRange::find_sorted_ids_bounds(
        size_t list_size,
        const idx_t* ids,
        size_t* jmin_out,
        size_t* jmax_out) const {
    FAISS_ASSERT(assume_sorted);
    if (list_size == 0 || imax <= ids[0] || imin > ids[list_size - 1]) {
        *jmin_out = *jmax_out = 0;
        return;
    }
    // bissection to find imin
    if (ids[0] >= imin) {
        *jmin_out = 0;
    } else {
        size_t j0 = 0, j1 = list_size;
        while (j1 > j0 + 1) {
            size_t jmed = (j0 + j1) / 2;
            if (ids[jmed] >= imin) {
                j1 = jmed;
            } else {
                j0 = jmed;
            }
        }
        *jmin_out = j1;
    }
    // bissection to find imax
    if (*jmin_out == list_size || ids[*jmin_out] >= imax) {
        *jmax_out = *jmin_out;
    } else {
        size_t j0 = *jmin_out, j1 = list_size;
        while (j1 > j0 + 1) {
            size_t jmed = (j0 + j1) / 2;
            if (ids[jmed] >= imax) {
                j1 = jmed;
            } else {
                j0 = jmed;
            }
        }
        *jmax_out = j1;
    }
}

/***********************************************************************
 * IDSelectorArray
 ***********************************************************************/

IDSelectorArray::IDSelectorArray(size_t n, const idx_t* ids) : n(n), ids(ids) {}

bool IDSelectorArray::is_member(idx_t id) const {
    for (idx_t i = 0; i < n; i++) {
        if (ids[i] == id)
            return true;
    }
    return false;
}

/***********************************************************************
 * IDSelectorBatch
 ***********************************************************************/

IDSelectorBatch::IDSelectorBatch(size_t n, const idx_t* indices) {
    nbits = 0;
    while (n > ((idx_t)1 << nbits)) {
        nbits++;
    }
    nbits += 5;
    // for n = 1M, nbits = 25 is optimal, see P56659518

    mask = ((idx_t)1 << nbits) - 1;
    bloom.resize((idx_t)1 << (nbits - 3), 0);
    for (idx_t i = 0; i < n; i++) {
        idx_t id = indices[i];
        set.insert(id);
        id &= mask;
        bloom[id >> 3] |= 1 << (id & 7);
    }
}

bool IDSelectorBatch::is_member(idx_t i) const {
    long im = i & mask;
    if (!(bloom[im >> 3] & (1 << (im & 7)))) {
        return 0;
    }
    return set.count(i);
}

/***********************************************************************
 * IDSelectorBitmap
 ***********************************************************************/

IDSelectorBitmap::IDSelectorBitmap(size_t n, const uint8_t* bitmap)
        : n(n), bitmap(bitmap) {}

bool IDSelectorBitmap::is_member(idx_t ii) const {
    uint64_t i = ii;
    if ((i >> 3) >= n) {
        return false;
    }
    return (bitmap[i >> 3] >> (i & 7)) & 1;
}

/*****************************************
 * MultiTenantIDSelector
 ******************************************/

MultiTenantIDSelector::MultiTenantIDSelector(tid_t tid, const AccessMap* access_map, const IDSelector* base_sel)
    : tid(tid), access_map(access_map), base_sel(base_sel) {}

bool MultiTenantIDSelector::is_member(idx_t id) const {
    // if a base selector is provided, perform prefiltering
    if (base_sel && !base_sel->is_member(id)) {
        return false;
    }

    // if the vector is not found, return false
    // consider raising an exception instead
    auto it = access_map->vid_to_bid.find(id);
    if (it == access_map->vid_to_bid.end()) {
        return false;
    }

    // check if the tenant is allowed to access it
    if (tid == ALL_TENANTS) {
        return true;
    }

    // search the sorted vector of tenant ids using lower_bound
    // only perform binary search if the vector is large enough
    idx_t bucket_id = it->second;
    auto& ttv = access_map->tid_to_vids[bucket_id];
    
    // if the bucket does not contain any vector accessible to the tenant, return false
    // this should never happen because we will perform Bloom filter checks before calling this function
    if (ttv.find(tid) == ttv.end()) {
        return false;
    }

    // check whether the vector with the given id is accessible to the tenant
    return ttv.at(tid).find(id) != ttv.at(tid).end();
}

} // namespace faiss
