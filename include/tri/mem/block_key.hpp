/**
 * @file block_key.hpp
 * @brief Block identifier for memory management
 * @author Yongze
 * @date 2025-08-19
 */

#pragma once

#include <cstddef>
#include <functional>
#include <iostream>

namespace tri {
namespace mem {

/**
 * @brief Unique identifier for a block in BlockedTriMatrix
 */
struct BlockKey {
    std::size_t block_row;  ///< Block row index
    std::size_t block_col;  ///< Block column index

    /**
     * @brief Default constructor
     */
    BlockKey() : block_row(0), block_col(0) {}

    /**
     * @brief Constructor with indices
     * @param row Block row index
     * @param col Block column index
     */
    BlockKey(std::size_t row, std::size_t col) : block_row(row), block_col(col) {}

    /**
     * @brief Equality operator
     */
    bool operator==(const BlockKey& other) const {
        return block_row == other.block_row && block_col == other.block_col;
    }

    /**
     * @brief Inequality operator
     */
    bool operator!=(const BlockKey& other) const { return !(*this == other); }

    /**
     * @brief Less-than operator for ordering
     */
    bool operator<(const BlockKey& other) const {
        if (block_row != other.block_row) {
            return block_row < other.block_row;
        }
        return block_col < other.block_col;
    }
};

/**
 * @brief Output stream operator for BlockKey
 */
inline std::ostream& operator<<(std::ostream& os, const BlockKey& key) {
    return os << "(" << key.block_row << "," << key.block_col << ")";
}

}  // namespace mem
}  // namespace tri

// Hash specialization for BlockKey
namespace std {
template <>
struct hash<tri::mem::BlockKey> {
    std::size_t operator()(const tri::mem::BlockKey& key) const {
        // Combine hashes of row and column
        std::size_t h1 = std::hash<std::size_t>{}(key.block_row);
        std::size_t h2 = std::hash<std::size_t>{}(key.block_col);
        return h1 ^ (h2 << 1);
    }
};
}  // namespace std