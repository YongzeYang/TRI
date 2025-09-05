#pragma once

/**
 * @file blocked_tri.hpp
 * @brief Blocked lower triangular matrix with memory-efficient storage
 * @author Yongze
 * @date 2025-08-13
 */

#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "tri/common/config.hpp"
#include "tri/common/macros.hpp"
#include "tri/core/matrix_base.hpp"

// Forward declaration for memory management
namespace tri {
namespace mem {
class BlockManager;
struct BlockKey;
template <typename T>
class BlockStorage;
}  // namespace mem
}  // namespace tri

namespace tri {
namespace core {

/**
 * @brief Blocked lower triangular matrix with chunked storage
 * @tparam T Element type (default: float)
 *
 * This class provides a blocked storage implementation for lower triangular matrices,
 * optimized for cache efficiency and memory management. Only the lower triangular
 * blocks are stored, and each block can be independently loaded/unloaded.
 */
template <typename T = float>
class TRI_API BlockedTriMatrix : public MatrixBase<T> {
   public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;

    /**
     * @brief Storage backend type
     */
    enum class StorageType {
        Memory,  ///< In-memory storage only
        Disk,    ///< Disk-based storage
        Hybrid   ///< Memory cache + disk backing
    };

    /**
     * @brief Set storage backend for evicted blocks
     * @param type Storage type to use
     * @param disk_path Path for disk storage (if applicable)
     * @return true if successful
     */
    bool set_storage_backend(StorageType type, const std::string& disk_path = ".");

    /**
     * @brief Get current storage backend
     * @return Pointer to storage backend or nullptr
     */
    std::shared_ptr<mem::BlockStorage<T>> get_storage_backend() const { return storage_backend_; }

    /**
     * @brief Single block structure
     */
    struct Block {
        std::unique_ptr<T[]> data;
        size_type access_count = 0;  // For future eviction policy
        bool is_loaded = false;      // For future lazy loading

        Block() = default;
        explicit Block(size_type size);
        Block(Block&&) noexcept = default;
        Block& operator=(Block&&) noexcept = default;

        // Delete copy operations
        Block(const Block&) = delete;
        Block& operator=(const Block&) = delete;
    };

    /**
     * @brief Block index structure
     */
    struct BlockIndex {
        size_type block_row;
        size_type block_col;
        size_type row_offset;  // Offset within block
        size_type col_offset;  // Offset within block

        bool is_valid() const noexcept { return block_row != static_cast<size_type>(-1); }

        static BlockIndex invalid() noexcept {
            return {static_cast<size_type>(-1), static_cast<size_type>(-1), 0, 0};
        }
    };

    // Constructors
    BlockedTriMatrix() = default;

    // Single-argument constructor for integral types
    template <typename IntType, typename = std::enable_if_t<std::is_integral_v<IntType> &&
                                                            !std::is_same_v<IntType, bool>>>
    explicit BlockedTriMatrix(IntType n, size_type block_size = tri::config::DEFAULT_BLOCK_SIZE);

    // Two-argument constructor for integral type + value
    template <typename IntType, typename = std::enable_if_t<std::is_integral_v<IntType> &&
                                                            !std::is_same_v<IntType, bool>>>
    BlockedTriMatrix(IntType n, size_type block_size, const T& value);

    // Construct from dense matrix (must have rows() and cols() methods)
    template <typename MatrixType, typename = std::enable_if_t<!std::is_integral_v<MatrixType>>>
    explicit BlockedTriMatrix(const MatrixType& dense,
                              size_type block_size = tri::config::DEFAULT_BLOCK_SIZE);

    // Rule of Five
    ~BlockedTriMatrix() = default;
    BlockedTriMatrix(const BlockedTriMatrix& other);
    BlockedTriMatrix(BlockedTriMatrix&&) noexcept = default;
    BlockedTriMatrix& operator=(const BlockedTriMatrix& other);
    BlockedTriMatrix& operator=(BlockedTriMatrix&&) noexcept = default;

    // Element access
    [[nodiscard]] reference operator()(size_type i, size_type j) override;
    [[nodiscard]] const_reference operator()(size_type i, size_type j) const override;
    [[nodiscard]] T at(size_type i, size_type j) const;
    void set(size_type i, size_type j, const T& value);

    // Block-level access
    [[nodiscard]] Block* get_block(size_type block_row, size_type block_col);
    [[nodiscard]] const Block* get_block(size_type block_row, size_type block_col) const;
    [[nodiscard]] pointer get_block_data(size_type block_row, size_type block_col);
    [[nodiscard]] const_pointer get_block_data(size_type block_row, size_type block_col) const;

    // Matrix properties
    [[nodiscard]] size_type rows() const noexcept override { return n_; }
    [[nodiscard]] size_type cols() const noexcept override { return n_; }
    [[nodiscard]] size_type size() const noexcept override { return n_ * n_; }
    [[nodiscard]] bool empty() const noexcept override { return n_ == 0; }
    [[nodiscard]] bool is_square() const noexcept override { return true; }

    // Block properties
    [[nodiscard]] size_type block_size() const noexcept { return block_size_; }
    [[nodiscard]] size_type num_block_rows() const noexcept { return num_blocks_; }
    [[nodiscard]] size_type num_block_cols() const noexcept { return num_blocks_; }
    [[nodiscard]] size_type total_blocks() const noexcept;
    [[nodiscard]] size_type allocated_blocks() const noexcept;

    // Data access (returns nullptr for blocked storage)
    [[nodiscard]] pointer data() noexcept override { return nullptr; }
    [[nodiscard]] const_pointer data() const noexcept override { return nullptr; }

    // Export to dense format
    void export_to_dense(pointer dest, size_type ld) const;
    [[nodiscard]] std::vector<T> to_dense_vector() const;

    // Block index calculation
    [[nodiscard]] BlockIndex compute_block_index(size_type i, size_type j) const noexcept;
    [[nodiscard]] size_type get_linear_block_index(size_type block_row,
                                                   size_type block_col) const noexcept;
    [[nodiscard]] bool is_block_in_lower_triangle(size_type block_row,
                                                  size_type block_col) const noexcept;

    // Memory management
    void load_block(size_type block_row, size_type block_col);
    void unload_block(size_type block_row, size_type block_col);
    void pin_block(size_type block_row, size_type block_col);
    void unpin_block(size_type block_row, size_type block_col);

    // Memory manager integration
    /**
     * @brief Set memory manager for block caching
     * @param manager Block manager instance
     */
    void set_block_manager(std::shared_ptr<mem::BlockManager> manager);

    /**
     * @brief Get current block manager
     * @return Block manager or nullptr if not set
     */
    std::shared_ptr<mem::BlockManager> get_block_manager() const { return block_manager_; }

    /**
     * @brief Create block manager with specified policy
     * @param max_blocks Maximum blocks in memory
     * @param policy_name Policy name ("LRU" or "AccessCount")
     * @return true if successful
     */
    bool create_block_manager(size_type max_blocks, const std::string& policy_name = "LRU");

    // Access tracking
    [[nodiscard]] size_type get_block_access_count(size_type block_row, size_type block_col) const;
    void reset_access_counts() noexcept;

    // Modifiers
    void clear() noexcept;
    void resize(size_type new_n, size_type new_block_size = 0);
    void fill(const T& value);
    void fill_block(size_type block_row, size_type block_col, const T& value);
    void set_diagonal(const T& value);
    void swap(BlockedTriMatrix& other) noexcept;

    // Static factory methods
    [[nodiscard]] static BlockedTriMatrix identity(
        size_type n, size_type block_size = tri::config::DEFAULT_BLOCK_SIZE);
    [[nodiscard]] static BlockedTriMatrix zeros(
        size_type n, size_type block_size = tri::config::DEFAULT_BLOCK_SIZE);

   private:
    // Replace the simple persistent_storage_ with:
    std::shared_ptr<mem::BlockStorage<T>> storage_backend_;

    // Add unique matrix ID for disk storage
    std::string matrix_id_;

    // Helper to generate unique matrix ID
    std::string generate_matrix_id() const;

    // Helper methods
    void allocate_blocks();
    void allocate_block(size_type block_row, size_type block_col);
    void deallocate_block(size_type block_row, size_type block_col);
    [[nodiscard]] bool should_store_block(size_type block_row, size_type block_col) const noexcept;
    [[nodiscard]] size_type get_block_actual_rows(size_type block_row) const noexcept;
    [[nodiscard]] size_type get_block_actual_cols(size_type block_col) const noexcept;

    // Private initialization helper
    void init(size_type n, size_type block_size);

    // Callbacks for block manager
    bool load_block_callback(const mem::BlockKey& key);
    bool unload_block_callback(const mem::BlockKey& key);

   private:
    size_type n_ = 0;            // Matrix dimension
    size_type block_size_ = 0;   // Size of each block
    size_type num_blocks_ = 0;   // Number of blocks in each dimension
    std::vector<Block> blocks_;  // Block storage (only lower triangular blocks)

    // Memory management
    std::shared_ptr<mem::BlockManager> block_manager_;

    // Persistent storage for blocks when evicted
    std::unordered_map<size_type, std::unique_ptr<T[]>> persistent_storage_;

    // For future extensions
    std::unordered_map<size_type, size_type> block_remap_;  // Optional remapping for sparse blocks
    size_type pinned_blocks_ = 0;                           // Number of pinned blocks
    size_type max_memory_blocks_ = static_cast<size_type>(-1);  // Maximum blocks in memory

    // Static zero value for upper triangular access
    static const T zero_value_;
};

// Template constructor implementations
template <typename T>
template <typename IntType, typename>
BlockedTriMatrix<T>::BlockedTriMatrix(IntType n, size_type block_size) {
    if (n < 0) {
        throw std::invalid_argument("Matrix dimension must be non-negative");
    }
    init(static_cast<size_type>(n), block_size);
}

template <typename T>
template <typename IntType, typename>
BlockedTriMatrix<T>::BlockedTriMatrix(IntType n, size_type block_size, const T& value) {
    if (n < 0) {
        throw std::invalid_argument("Matrix dimension must be non-negative");
    }
    init(static_cast<size_type>(n), block_size);
    fill(value);
}

// Static member definition
template <typename T>
const T BlockedTriMatrix<T>::zero_value_ = T{0};

// Non-member swap
template <typename T>
void swap(BlockedTriMatrix<T>& lhs, BlockedTriMatrix<T>& rhs) noexcept;

// Extern template instantiations for common types
extern template class BlockedTriMatrix<float>;
extern template class BlockedTriMatrix<double>;

}  // namespace core
}  // namespace tri