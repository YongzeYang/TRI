/**
 * @file block_storage.hpp
 * @brief Block storage backend interface and implementations
 * @author Yongze
 * @date 2025-08-20
 */

#pragma once

#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "block_key.hpp"

namespace tri {
namespace mem {

/**
 * @brief Abstract interface for block storage backends
 * @tparam T Element type
 */
template <typename T>
class BlockStorage {
   public:
    virtual ~BlockStorage() = default;

    /**
     * @brief Store a block of data
     * @param key Block identifier
     * @param data Pointer to block data
     * @param size Number of elements in block
     * @return true if successful
     */
    virtual bool store(const BlockKey& key, const T* data, std::size_t size) = 0;

    /**
     * @brief Load a block of data
     * @param key Block identifier
     * @param data Buffer to load data into
     * @param size Number of elements to load
     * @return true if successful
     */
    virtual bool load(const BlockKey& key, T* data, std::size_t size) = 0;

    /**
     * @brief Check if a block exists in storage
     * @param key Block identifier
     * @return true if block exists
     */
    virtual bool exists(const BlockKey& key) const = 0;

    /**
     * @brief Remove a block from storage
     * @param key Block identifier
     * @return true if successful
     */
    virtual bool remove(const BlockKey& key) = 0;

    /**
     * @brief Clear all stored blocks
     */
    virtual void clear() = 0;

    /**
     * @brief Get storage type name
     * @return Storage type as string
     */
    virtual std::string type() const = 0;

    /**
     * @brief Get current storage size in bytes
     * @return Size in bytes
     */
    virtual std::size_t size_bytes() const = 0;
};

/**
 * @brief In-memory storage backend
 * @tparam T Element type
 */
template <typename T>
class MemoryStorage : public BlockStorage<T> {
   private:
    std::unordered_map<BlockKey, std::unique_ptr<T[]>> storage_;
    std::unordered_map<BlockKey, std::size_t> sizes_;

   public:
    MemoryStorage() = default;
    ~MemoryStorage() override = default;

    bool store(const BlockKey& key, const T* data, std::size_t size) override {
        if (!data || size == 0) return false;

        auto buffer = std::make_unique<T[]>(size);
        std::copy_n(data, size, buffer.get());
        storage_[key] = std::move(buffer);
        sizes_[key] = size;
        return true;
    }

    bool load(const BlockKey& key, T* data, std::size_t size) override {
        if (!data || size == 0) return false;

        auto it = storage_.find(key);
        if (it == storage_.end()) return false;

        auto size_it = sizes_.find(key);
        if (size_it == sizes_.end() || size_it->second != size) return false;

        std::copy_n(it->second.get(), size, data);
        return true;
    }

    bool exists(const BlockKey& key) const override { return storage_.find(key) != storage_.end(); }

    bool remove(const BlockKey& key) override {
        sizes_.erase(key);
        return storage_.erase(key) > 0;
    }

    void clear() override {
        storage_.clear();
        sizes_.clear();
    }

    std::string type() const override { return "MemoryStorage"; }

    std::size_t size_bytes() const override {
        std::size_t total = 0;
        for (const auto& [key, size] : sizes_) {
            total += size * sizeof(T);
        }
        return total;
    }
};

/**
 * @brief Disk-based storage backend
 * @tparam T Element type
 */
template <typename T>
class DiskStorage : public BlockStorage<T> {
   private:
    std::filesystem::path base_path_;
    std::string prefix_;
    bool use_compression_;

    std::string get_filename(const BlockKey& key) const {
        std::ostringstream oss;
        oss << prefix_ << "_" << key.block_row << "_" << key.block_col << ".blk";
        return (base_path_ / oss.str()).string();
    }

   public:
    /**
     * @brief Constructor
     * @param base_path Directory for storing blocks
     * @param prefix Filename prefix
     * @param use_compression Enable compression (future feature)
     */
    DiskStorage(const std::string& base_path = ".", const std::string& prefix = "block",
                bool use_compression = false)
        : base_path_(base_path), prefix_(prefix), use_compression_(use_compression) {
        // Create directory if it doesn't exist
        if (!std::filesystem::exists(base_path_)) {
            std::filesystem::create_directories(base_path_);
        }
    }

    ~DiskStorage() override = default;

    bool store(const BlockKey& key, const T* data, std::size_t size) override {
        if (!data || size == 0) return false;

        std::string filename = get_filename(key);
        std::ofstream file(filename, std::ios::binary);
        if (!file) return false;

        // Write size first
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));

        // Write data
        file.write(reinterpret_cast<const char*>(data), size * sizeof(T));

        return file.good();
    }

    bool load(const BlockKey& key, T* data, std::size_t size) override {
        if (!data || size == 0) return false;

        std::string filename = get_filename(key);
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;

        // Read and verify size
        std::size_t stored_size;
        file.read(reinterpret_cast<char*>(&stored_size), sizeof(stored_size));
        if (stored_size != size) return false;

        // Read data
        file.read(reinterpret_cast<char*>(data), size * sizeof(T));

        return file.good();
    }

    bool exists(const BlockKey& key) const override {
        return std::filesystem::exists(get_filename(key));
    }

    bool remove(const BlockKey& key) override {
        std::string filename = get_filename(key);
        if (std::filesystem::exists(filename)) {
            return std::filesystem::remove(filename);
        }
        return false;
    }

    void clear() override {
        // Remove all block files with our prefix
        for (const auto& entry : std::filesystem::directory_iterator(base_path_)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                std::string suffix = ".blk";
                if (filename.find(prefix_ + "_") == 0 && filename.length() >= suffix.length() &&
                    filename.compare(filename.length() - suffix.length(), suffix.length(),
                                     suffix) == 0) {
                    std::filesystem::remove(entry.path());
                }
            }
        }
    }

    std::string type() const override {
        std::string result = "DiskStorage";
        if (use_compression_) {
            result += " (compressed)";
        }
        return result;
    }

    std::size_t size_bytes() const override {
        std::size_t total = 0;
        for (const auto& entry : std::filesystem::directory_iterator(base_path_)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                std::string suffix = ".blk";
                if (filename.find(prefix_ + "_") == 0 && filename.length() >= suffix.length() &&
                    filename.compare(filename.length() - suffix.length(), suffix.length(),
                                     suffix) == 0) {
                    total += entry.file_size();
                }
            }
        }
        return total;
    }

    /**
     * @brief Set base path for storage
     * @param path New base path
     */
    void set_base_path(const std::string& path) {
        base_path_ = path;
        if (!std::filesystem::exists(base_path_)) {
            std::filesystem::create_directories(base_path_);
        }
    }
};

/**
 * @brief Hybrid storage that uses memory as cache and disk as backing store
 * @tparam T Element type
 */
template <typename T>
class HybridStorage : public BlockStorage<T> {
   private:
    std::unique_ptr<MemoryStorage<T>> memory_;
    std::unique_ptr<DiskStorage<T>> disk_;
    std::size_t max_memory_blocks_;

   public:
    HybridStorage(std::size_t max_memory_blocks, const std::string& disk_path = ".",
                  const std::string& prefix = "block")
        : memory_(std::make_unique<MemoryStorage<T>>()),
          disk_(std::make_unique<DiskStorage<T>>(disk_path, prefix)),
          max_memory_blocks_(max_memory_blocks) {}

    bool store(const BlockKey& key, const T* data, std::size_t size) override {
        // Always store to disk for persistence
        bool disk_success = disk_->store(key, data, size);

        // Also cache in memory if space available
        memory_->store(key, data, size);

        return disk_success;
    }

    bool load(const BlockKey& key, T* data, std::size_t size) override {
        // Try memory first
        if (memory_->load(key, data, size)) {
            return true;
        }

        // Fall back to disk
        if (disk_->load(key, data, size)) {
            // Cache in memory for future access
            memory_->store(key, data, size);
            return true;
        }

        return false;
    }

    bool exists(const BlockKey& key) const override {
        return memory_->exists(key) || disk_->exists(key);
    }

    bool remove(const BlockKey& key) override {
        bool mem_removed = memory_->remove(key);
        bool disk_removed = disk_->remove(key);
        return mem_removed || disk_removed;
    }

    void clear() override {
        memory_->clear();
        disk_->clear();
    }

    std::string type() const override { return "HybridStorage"; }

    std::size_t size_bytes() const override { return memory_->size_bytes() + disk_->size_bytes(); }
};

}  // namespace mem
}  // namespace tri