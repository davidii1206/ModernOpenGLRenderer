#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace gfx {

class AssetCache {
public:
    AssetCache() = default;
    ~AssetCache() = default;

    AssetCache(const AssetCache&) = delete;
    AssetCache& operator=(const AssetCache&) = delete;

    template<typename T, typename LoadFn>
    std::shared_ptr<T> get(const std::string& key, LoadFn&& loader) {
        auto it = entries_.find(key);
        if (it != entries_.end()) {
            return std::static_pointer_cast<T>(it->second);
        }
        auto asset = loader();
        if (asset) {
            entries_[key] = asset;
        }
        return asset;
    }

    template<typename T>
    std::shared_ptr<T> get(const std::string& key) const {
        auto it = entries_.find(key);
        if (it != entries_.end()) {
            return std::static_pointer_cast<T>(it->second);
        }
        return nullptr;
    }

    template<typename T>
    void put(const std::string& key, std::shared_ptr<T> asset) {
        entries_[key] = std::move(asset);
    }

    bool contains(const std::string& key) const {
        return entries_.find(key) != entries_.end();
    }

    void remove(const std::string& key) {
        entries_.erase(key);
    }

    void clear() {
        entries_.clear();
    }

    std::size_t count() const {
        return entries_.size();
    }

private:
    std::unordered_map<std::string, std::shared_ptr<void>> entries_;
};

} // namespace gfx
