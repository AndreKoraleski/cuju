// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The host-only cursor that loads or saves a state as the raw little-endian bytes of its fields.
#pragma once

#include "../host_device.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuju {
namespace state {

// One primitive field of a save, in the order of GRF's schema.
struct SchemaEntry {
    int position;
    int size;
    const char *type;
    const char *field;
};

// A field of the world, `path` as in `match.players[3].base.isActive`, `offset` in memory from
// `Archive::layoutBase`, `position` in the saved state.
struct LayoutEntry {
    std::string path;
    size_t offset;
    int position;
    int size;
    const char *type;
};

enum PointerKind {
    POINTER_PLAYER,
    POINTER_TEAM,
    POINTER_GAMER,
    POINTER_KEYBOARD,
    POINTER_ANIMATION,
    POINTER_KINDS
};

// A loaded pointer field, which the codec validates against its collection.
struct LoadedPointer {
    const int32_t *value;
    PointerKind kind;
    const char *field;
    int position;
};

// A save may also record the layout of the world. `Enter` and `Leave` give each field its path,
// and with `layoutStorage` every list visits its whole storage.
class Archive {
public:
    static Archive Loader(const std::string &data) { return Archive(true, data); }
    static Archive Saver() { return Archive(false, std::string()); }

    bool load() const { return loading; }
    const std::string &data() const { return buffer; }
    size_t position() const { return offset; }
    bool AtEnd() const { return offset == buffer.size(); }

    std::vector<SchemaEntry> *schema = nullptr;
    // The path of the next field, for the schema and the error messages.
    const char *field = "";
    std::vector<LayoutEntry> *layout = nullptr;
    const void *layoutBase = nullptr;
    // Covers every list's whole storage and the members a save skips, else only the saved fields.
    bool layoutStorage = true;

    // Each call needs a matching `Leave`. Both do nothing without a layout.
    void Enter(const std::string &name, int index = -1) {
        if (!layout) return;
        scopes.push_back(index < 0 ? name : name + "[" + std::to_string(index) + "]");
    }
    void Leave() {
        if (layout) scopes.pop_back();
    }
    bool Covering() const { return layout && layoutStorage; }
    int Count(int count, int capacity) const { return Covering() ? capacity : count; }

    template <class T>
    void Process(T &value, const char *type = "?") {
        if (schema) schema->push_back({(int)offset, (int)sizeof(T), type, field});
        if (layout) {
            std::string path;
            for (const std::string &scope : scopes) path += (path.empty() ? "" : ".") + scope;
            if (*field) path += (path.empty() ? "" : ".") + std::string(field);
            layout->push_back({path,
                               (size_t)(reinterpret_cast<const char *>(&value) -
                                        static_cast<const char *>(layoutBase)),
                               (int)offset, (int)sizeof(T), type});
        }
        if (loading) {
            if (offset + sizeof(T) > buffer.size())
                throw std::runtime_error(std::string("state too short at ") + field);
            memcpy(&value, &buffer[offset], sizeof(T));
        } else {
            buffer.append(reinterpret_cast<const char *>(&value), sizeof(T));
        }
        offset += sizeof(T);
    }

    void ProcessPointer(int32_t &value, PointerKind kind, const char *type) {
        const int position = (int)offset;
        Process(value, type);
        if (loading && pointers) pointers->push_back({&value, kind, field, position});
    }

    std::vector<LoadedPointer> *pointers = nullptr;

    void ProcessString(std::string &text) {
        int32_t length = (int32_t)text.size();
        Process(length, "i");
        if (loading) {
            if (length < 0 || (size_t)length > buffer.size() - offset)
                throw std::runtime_error(std::string("state too short at ") + field);
            text.resize(length);
        }
        for (char &character : text) Process(character, "c");
    }

private:
    Archive(bool load, std::string data) : loading(load), buffer(std::move(data)) {}
    bool loading;
    std::string buffer;
    size_t offset = 0;
    std::vector<std::string> scopes;
};

class ArchiveScope {
public:
    ArchiveScope(Archive &archive, const std::string &name, int index = -1) : archive_(archive) {
        archive_.Enter(name, index);
    }
    ~ArchiveScope() { archive_.Leave(); }
    ArchiveScope(const ArchiveScope &) = delete;
    ArchiveScope &operator=(const ArchiveScope &) = delete;

private:
    Archive &archive_;
};

// Reads the enclosing function's `Archive` parameter, which must be named `archive`.
#define PROCESS_STRUCT(x)                               \
    do {                                                \
        ::cuju::state::ArchiveScope scope(archive, #x); \
        (x).Process(archive);                           \
    } while (0)

// A state stores it as an `int32_t` count, then the elements.
template <class T, int Capacity>
struct FixedVector {
    static constexpr int CAPACITY = Capacity;
    int32_t count = 0;
    T elements[Capacity];
    CUJU_FUNCTION T &operator[](int index) { return elements[index]; }
    CUJU_FUNCTION const T &operator[](int index) const { return elements[index]; }
    CUJU_FUNCTION void push_back(const T &element) {
        if (count >= Capacity) CUJU_FATAL("a list of the state outgrows its capacity");
        elements[count++] = element;
    }
    CUJU_FUNCTION T *begin() { return elements; }
    CUJU_FUNCTION T *end() { return elements + count; }
    CUJU_FUNCTION const T *begin() const { return elements; }
    CUJU_FUNCTION const T *end() const { return elements + count; }
};

template <class T, int Capacity, class ElementFunction>
void ProcessVector(Archive &archive, FixedVector<T, Capacity> &container,
                   ElementFunction &&processElement) {
    // The name is a string literal of the caller, so it outlives the call.
    const char *const name = archive.field;
    {
        ArchiveScope scope(archive, name);
        if (archive.layout) archive.field = "count";
        archive.Process(container.count, "i");
        archive.field = name;
    }
    if (container.count < 0 || container.count > Capacity)
        throw std::runtime_error(std::string("capacity exceeded: ") + name +
                                 " n=" + std::to_string(container.count));
    for (int index = 0; index < archive.Count(container.count, Capacity); index++) {
        ArchiveScope scope(archive, name, index);
        // The layout names the element by its scope. The schema keeps the name of the list.
        archive.field = archive.layout ? "" : name;
        processElement(container.elements[index]);
    }
}

}  // namespace state
}  // namespace cuju
