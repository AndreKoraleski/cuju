// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The static scene graph of a humanoid, and the world transforms and bounding boxes of its nodes.
#pragma once

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../inline_vector.h"
#include "../quaternion.h"
#include "../state/environment_state.h"
#include "../vector3.h"
#include "animation_data.h"

namespace cuju {
namespace animation {

constexpr int LOCAL_MODE_RELATIVE = 0, LOCAL_MODE_ABSOLUTE = 1;

struct SkeletonGeometry {
    bool footOrLowerLeg;
    int32_t localMode;
    Vector3 position;
    Quaternion rotation;
    Vector3 scale;
    // In mesh space.
    Vector3 boundingBoxMinimum, boundingBoxMaximum;
};
constexpr int MAXIMUM_GEOMETRIES_PER_NODE = 2;
struct SkeletonNode {
    int32_t parent;
    int32_t localMode;
    // The transform at load. The state holds the live one.
    Vector3 position;
    Quaternion rotation;
    Vector3 scale;
    InlineVector<SkeletonGeometry, MAXIMUM_GEOMETRIES_PER_NODE> geometries;
};
struct Skeleton {
    int32_t nodeMap[state::BODY_PARTS];
    // Pre-order, so every parent comes before its children.
    InlineVector<SkeletonNode, state::HUMANOID_NODES> nodes;
};

inline std::vector<Skeleton> ParseSkeletons(const std::string &bytes) {
    if (bytes.compare(0, 7, "CUJHUM1") != 0)
        throw std::runtime_error("expected the skeletons of a bundle, found other bytes");
    ByteReader reader(bytes, "the skeletons", 7);
    std::vector<Skeleton> skeletons(reader.get<int32_t>());
    for (Skeleton &skeleton : skeletons) {
        for (int bodyPart = 0; bodyPart < state::BODY_PARTS; bodyPart++)
            skeleton.nodeMap[bodyPart] = reader.get<int32_t>();
        skeleton.nodes.resize(reader.get<int32_t>());
        for (SkeletonNode &node : skeleton.nodes) {
            reader.ReadString();
            node.parent = reader.get<int32_t>();
            node.localMode = reader.get<int32_t>();
            node.position = reader.get<Vector3>();
            node.rotation = reader.get<Quaternion>();
            node.scale = reader.get<Vector3>();
            node.geometries.resize(reader.get<int32_t>());
            for (SkeletonGeometry &geometry : node.geometries) {
                const std::string name = reader.ReadString();
                geometry.footOrLowerLeg = name == "left_foot" || name == "right_foot" ||
                                          name == "left_lowerleg" || name == "right_lowerleg";
                geometry.localMode = reader.get<int32_t>();
                geometry.position = reader.get<Vector3>();
                geometry.rotation = reader.get<Quaternion>();
                geometry.scale = reader.get<Vector3>();
                geometry.boundingBoxMinimum = reader.get<Vector3>();
                geometry.boundingBoxMaximum = reader.get<Vector3>();
            }
        }
    }
    if (!reader.AtEnd()) throw std::runtime_error("the skeletons hold bytes after their end");
    return skeletons;
}

inline int BodyPartFromString(const std::string &part) {
    static const char *names[] = {"middle",        "neck",           "left_thigh", "right_thigh",
                                  "left_knee",     "right_knee",     "left_ankle", "right_ankle",
                                  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                                  "body",          "player"};
    for (int bodyPart = 0; bodyPart < 14; bodyPart++)
        if (part == names[bodyPart]) return bodyPart;
    return -1;
}

// A world transform.
struct Derived {
    Vector3 position, scale;
    Quaternion rotation;
};
CUJU_FUNCTION_NO_INLINE inline Derived DeriveChild(const Derived *parent, int localMode,
                                                   const Vector3 &position,
                                                   const Quaternion &rotation,
                                                   const Vector3 &scale) {
    Derived derived;
    if (localMode == LOCAL_MODE_RELATIVE && parent) {
        derived.rotation = (parent->rotation * rotation).GetNormalized();
        derived.scale = parent->scale * scale;
        derived.position = parent->rotation * (parent->scale * position);
        derived.position += parent->position;
    } else {
        derived.rotation = rotation;
        derived.scale = scale;
        derived.position = position;
    }
    return derived;
}

CUJU_FUNCTION inline void DeriveNodes(const Skeleton &skeleton, const state::NodeState *nodes,
                                      Derived derived[state::HUMANOID_NODES]) {
    for (size_t node = 0; node < skeleton.nodes.size(); node++) {
        const Derived *parent =
            skeleton.nodes[node].parent >= 0 ? &derived[skeleton.nodes[node].parent] : nullptr;
        derived[node] = DeriveChild(parent, nodes[node].localMode, nodes[node].position,
                                    nodes[node].rotation, nodes[node].scale);
    }
}

struct AxisAlignedBoundingBox {
    Vector3 minimumCorner, maximumCorner;

    // GRF: the maximum starts at `numeric_limits<float>::min() / 100`, just above zero, so on an
    // axis where every point is negative it stays near zero. Collisions depend on it.
    CUJU_FUNCTION void Reset() {
        const float smallest = std::numeric_limits<float>::min() / 100;
        const float largest = std::numeric_limits<float>::max() / 100;
        minimumCorner.Set(largest, largest, largest);
        maximumCorner.Set(smallest, smallest, smallest);
    }
    CUJU_FUNCTION_NO_INLINE AxisAlignedBoundingBox Rotated(const Quaternion &rotation) const {
        Vector3 corners[8] = {Vector3(minimumCorner.coordinates[0], minimumCorner.coordinates[1],
                                      minimumCorner.coordinates[2]),
                              Vector3(maximumCorner.coordinates[0], minimumCorner.coordinates[1],
                                      minimumCorner.coordinates[2]),
                              Vector3(minimumCorner.coordinates[0], maximumCorner.coordinates[1],
                                      minimumCorner.coordinates[2]),
                              Vector3(maximumCorner.coordinates[0], maximumCorner.coordinates[1],
                                      minimumCorner.coordinates[2]),
                              Vector3(minimumCorner.coordinates[0], minimumCorner.coordinates[1],
                                      maximumCorner.coordinates[2]),
                              Vector3(maximumCorner.coordinates[0], minimumCorner.coordinates[1],
                                      maximumCorner.coordinates[2]),
                              Vector3(minimumCorner.coordinates[0], maximumCorner.coordinates[1],
                                      maximumCorner.coordinates[2]),
                              Vector3(maximumCorner.coordinates[0], maximumCorner.coordinates[1],
                                      maximumCorner.coordinates[2])};
        AxisAlignedBoundingBox bounds;
        bounds.Reset();
        for (Vector3 &corner : corners) {
            corner = rotation * corner;
            if (corner.coordinates[0] < bounds.minimumCorner.coordinates[0])
                bounds.minimumCorner.coordinates[0] = corner.coordinates[0];
            if (corner.coordinates[0] > bounds.maximumCorner.coordinates[0])
                bounds.maximumCorner.coordinates[0] = corner.coordinates[0];
            if (corner.coordinates[1] < bounds.minimumCorner.coordinates[1])
                bounds.minimumCorner.coordinates[1] = corner.coordinates[1];
            if (corner.coordinates[1] > bounds.maximumCorner.coordinates[1])
                bounds.maximumCorner.coordinates[1] = corner.coordinates[1];
            if (corner.coordinates[2] < bounds.minimumCorner.coordinates[2])
                bounds.minimumCorner.coordinates[2] = corner.coordinates[2];
            if (corner.coordinates[2] > bounds.maximumCorner.coordinates[2])
                bounds.maximumCorner.coordinates[2] = corner.coordinates[2];
        }
        return bounds;
    }
    CUJU_FUNCTION AxisAlignedBoundingBox Translated(const Vector3 &offset) const {
        AxisAlignedBoundingBox bounds = *this;
        bounds.minimumCorner += offset;
        bounds.maximumCorner += offset;
        return bounds;
    }
    CUJU_FUNCTION_NO_INLINE bool Intersects(const Vector3 &center, float radius) const {
        float difference, squaredDistance = 0;
        for (int axis = 0; axis < 3; axis++) {
            if (center.coordinates[axis] < minimumCorner.coordinates[axis]) {
                difference = center.coordinates[axis] - minimumCorner.coordinates[axis];
                squaredDistance += difference * difference;
            } else if (center.coordinates[axis] > maximumCorner.coordinates[axis]) {
                difference = center.coordinates[axis] - maximumCorner.coordinates[axis];
                squaredDistance += difference * difference;
            }
        }
        return squaredDistance <= radius * radius;
    }
    // Exact: each coordinate divides by 2 in double and rounds to float.
    CUJU_FUNCTION Vector3 Center() const {
        float x = (minimumCorner.coordinates[0] + maximumCorner.coordinates[0]) / 2.0;
        float y = (minimumCorner.coordinates[1] + maximumCorner.coordinates[1]) / 2.0;
        float z = (minimumCorner.coordinates[2] + maximumCorner.coordinates[2]) / 2.0;
        return Vector3(x, y, z);
    }
    // Exact: squares and roots in double, as `std::pow(float, int)` does, rounded to float twice.
    CUJU_FUNCTION float Radius() const {
        float x = maximumCorner.coordinates[0] - minimumCorner.coordinates[0];
        float y = maximumCorner.coordinates[1] - minimumCorner.coordinates[1];
        float z = maximumCorner.coordinates[2] - minimumCorner.coordinates[2];
        float length = (float)sqrt((double)x * x + (double)y * y);
        return (float)(sqrt((double)length * length + (double)z * z) / 2.0);
    }
};

struct WorldGeometry {
    AxisAlignedBoundingBox boundingBox;
    Vector3 position;
};
// GRF: the box is the mesh box rotated and moved by the world transform. The scale does not enter.
CUJU_FUNCTION inline WorldGeometry GeometryWorld(const SkeletonGeometry &geometry,
                                                 const Derived &node) {
    Derived derived = DeriveChild(&node, geometry.localMode, geometry.position, geometry.rotation,
                                  geometry.scale);
    AxisAlignedBoundingBox local{geometry.boundingBoxMinimum, geometry.boundingBoxMaximum};
    return WorldGeometry{local.Rotated(derived.rotation).Translated(derived.position),
                         derived.position};
}

}  // namespace animation
}  // namespace cuju
