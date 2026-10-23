// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

// The animation tables, in the state's animation index order, as plain data for host and device.
#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../quaternion.h"
#include "../span.h"
#include "../state/environment_state.h"
#include "../vector3.h"

namespace cuju {
namespace animation {

using state::Radian;

struct KeyFrame {
    int32_t frame;
    Quaternion orientation;
    Vector3 position;
};

struct InterpolationEntry {
    Quaternion orientation;
    Vector3 position;
    // Yaw of `orientation`, as `Quaternion::GetAngles` gives it.
    float angleZ;
};

struct MiddleEntry {
    Quaternion orientation;
    Vector3 position;
    Quaternion normalizedOrientation;
    // Squared length in `[1e-6, 2)`. `Normalize` scales by 2/3 to 2, so every zero keeps its sign
    // and no other element becomes zero.
    bool normalizedKeepsZeros;
};

struct NodeAnimation {
    int32_t bodyPart;
    Span<KeyFrame> keys;
    // Poses of the frames below `interpolationFrames`. Null computes every frame when read.
    const InterpolationEntry *interpolation = nullptr;
    int32_t interpolationFrames = 0;
    // Halfway poses of `interpolation`, null on the GPU, which blends the two frames itself.
    const MiddleEntry *middle = nullptr;
};

struct FootballKeyFrame {
    int32_t frame;
    Quaternion orientation;
    Vector3 position;
    float power;
};

enum StateString : int32_t {
    STRING_EMPTY = 0,
    STRING_LAY_BACK,
    STRING_LAY_FRONT,
    STRING_LEFT_ELBOW,
    STRING_RIGHT_ELBOW
};
inline int32_t InternStateString(const std::string &text) {
    static const char *const known[] = {"", "lay_back", "lay_front", "left_elbow", "right_elbow"};
    for (int32_t index = 0; index < 5; index++)
        if (text == known[index]) return index;
    throw std::runtime_error("expected a known state string in the animation tables, found `" +
                             text + "`");
}

struct Animation {
    // `animationType` is an `e_AnimationString`.
    int32_t frameCount, currentFoot, animationType;
    Vector3 translation, incomingMovement;
    float incomingVelocity;
    Vector3 outgoingMovement, outgoingDirection, incomingBodyDirection, outgoingBodyDirection;
    float outgoingVelocity;
    Radian outgoingAngle, incomingBodyAngle, outgoingBodyAngle;
    int32_t outgoingFoot;
    float idleLevel;
    int32_t quadrantID;
    float specialVariable1, specialVariable2;
    bool lastDitch, baseAnimation;
    // `StateString` values.
    int32_t outgoingSpecialState, incomingRetainState, incomingSpecialState;
    Span<NodeAnimation> nodes;
    bool hasFootball;
    Span<FootballKeyFrame> football;
    bool hasPositions;
    Span<Vector3> positions;
};

// A cell of the end states that the animation selection searches. `velocity` is an `e_Velocity`.
struct Quadrant {
    int32_t id;
    Vector3 position;
    int32_t velocity;
    Radian angle;
};

struct AnimationCollection {
    Span<Animation> animations;
    Span<Quadrant> quadrants;
    Radian maximumIncomingBallDirectionDeviation, maximumOutgoingBallDirectionDeviation;
};

// Owns the host copy. `view` points into its vectors, so it cannot be copied.
struct AnimationStore {
    std::vector<Animation> animations;
    std::vector<NodeAnimation> nodes;
    std::vector<KeyFrame> keys;
    std::vector<FootballKeyFrame> football;
    std::vector<Vector3> positions;
    std::vector<Quadrant> quadrants;
    std::vector<InterpolationEntry> interpolation;
    // At the offsets of `interpolation`.
    std::vector<MiddleEntry> middle;
    std::vector<std::string> names;
    std::vector<std::map<std::string, std::string>> variables;
    AnimationCollection view;

    AnimationStore() = default;
    AnimationStore(const AnimationStore &) = delete;
    AnimationStore &operator=(const AnimationStore &) = delete;

    std::string GetVariable(size_t animationIndex, const std::string &key) const {
        auto entry = variables[animationIndex].find(key);
        return entry == variables[animationIndex].end() ? std::string() : entry->second;
    }
};

// A cursor over little-endian bytes. `what` names the input in the messages.
class ByteReader {
public:
    ByteReader(const std::string &bytes, const char *what, size_t start)
        : bytes(bytes), what(what), offset(start) {}
    template <class T>
    T get() {
        if (offset + sizeof(T) > bytes.size())
            throw std::runtime_error(std::string(what) + " end early");
        T value;
        memcpy(&value, &bytes[offset], sizeof(T));
        offset += sizeof(T);
        return value;
    }
    std::string ReadString() {
        int32_t length = get<int32_t>();
        if (length < 0 || offset + length > bytes.size())
            throw std::runtime_error(std::string("a string of ") + what + " runs past their end");
        std::string text = bytes.substr(offset, length);
        offset += length;
        return text;
    }
    bool AtEnd() const { return offset == bytes.size(); }
    size_t Position() const { return offset; }

private:
    const std::string &bytes;
    const char *what;
    size_t offset;
};

// `animations.bin` holds `CUJANIM1`, the animations, the quadrants and the two ball direction
// deviations. A string is an int32 length and its bytes, a radian the 8 bytes of `Radian`.
inline std::unique_ptr<AnimationStore> ParseAnimations(const std::string &bytes) {
    if (bytes.compare(0, 8, "CUJANIM1") != 0)
        throw std::runtime_error("expected the animation tables, found another file");
    ByteReader reader(bytes, "the animation tables", 8);
    auto store = std::make_unique<AnimationStore>();
    AnimationStore &tables = *store;
    // The vectors grow while reading, so offsets are kept and the spans are set at the end.
    struct Ranges {
        size_t firstNode, nodeCount, firstFootballKey, footballKeyCount, firstPosition,
            positionCount;
    };
    std::vector<Ranges> ranges;
    std::vector<std::pair<size_t, size_t>> nodeKeys;
    int32_t animationCount = reader.get<int32_t>();
    tables.animations.resize(animationCount);
    tables.names.resize(animationCount);
    tables.variables.resize(animationCount);
    ranges.resize(animationCount);
    for (int32_t animationIndex = 0; animationIndex < animationCount; animationIndex++) {
        Animation &animationData = tables.animations[animationIndex];
        Ranges &range = ranges[animationIndex];
        tables.names[animationIndex] = reader.ReadString();
        animationData.frameCount = reader.get<int32_t>();
        animationData.currentFoot = reader.get<int32_t>();
        animationData.animationType = reader.get<int32_t>();
        animationData.translation = reader.get<Vector3>();
        animationData.incomingMovement = reader.get<Vector3>();
        animationData.incomingVelocity = reader.get<float>();
        animationData.outgoingMovement = reader.get<Vector3>();
        animationData.outgoingDirection = reader.get<Vector3>();
        animationData.incomingBodyDirection = reader.get<Vector3>();
        animationData.outgoingBodyDirection = reader.get<Vector3>();
        animationData.outgoingVelocity = reader.get<float>();
        animationData.outgoingAngle = reader.get<Radian>();
        animationData.incomingBodyAngle = reader.get<Radian>();
        animationData.outgoingBodyAngle = reader.get<Radian>();
        animationData.outgoingFoot = reader.get<int32_t>();
        animationData.idleLevel = reader.get<float>();
        animationData.quadrantID = reader.get<int32_t>();
        animationData.specialVariable1 = reader.get<float>();
        animationData.specialVariable2 = reader.get<float>();
        animationData.lastDitch = reader.get<uint8_t>() != 0;
        animationData.baseAnimation = reader.get<uint8_t>() != 0;
        animationData.outgoingSpecialState = InternStateString(reader.ReadString());
        animationData.incomingRetainState = InternStateString(reader.ReadString());
        animationData.incomingSpecialState = InternStateString(reader.ReadString());
        int32_t variableCount = reader.get<int32_t>();
        for (int variable = 0; variable < variableCount; variable++) {
            std::string key = reader.ReadString();
            tables.variables[animationIndex][key] = reader.ReadString();
        }
        int32_t nodeCount = reader.get<int32_t>();
        range.firstNode = tables.nodes.size();
        range.nodeCount = nodeCount;
        for (int node = 0; node < nodeCount; node++) {
            NodeAnimation nodeAnimation{};
            nodeAnimation.bodyPart = reader.get<int32_t>();
            // `AnimationApplyPrepare` indexes by body part and needs them distinct within an
            // animation.
            if (nodeAnimation.bodyPart < 0 || nodeAnimation.bodyPart >= state::BODY_PARTS)
                throw std::runtime_error(
                    "expected body parts from 0 to " + std::to_string(state::BODY_PARTS - 1) +
                    " in the animation tables, found " + std::to_string(nodeAnimation.bodyPart));
            for (size_t earlier = range.firstNode; earlier < tables.nodes.size(); earlier++)
                if (tables.nodes[earlier].bodyPart == nodeAnimation.bodyPart)
                    throw std::runtime_error(
                        "the animation tables repeat a body part within an animation");
            int32_t keyCount = reader.get<int32_t>();
            nodeKeys.push_back({tables.keys.size(), (size_t)keyCount});
            for (int keyIndex = 0; keyIndex < keyCount; keyIndex++) {
                KeyFrame keyFrame;
                keyFrame.frame = reader.get<int32_t>();
                keyFrame.orientation = reader.get<Quaternion>();
                keyFrame.position = reader.get<Vector3>();
                tables.keys.push_back(keyFrame);
            }
            tables.nodes.push_back(nodeAnimation);
        }
        animationData.hasFootball = reader.get<int32_t>() != 0;
        range.firstFootballKey = tables.football.size();
        range.footballKeyCount = 0;
        if (animationData.hasFootball) {
            int32_t footballKeyCount = reader.get<int32_t>();
            range.footballKeyCount = footballKeyCount;
            for (int footballIndex = 0; footballIndex < footballKeyCount; footballIndex++) {
                FootballKeyFrame footballKey;
                footballKey.frame = reader.get<int32_t>();
                footballKey.orientation = reader.get<Quaternion>();
                footballKey.position = reader.get<Vector3>();
                footballKey.power = reader.get<float>();
                tables.football.push_back(footballKey);
            }
        }
        int32_t positionCount = reader.get<int32_t>();
        animationData.hasPositions = positionCount >= 0;
        range.firstPosition = tables.positions.size();
        range.positionCount = positionCount > 0 ? positionCount : 0;
        for (int position = 0; position < positionCount; position++)
            tables.positions.push_back(reader.get<Vector3>());
    }
    int32_t quadrantCount = reader.get<int32_t>();
    tables.quadrants.resize(quadrantCount);
    for (Quadrant &quadrant : tables.quadrants) {
        quadrant.id = reader.get<int32_t>();
        quadrant.position = reader.get<Vector3>();
        quadrant.velocity = reader.get<int32_t>();
        quadrant.angle = reader.get<Radian>();
    }
    tables.view.maximumIncomingBallDirectionDeviation = reader.get<Radian>();
    tables.view.maximumOutgoingBallDirectionDeviation = reader.get<Radian>();
    if (!reader.AtEnd())
        throw std::runtime_error("the animation tables hold bytes after their end");
    for (size_t node = 0; node < tables.nodes.size(); node++)
        tables.nodes[node].keys = Span<KeyFrame>{tables.keys.data() + nodeKeys[node].first,
                                                 (int32_t)nodeKeys[node].second};
    for (int32_t animationIndex = 0; animationIndex < animationCount; animationIndex++) {
        Animation &animationData = tables.animations[animationIndex];
        const Ranges &range = ranges[animationIndex];
        animationData.nodes =
            Span<NodeAnimation>{tables.nodes.data() + range.firstNode, (int32_t)range.nodeCount};
        animationData.football = Span<FootballKeyFrame>{
            tables.football.data() + range.firstFootballKey, (int32_t)range.footballKeyCount};
        animationData.positions = Span<Vector3>{tables.positions.data() + range.firstPosition,
                                                (int32_t)range.positionCount};
    }
    tables.view.animations =
        Span<Animation>{tables.animations.data(), (int32_t)tables.animations.size()};
    tables.view.quadrants =
        Span<Quadrant>{tables.quadrants.data(), (int32_t)tables.quadrants.size()};
    return store;
}

}  // namespace animation
}  // namespace cuju
