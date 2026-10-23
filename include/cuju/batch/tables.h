// Copyright 2026 The cuju Authors
// SPDX-License-Identifier: Apache-2.0

// Copies the read-only tables of the simulation into a backend and rebases their pointers there.
#pragma once

#include <vector>

#include "../simulation/camera.h"
#include "../simulation/humanoid.h"
#include "../simulation/select_animation.h"

namespace cuju {
namespace batch {

template <class T>
Span<T> Rebase(Span<T> span, const T *hostBase, const T *base) {
    if (span.count == 0 || span.pointer == nullptr) return Span<T>{nullptr, span.count};
    return Span<T>{base + (span.pointer - hostBase), span.count};
}

// Fills `tables.context` with backend copies. `Tables::Upload(data, count)` and `Upload(vector)`
// return a copy that `tables` owns, or null for no values. `HOST` adds the host-only tables.
template <class Tables>
void UploadTables(Tables &tables, const animation::AnimationStore &store,
                  const std::vector<animation::AnimationDerived> &derived,
                  const std::vector<animation::Skeleton> &skeletons,
                  const simulation::StaticData &staticData) {
    const animation::KeyFrame *deviceKeys = tables.Upload(store.keys);
    const animation::InterpolationEntry *deviceInterpolation = tables.Upload(store.interpolation);
    const animation::MiddleEntry *deviceMiddle =
        Tables::HOST ? tables.Upload(store.middle) : nullptr;
    std::vector<animation::NodeAnimation> nodes = store.nodes;
    for (animation::NodeAnimation &node : nodes) {
        node.keys = Rebase(node.keys, store.keys.data(), deviceKeys);
        if (node.interpolation)
            node.interpolation =
                deviceInterpolation + (node.interpolation - store.interpolation.data());
        node.middle = node.middle && deviceMiddle
                          ? deviceMiddle + (node.middle - store.middle.data())
                          : nullptr;
    }
    const animation::NodeAnimation *deviceNodes = tables.Upload(nodes);
    const animation::FootballKeyFrame *deviceFootball = tables.Upload(store.football);
    const Vector3 *devicePositions = tables.Upload(store.positions);
    std::vector<animation::Animation> animations = store.animations;
    for (animation::Animation &entry : animations) {
        entry.nodes = Rebase(entry.nodes, store.nodes.data(), deviceNodes);
        entry.football = Rebase(entry.football, store.football.data(), deviceFootball);
        entry.positions = Rebase(entry.positions, store.positions.data(), devicePositions);
    }
    animation::AnimationCollection view = store.view;
    view.animations =
        Span<animation::Animation>{tables.Upload(animations), (int32_t)animations.size()};
    view.quadrants =
        Span<animation::Quadrant>{tables.Upload(store.quadrants), (int32_t)store.quadrants.size()};
    tables.context.animations = tables.Upload(&view, 1);
    tables.context.derived = tables.Upload(derived);
    tables.context.skeletons = tables.Upload(skeletons);
    tables.context.staticData = tables.Upload(&staticData, 1);
    std::vector<float> cameraWeights(simulation::CAMERA_WEIGHTS);
    for (int count = 1; count <= (int)simulation::cameraPositionSize; count++)
        for (int index = 0; index < count; index++)
            cameraWeights[count * (count - 1) / 2 + index] = simulation::CameraWeight(index, count);
    tables.context.cameraWeights = tables.Upload(cameraWeights);
    std::vector<int16_t> functionTypeAnimations;
    std::vector<int32_t> functionTypeOffsets;
    simulation::BuildFunctionIndex(store.view, functionTypeAnimations, functionTypeOffsets);
    tables.context.functionTypeAnimations = tables.Upload(functionTypeAnimations);
    tables.context.functionTypeOffsets = tables.Upload(functionTypeOffsets);
    std::vector<simulation::CrudeRecord> crude(store.animations.size());
    for (size_t index = 0; index < crude.size(); index++)
        crude[index] = simulation::MakeCrudeRecord(store.animations[index], derived[index]);
    tables.context.crude = tables.Upload(crude);
    // `smoothedMovement` parallels `store.positions` entry for entry.
    std::vector<simulation::SelectionRecord> selection(store.animations.size());
    std::vector<Vector3> smoothedMovement(store.positions.size());
    for (size_t index = 0; index < selection.size(); index++) {
        const animation::Animation &animationData = store.animations[index];
        const int32_t offset = (int32_t)(animationData.positions.pointer - store.positions.data());
        selection[index] = simulation::MakeSelectionRecord(animationData, derived[index], offset);
        if (selection[index].smoothedMovementOffset >= 0)
            for (uint32_t frame = 0; frame < (uint32_t)animationData.positions.size(); frame++)
                smoothedMovement[offset + frame] =
                    simulation::CalculateMovementAtFrame(animationData.positions, frame, 1);
    }
    tables.context.selection = tables.Upload(selection);
    tables.context.smoothedMovement = tables.Upload(smoothedMovement);
    tables.context.requestCommand = nullptr;
}

}  // namespace batch
}  // namespace cuju
