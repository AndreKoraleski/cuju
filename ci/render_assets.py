# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Converts GRF's public domain media into the web assets of cuju's browser renderer.

    python ci/render_assets.py <media> [--out python/cuju/render/web/assets]

<media> is GRF's `third_party/gfootball_engine/data/media`. The team kits come from the sibling
`data/databases`, or from `--databases`. The script writes the glb meshes, the textures and
`manifest.json`, then reloads every glb and checks it. The output depends only on the media and
on the versions of Pillow and libwebp, so a rerun reproduces the files byte for byte.

GRF's media is public domain under the Unlicense, `third_party/gfootball_engine/LICENSE`.
"""

import argparse
import io
import json
import math
import os
import re
import struct
import sys
import xml.etree.ElementTree as ElementTree

import numpy as np
from PIL import Image

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'python', 'cuju', 'render', 'web', 'assets'
)

# The shared skeleton of the renderer, in preorder, as GRF's Node::GetNodes walks player.object
# under the root that the humanoid code names player.
NODES = [
    'player',
    'body',
    'middle',
    'neck',
    'left_shoulder',
    'left_elbow',
    'right_shoulder',
    'right_elbow',
    'left_thigh',
    'left_knee',
    'left_ankle',
    'right_thigh',
    'right_knee',
    'right_ankle',
]
PARENTS = [-1, 0, 1, 2, 2, 4, 2, 6, 1, 8, 9, 1, 11, 12]

# GRF's constants from gamedefines.hpp.
DEFAULT_PLAYER_HEIGHT = 1.92
PITCH_HALF_W = 55.0
PITCH_HALF_H = 36.0
PITCH_FULL_HALF_W = 60.0
PITCH_FULL_HALF_H = 40.0

# The appearance of GRF's eleven players by index, the same on both teams, from PlayerData.
PLAYERS = [
    (1, 'long02', 'blonde', 1.87),
    (2, 'short02', 'black', 1.70),
    (4, 'long02', 'black', 1.74),
    (1, 'medium01', 'black', 1.93),
    (1, 'short01', 'black', 1.72),
    (4, 'long02', 'black', 1.71),
    (3, 'long02', 'black', 1.89),
    (1, 'long01', 'blonde', 1.84),
    (1, 'medium01', 'black', 1.74),
    (2, 'long02', 'black', 1.81),
    (3, 'short02', 'black', 1.69),
]
HAIRSTYLES = ['long01', 'long02', 'medium01', 'short01', 'short02']
HAIR_COLORS = ['black', 'blonde', 'brown', 'darkblonde', 'red']

# The fullbody materials by GRF's diffuse map, grouped as the renderer colors them. GRF swaps
# every skin.jpg diffuse for the player's skin tone and kit_template.png for the team's kit.
FULLBODY_GROUPS = {
    'kit_template.png': 'kit',
    'skin.jpg': 'skin',
    'shoe.jpg': 'shoe',
    'shoe_sole.jpg': 'sole',
}
GROUP_ORDER = ['kit', 'skin', 'shoe', 'sole']

FLOAT, UBYTE, USHORT, UINT = 5126, 5121, 5123, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


# ASE parsing, as GRF's tree loader and ASELoader read the files.


class AseEntry:
    """One starred line of an ASE file with its values and its braced block."""

    __slots__ = ('children', 'name', 'values')

    def __init__(self, name, values, children):
        self.name = name
        self.values = values
        self.children = children

    def find(self, name):
        """Returns the first child entry of that name, or None."""
        for child in self.children or ():
            if child.name == name:
                return child
        return None

    def all(self, name):
        """Returns every child entry of that name, in file order."""
        return [child for child in self.children or () if child.name == name]


def parse_ase(path):
    """Reads an ASE file into a tree of entries."""
    root = AseEntry('', [], [])
    stack = [root]
    with open(path, encoding='latin-1') as handle:
        for line in handle:
            tokens = re.findall(r'"[^"]*"|\S+', line)
            if not tokens:
                continue
            if tokens[0] == '}':
                stack.pop()
                continue
            if not tokens[0].startswith('*'):
                continue
            values = tokens[1:]
            children = None
            if values and values[-1] == '{':
                values = values[:-1]
                children = []
            entry = AseEntry(tokens[0][1:], values, children)
            stack[-1].children.append(entry)
            if children is not None:
                stack.append(entry)
    return root


def _f32(text):
    return np.float32(float(text))


def _unquote(text):
    return text[1:-1] if text.startswith('"') else text


def load_ase(path):
    """Loads an ASE file as GRF's ASELoader builds its triangle meshes.

    Returns the materials, each a dictionary of the diffuse, bump, shine and self illumination map
    names, and the meshes, one per GEOMOBJECT, each with float32 corner arrays of shape [T, 3, 3]
    for positions, normals and texture coordinates. GRF negates the second texture coordinate.
    """
    tree = parse_ase(path)
    materials = []
    material_list = tree.find('MATERIAL_LIST')
    for material in material_list.all('MATERIAL'):
        maps = {}
        for key in ('MAP_DIFFUSE', 'MAP_BUMP', 'MAP_SHINE', 'MAP_SELFILLUM'):
            block = material.find(key)
            maps[key] = _unquote(block.find('BITMAP').values[0]) if block else None
        maps['name'] = _unquote(material.find('MATERIAL_NAME').values[0])
        materials.append(maps)
    meshes = []
    for geom in tree.all('GEOMOBJECT'):
        node_tm = geom.find('NODE_TM')
        mesh = geom.find('MESH')
        vertex_count = int(mesh.find('MESH_NUMVERTEX').values[0])
        vertices = np.zeros((vertex_count, 3), np.float32)
        for entry in mesh.find('MESH_VERTEX_LIST').children:
            vertices[int(entry.values[0])] = [_f32(value) for value in entry.values[1:4]]
        face_list = mesh.find('MESH_FACE_LIST').children
        faces = np.zeros((len(face_list), 3), np.int64)
        for entry in face_list:
            index = int(entry.values[0].rstrip(':'))
            faces[index] = [int(entry.values[2]), int(entry.values[4]), int(entry.values[6])]
        # GRF turns the vertex normals by the node rotation with normalized rows, row vector first.
        rotation = np.array(
            [
                [_f32(value) for value in node_tm.find(row).values[:3]]
                for row in ('TM_ROW0', 'TM_ROW1', 'TM_ROW2')
            ],
            np.float32,
        )
        rotation /= np.linalg.norm(rotation, axis=1, keepdims=True)
        normals = np.zeros((len(faces), 3, 3), np.float32)
        normal_entries = mesh.find('MESH_NORMALS').children
        for start in range(0, len(normal_entries), 4):
            face = int(normal_entries[start].values[0])
            for corner in range(3):
                values = normal_entries[start + 1 + corner].values[1:4]
                normal = np.array([_f32(value) for value in values], np.float32) @ rotation
                normals[face, corner] = normal / np.linalg.norm(normal)
        uvs = np.zeros((len(faces), 3, 3), np.float32)
        if int(mesh.find('MESH_NUMTVERTEX').values[0]) > 0:
            tverts = np.zeros((int(mesh.find('MESH_NUMTVERTEX').values[0]), 3), np.float32)
            for entry in mesh.find('MESH_TVERTLIST').children:
                u, v, w = (_f32(value) for value in entry.values[1:4])
                tverts[int(entry.values[0])] = [u, -v, w]
            for entry in mesh.find('MESH_TFACELIST').children:
                uvs[int(entry.values[0])] = tverts[[int(value) for value in entry.values[1:4]]]
        reference = geom.find('MATERIAL_REF')
        meshes.append(
            {
                'name': _unquote(geom.find('NODE_NAME').values[0]),
                'material': int(reference.values[0]) if reference else -1,
                'positions': vertices[faces],
                'normals': normals,
                'uvs': uvs,
                'tree': mesh,
                'faces': faces,
                'vertices': vertices,
            }
        )
    return materials, meshes


def vertex_colors(meshes):
    """Maps each fullbody vertex position to its rounded vertex color, as GRF's GetVertexColors.

    The first color seen for a position wins, over the objects and their color faces in order.
    """
    colors = {}
    for mesh in meshes:
        tree = mesh['tree']
        palette = []
        for entry in tree.find('MESH_CVERTLIST').children:
            channels = np.array([_f32(value) for value in entry.values[1:4]], np.float32)
            channels = channels * np.float32(255)
            palette.append(tuple(math.floor(float(channel) + 0.5) for channel in channels))
        cfaces = {}
        for entry in tree.find('MESH_CFACELIST').children:
            cfaces[int(entry.values[0])] = [int(value) for value in entry.values[1:4]]
        for face in range(len(cfaces)):
            for corner in range(3):
                position = tuple(mesh['vertices'][mesh['faces'][face, corner]].tolist())
                colors.setdefault(position, palette[cfaces[face][corner]])
    return colors


# Quaternions are stored x, y, z, w like GRF's, with GRF's Hamilton product.


def quat_mul(a, b):
    """Returns the product a times b."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by + ay * bw + az * bx - ax * bz,
            aw * bz + az * bw + ax * by - ay * bx,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_rotate(q, v):
    """Rotates the vectors v of shape [..., 3] by q."""
    axis = np.asarray(q[:3], np.float64)
    v = np.asarray(v, np.float64)
    uv = np.cross(axis, v)
    uuv = np.cross(axis, uv)
    return v + uv * (2.0 * q[3]) + uuv * 2.0


def quat_inverse(q):
    """Returns the inverse of a unit quaternion."""
    return np.array([-q[0], -q[1], -q[2], q[3]]) / np.linalg.norm(q)


def quat_matrix(q):
    """Returns the 3x3 rotation matrix of q."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def angle_axis(degrees, axis):
    """Returns the quaternion of GRF's rotation strings, degrees then an unnormalized axis."""
    half = math.radians(degrees) * 0.5
    return np.array(
        [
            math.sin(half) * axis[0],
            math.sin(half) * axis[1],
            math.sin(half) * axis[2],
            math.cos(half),
        ]
    )


# The skeleton.


def load_skeleton(media):
    """Reads player.object into local positions and rotations in the order of NODES."""
    root = ElementTree.parse(os.path.join(media, 'objects', 'players', 'player.object')).getroot()
    names, parents, positions, rotations = [], [], [], []

    def visit(element, parent):
        names.append(element.findtext('name').strip())
        parents.append(parent)
        index = len(names) - 1
        values = [float(value) for value in element.findtext('position').split(',')]
        positions.append(np.array(values))
        degrees, *axis = (float(value) for value in element.findtext('rotation').split(','))
        rotations.append(angle_axis(degrees, axis))
        for child in element.findall('node'):
            visit(child, index)

    names.append('player')
    parents.append(-1)
    positions.append(np.zeros(3))
    rotations.append(np.array([0.0, 0.0, 0.0, 1.0]))
    for child in root.findall('node'):
        visit(child, 0)
    if names != NODES or parents != PARENTS:
        raise ValueError(f'player.object has nodes {names} with parents {parents}')
    return np.array(positions), np.array(rotations)


def load_pose(path):
    """Reads frame 0 of a GRF .anim.util file, the local rotation of each node by name."""
    rotations = {}
    with open(path, encoding='latin-1') as handle:
        for line in handle:
            if line.startswith('<') or line.startswith('extension'):
                break
            tokens = line.strip().split(',')
            if tokens[0] == 'player' or len(tokens) < 6:
                continue
            rotations[tokens[0]] = np.array([float(value) for value in tokens[2:6]])
    return rotations


def world_pose(local_positions, local_rotations):
    """Composes local transforms into world positions and rotations, as GRF's Spatial does."""
    positions = np.zeros((len(NODES), 3))
    rotations = np.zeros((len(NODES), 4))
    for index, parent in enumerate(PARENTS):
        if parent < 0:
            positions[index] = local_positions[index]
            rotations[index] = local_rotations[index]
            continue
        rotation = quat_mul(rotations[parent], local_rotations[index])
        rotations[index] = rotation / np.linalg.norm(rotation)
        offset = quat_rotate(rotations[parent], local_positions[index])
        positions[index] = offset + positions[parent]
    return positions, rotations


def animation_pose(local_positions, path):
    """Applies frame 0 of an animation like GRF's Animation::Apply, replacing the rotations."""
    pose = load_pose(path)
    rotations = [np.array([0.0, 0.0, 0.0, 1.0])]
    for name in NODES[1:]:
        rotation = pose[name]
        rotations.append(rotation / np.linalg.norm(rotation))
    return world_pose(local_positions, np.array(rotations))


# The fullbody player.


def prepare_fullbody(media):
    """Ports HumanoidBase::PrepareFullbodyModel and the first UpdateFullbodyModel.

    GRF models fullbody.ase in the pose of base.anim.util. It merges the corners of each object
    that share a position and a texture coordinate, keeping the first normal. Each vertex color
    channel holds a joint and a weight as joint times 10 plus weight times 9. GRF then moves every
    vertex into the straight pose, all world rotations identity at the joint positions of
    straight.anim.util, by blending the bones' transforms. That straight mesh is the bind pose.
    """
    materials, meshes = load_ase(
        os.path.join(media, 'objects', 'players', 'models', 'fullbody.ase')
    )
    colors = vertex_colors(meshes)
    local_positions, rest_rotations = load_skeleton(media)
    animations = os.path.join(media, 'animations')
    base_positions, base_rotations = animation_pose(
        local_positions, os.path.join(animations, 'base.anim.util')
    )
    straight_positions, straight_rotations = animation_pose(
        local_positions, os.path.join(animations, 'straight.anim.util')
    )
    if not np.allclose(straight_rotations, [0, 0, 0, 1], atol=1e-6):
        raise ValueError('straight.anim.util is not the identity pose')

    positions, normals, uvs, joints, weights, groups, sources = [], [], [], [], [], [], []
    multi_bone = 0
    for mesh in meshes:
        diffuse = os.path.basename(materials[mesh['material']]['MAP_DIFFUSE'])
        group = FULLBODY_GROUPS[diffuse]
        corner_positions = mesh['positions'].reshape(-1, 3)
        corner_normals = mesh['normals'].reshape(-1, 3)
        corner_uvs = mesh['uvs'].reshape(-1, 3)
        unique = {}
        first = []
        indices = []
        for corner in range(len(corner_positions)):
            key = (tuple(corner_positions[corner].tolist()), tuple(corner_uvs[corner].tolist()))
            if key not in unique:
                unique[key] = len(first)
                first.append(corner)
            indices.append(unique[key])
        offset = len(positions)
        for corner in first:
            position = corner_positions[corner]
            color = colors[tuple(position.tolist())]
            bones = []
            total = 0.0
            for channel in color:
                joint = math.floor(channel * 0.1)
                weight = (channel - joint * 10.0) / 9.0
                total += weight
                bones.append((joint, weight))
            if bones[0][1] == 0.0:
                raise ValueError(f'vertex {position} has no weight on its first channel')
            kept = {}
            for joint, weight in bones:
                if weight > 0.01:
                    kept[joint] = kept.get(joint, 0.0) + weight / total
            # Two channels on one joint move a vertex like their summed weight, and glTF lists
            # each joint of a vertex once.
            bones = list(kept.items())
            multi_bone += len(bones) > 1
            # GRF's joint ids index the nodes under player, so the renderer's index is one more.
            moved = np.zeros(3)
            normal = np.zeros(3)
            for joint, weight in bones:
                node = joint + 1
                inverse = quat_inverse(base_rotations[node])
                moved += weight * (
                    quat_rotate(inverse, position - base_positions[node]) + straight_positions[node]
                )
                normal += weight * quat_rotate(inverse, corner_normals[corner])
            positions.append(moved)
            normals.append(normal / np.linalg.norm(normal))
            uvs.append(corner_uvs[corner][:2])
            joints.append([joint + 1 for joint, _ in bones] + [0] * (4 - len(bones)))
            weights.append([weight for _, weight in bones] + [0.0] * (4 - len(bones)))
            sources.append(position)
        groups.append((group, mesh['name'], np.array(indices, np.int64).reshape(-1, 3) + offset))

    return {
        'positions': np.array(positions, np.float32),
        'normals': np.array(normals, np.float32),
        'uvs': _gltf_uv(np.array(uvs, np.float32)),
        'joints': np.array(joints, np.uint8),
        'weights': np.array(weights, np.float32),
        'groups': groups,
        'sources': np.array(sources, np.float64),
        'local_positions': local_positions,
        'rest_rotations': rest_rotations,
        'straight_positions': straight_positions,
        'base_positions': base_positions,
        'base_rotations': base_rotations,
        'multi_bone': multi_bone,
    }


def _gltf_uv(uvs):
    # GRF's second coordinate is the ASE's negated, and its textures start at the top row like
    # glTF's, so adding one keeps it the same modulo one and inside the unit square.
    out = uvs.copy()
    out[:, 1] = out[:, 1] + np.float32(1.0)
    return out


def flatten_mesh(mesh):
    """Merges identical corners of a GRF triangle soup into indexed vertices."""
    positions = mesh['positions'].reshape(-1, 3)
    normals = mesh['normals'].reshape(-1, 3)
    uvs = mesh['uvs'].reshape(-1, 3)[:, :2]
    stacked = np.concatenate([positions, normals, uvs], axis=1)
    unique, first, inverse = np.unique(stacked, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    vertices = unique[order]
    return {
        'positions': np.ascontiguousarray(vertices[:, 0:3], np.float32),
        'normals': np.ascontiguousarray(vertices[:, 3:6], np.float32),
        'uvs': _gltf_uv(np.ascontiguousarray(vertices[:, 6:8], np.float32)),
        'indices': rank[inverse.reshape(-1)].reshape(-1, 3),
    }


# glTF writing.


class Gltf:
    """Builds a binary glTF 2.0 file from numpy arrays."""

    def __init__(self):
        self.doc = {
            'asset': {'version': '2.0', 'generator': 'cuju ci/render_assets.py'},
            'scene': 0,
            'scenes': [{'nodes': []}],
            'nodes': [],
            'meshes': [],
            'materials': [],
            'accessors': [],
            'bufferViews': [],
            'buffers': [{'byteLength': 0}],
        }
        self.binary = bytearray()

    def view(self, data, target=None):
        """Appends raw bytes as a buffer view and returns its index."""
        while len(self.binary) % 4:
            self.binary.append(0)
        view = {'buffer': 0, 'byteOffset': len(self.binary), 'byteLength': len(data)}
        if target is not None:
            view['target'] = target
        self.binary.extend(data)
        self.doc['bufferViews'].append(view)
        return len(self.doc['bufferViews']) - 1

    def accessor(self, array, kind, target=None, bounds=False):
        """Appends an array as an accessor of type kind and returns its index."""
        array = np.ascontiguousarray(array)
        component = {
            np.dtype(np.float32): FLOAT,
            np.dtype(np.uint8): UBYTE,
            np.dtype(np.uint16): USHORT,
            np.dtype(np.uint32): UINT,
        }[array.dtype]
        accessor = {
            'bufferView': self.view(array.tobytes(), target),
            'componentType': component,
            'count': int(array.shape[0]),
            'type': kind,
        }
        if bounds:
            flat = array.reshape(array.shape[0], -1)
            accessor['min'] = [_number(value) for value in flat.min(axis=0)]
            accessor['max'] = [_number(value) for value in flat.max(axis=0)]
        self.doc['accessors'].append(accessor)
        return len(self.doc['accessors']) - 1

    def indices(self, triangles, vertex_count):
        """Appends triangle indices in the smallest unsigned type that holds them."""
        flat = np.asarray(triangles).reshape(-1)
        dtype = np.uint16 if vertex_count < 65536 else np.uint32
        return self.accessor(flat.astype(dtype), 'SCALAR', ELEMENT_ARRAY_BUFFER, bounds=True)

    def material(self, name, color=(1.0, 1.0, 1.0, 1.0), texture=None, **extra):
        """Appends a matte material and returns its index."""
        material = {
            'name': name,
            'pbrMetallicRoughness': {
                'baseColorFactor': [float(value) for value in color],
                'metallicFactor': 0.0,
                'roughnessFactor': 0.9,
            },
        }
        if texture is not None:
            material['pbrMetallicRoughness']['baseColorTexture'] = {'index': texture}
        material.update(extra)
        self.doc['materials'].append(material)
        return len(self.doc['materials']) - 1

    def texture(self, data, mime):
        """Embeds an image and returns the texture index."""
        self.doc.setdefault('images', []).append({'bufferView': self.view(data), 'mimeType': mime})
        self.doc.setdefault(
            'samplers', [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}]
        )
        image = len(self.doc['images']) - 1
        texture = {'sampler': 0}
        if mime == 'image/webp':
            texture['extensions'] = {'EXT_texture_webp': {'source': image}}
            for key in ('extensionsUsed', 'extensionsRequired'):
                if 'EXT_texture_webp' not in self.doc.setdefault(key, []):
                    self.doc[key].append('EXT_texture_webp')
        else:
            texture['source'] = image
        self.doc.setdefault('textures', []).append(texture)
        return len(self.doc['textures']) - 1

    def node(self, node, root=False):
        """Appends a node and returns its index."""
        self.doc['nodes'].append(node)
        index = len(self.doc['nodes']) - 1
        if root:
            self.doc['scenes'][0]['nodes'].append(index)
        return index

    def static_mesh(self, name, parts):
        """Appends a mesh of indexed parts, each a vertex dictionary with a material index."""
        primitives = []
        for vertices, material in parts:
            count = len(vertices['positions'])
            attributes = {
                'POSITION': self.accessor(vertices['positions'], 'VEC3', ARRAY_BUFFER, True),
                'NORMAL': self.accessor(vertices['normals'], 'VEC3', ARRAY_BUFFER),
                'TEXCOORD_0': self.accessor(vertices['uvs'], 'VEC2', ARRAY_BUFFER),
            }
            primitives.append(
                {
                    'attributes': attributes,
                    'indices': self.indices(vertices['indices'], count),
                    'material': material,
                }
            )
        self.doc['meshes'].append({'name': name, 'primitives': primitives})
        return len(self.doc['meshes']) - 1

    def write(self, path):
        """Writes the glb file."""
        if not self.doc['materials']:
            del self.doc['materials']
        while len(self.binary) % 4:
            self.binary.append(0)
        self.doc['buffers'][0]['byteLength'] = len(self.binary)
        text = json.dumps(self.doc, separators=(',', ':')).encode('utf-8')
        text += b' ' * (-len(text) % 4)
        total = 12 + 8 + len(text) + 8 + len(self.binary)
        with open(path, 'wb') as handle:
            handle.write(struct.pack('<4sII', b'glTF', 2, total))
            handle.write(struct.pack('<I4s', len(text), b'JSON'))
            handle.write(text)
            handle.write(struct.pack('<I4s', len(self.binary), b'BIN\x00'))
            handle.write(bytes(self.binary))


def _number(value):
    value = float(value)
    return int(value) if value.is_integer() and abs(value) < 2**53 else value


def _translation_matrix(offset):
    matrix = np.eye(4)
    matrix[:3, 3] = offset
    return matrix


def write_player(fullbody, path):
    """Writes the skinned fullbody player in the engine frame."""
    gltf = Gltf()
    local = fullbody['local_positions']
    children = [[] for _ in NODES]
    for index, parent in enumerate(PARENTS):
        if parent >= 0:
            children[parent].append(index)
    for index, name in enumerate(NODES):
        node = {'name': name}
        if np.any(local[index]):
            node['translation'] = [float(value) for value in local[index]]
        if children[index]:
            node['children'] = children[index]
        gltf.node(node, root=index == 0)
    # The bind pose is the straight pose, every world rotation identity, so each inverse bind
    # matrix only takes the joint's straight position away.
    inverse_bind = np.stack(
        [
            _translation_matrix(-position).T.reshape(-1)
            for position in fullbody['straight_positions']
        ]
    ).astype(np.float32)
    gltf.doc['skins'] = [
        {
            'name': 'grf_humanoid',
            'joints': list(range(len(NODES))),
            'skeleton': 0,
            'inverseBindMatrices': gltf.accessor(inverse_bind, 'MAT4'),
        }
    ]
    attributes = {
        'POSITION': gltf.accessor(fullbody['positions'], 'VEC3', ARRAY_BUFFER, True),
        'NORMAL': gltf.accessor(fullbody['normals'], 'VEC3', ARRAY_BUFFER),
        'TEXCOORD_0': gltf.accessor(fullbody['uvs'], 'VEC2', ARRAY_BUFFER),
        'JOINTS_0': gltf.accessor(fullbody['joints'], 'VEC4', ARRAY_BUFFER),
        'WEIGHTS_0': gltf.accessor(fullbody['weights'], 'VEC4', ARRAY_BUFFER),
    }
    textures = {
        'kit': 'kit texture of the team, or kit_mask.png recolored',
        'skin': 'skin_0N.png of the player',
        'shoe': 'shoe.webp',
        'sole': 'shoe_sole.png',
    }
    primitives = []
    for group in GROUP_ORDER:
        triangles = [indices for name, _, indices in fullbody['groups'] if name == group]
        material = gltf.material(group, extras={'texture': textures[group]})
        primitives.append(
            {
                'attributes': attributes,
                'indices': gltf.indices(np.concatenate(triangles), len(fullbody['positions'])),
                'material': material,
            }
        )
    gltf.doc['meshes'].append({'name': 'fullbody', 'primitives': primitives})
    gltf.node({'name': 'fullbody', 'mesh': 0, 'skin': 0}, root=True)
    gltf.write(path)


def write_static(path, name, parts, materials):
    """Writes an unskinned glb from parts of indexed vertices and material dictionaries."""
    gltf = Gltf()
    indices = []
    for material in materials:
        material = dict(material)
        texture = material.pop('texture', None)
        if texture is not None:
            material['texture'] = gltf.texture(*texture)
        indices.append(gltf.material(**material))
    mesh = gltf.static_mesh(name, [(vertices, indices[slot]) for vertices, slot in parts])
    gltf.node({'name': name, 'mesh': mesh}, root=True)
    gltf.write(path)


def merge_parts(meshes):
    """Concatenates flattened meshes into one indexed vertex set."""
    out = {'positions': [], 'normals': [], 'uvs': [], 'indices': []}
    offset = 0
    for mesh in meshes:
        flat = flatten_mesh(mesh)
        for key in ('positions', 'normals', 'uvs'):
            out[key].append(flat[key])
        out['indices'].append(flat['indices'] + offset)
        offset += len(flat['positions'])
    return {key: np.concatenate(value) for key, value in out.items()}


# Textures.


def save_image(image, path, lossless=False, quality=85):
    """Saves a WebP or PNG, deterministically, without metadata."""
    if path.endswith('.png'):
        image.save(path, format='PNG', optimize=True)
    elif lossless:
        image.save(path, format='WEBP', lossless=True, quality=100, method=6, exact=False)
    else:
        image.save(path, format='WEBP', quality=quality, method=6)


def open_rgb(path):
    """Opens a BMP as RGB, or RGBA when its alpha channel carries anything."""
    image = Image.open(path)
    image.load()
    if image.mode == 'RGBA' and image.getchannel('A').getextrema() == (255, 255):
        return image.convert('RGB')
    return image.convert('RGBA') if 'A' in image.mode else image.convert('RGB')


def kit_mask(media, fullbody, size):
    """Paints which kit texel is shirt, shorts or socks into red, green and blue.

    GRF's kit_template.png tints the shirt in one brown and the collar, shorts and socks in a
    darker brown over a green grid. The shirt hem is the row below the last shirt tint, the socks
    are the rows of the separate sock objects, and the shorts lie between. The regions grow into
    the unused black so that texture filtering at their edges keeps the kit colors.
    """
    template = np.asarray(
        Image.open(
            os.path.join(media, 'objects', 'players', 'textures', 'kit_template.bmp')
        ).convert('RGB')
    )
    shirt = np.all(template == (78, 44, 14), axis=2)
    dark = np.all(template == (50, 34, 19), axis=2)
    grid = np.all(template == (0, 169, 0), axis=2)
    height, width = shirt.shape
    rows = np.arange(height)[:, None]
    # The grid lines are two texels wide, so a grid texel takes a tint found two texels away on
    # both sides. The second pass fills the crossings of the lines.
    for _ in range(2):
        for tint in (shirt, dark):
            fill = grid & ~shirt & ~dark
            for axis in (0, 1):
                tint |= fill & _shift(tint, 2, axis) & _shift(tint, -2, axis)
    hem = int(np.nonzero(shirt.any(axis=1))[0].max()) + 1
    sock_top = _sock_top(fullbody, height)
    labels = np.zeros((height, width), np.int8)
    labels[shirt | (dark & (rows < hem))] = 1
    labels[dark & (rows >= hem) & (rows < sock_top)] = 2
    labels[dark & (rows >= sock_top)] = 3
    # Each pass grows the regions by one texel into the unused texels, the first direction first.
    for _ in range(8):
        grown = labels.copy()
        for axis, step in ((0, 1), (0, -1), (1, 1), (1, -1)):
            neighbor = _shift(labels, step, axis)
            take = (grown == 0) & (neighbor > 0)
            grown[take] = neighbor[take]
        labels = grown
    mask = np.zeros((height, width, 3), np.uint8)
    for label in (1, 2, 3):
        mask[labels == label, label - 1] = 255
    image = Image.fromarray(mask, 'RGB')
    return image.resize((size, size), Image.Resampling.BOX), labels, hem, sock_top


def kit_colors(kit, labels):
    """Returns the median sRGB color of a kit's shirt, shorts and socks, for flat drawings."""
    pixels = np.asarray(kit.convert('RGB'))
    colors = {}
    for label, part in ((1, 'shirt'), (2, 'shorts'), (3, 'socks')):
        region = pixels[labels == label]
        colors[part] = [int(value) for value in np.median(region, axis=0)]
    return colors


def _shift(array, step, axis):
    # Moves an array by step along axis and fills the uncovered edge with zeros, without wrapping.
    out = np.zeros_like(array)
    source = [slice(None), slice(None)]
    target = [slice(None), slice(None)]
    if step > 0:
        source[axis], target[axis] = slice(None, -step), slice(step, None)
    else:
        source[axis], target[axis] = slice(-step, None), slice(None, step)
    out[tuple(target)] = array[tuple(source)]
    return out


def _sock_top(fullbody, height):
    # The socks start two texels above the highest texel row that the sock objects reach.
    sock_rows = []
    for group, name, triangles in fullbody['groups']:
        if group == 'kit' and name.startswith('sock'):
            sock_rows.append(fullbody['uvs'][triangles.reshape(-1), 1].min())
    return math.floor(min(sock_rows) * height) - 2


def bake_pitch(media, width, height, normal_width, normal_height):
    """Bakes GRF's procedural pitch, GetPitchDiffuseColor and GetPitchNormalColor.

    GRF adds Perlin noise and per texel random noise from an unseeded generator, and picks the
    mowing band width at random. The bake leaves the noise out and takes the 11 m bands. The image
    is north up, its top row at y 40 and its left column at x minus 60, the full extent.
    """
    pitch = os.path.join(media, 'textures', 'pitch')
    seamless = np.asarray(
        Image.open(os.path.join(pitch, 'seamlessgrass08.bmp')).convert('RGB'), np.float32
    )
    overlay = np.asarray(Image.open(os.path.join(pitch, 'overlay.bmp')).convert('RGBA'), np.float32)
    overlay_h, overlay_w = overlay.shape[:2]
    # GRF bakes four quadrants at 2048 by 1024, so its texels sit on the overlay's texels.
    grid_x = (
        np.arange(overlay_w, dtype=np.float32) / overlay_w * 2 * PITCH_FULL_HALF_W
        - PITCH_FULL_HALF_W
    )
    grid_y = (
        np.arange(overlay_h, dtype=np.float32) / overlay_h * 2 * PITCH_FULL_HALF_H
        - PITCH_FULL_HALF_H
    )
    x = grid_x[None, :]
    y = grid_y[:, None]
    contrast = 0.4
    red_to_blue = 1.0
    base = np.array(
        [
            (35 - contrast * 10) * red_to_blue * 2.0,
            46 * 2.0,
            (25 - contrast * 10) * (2.0 - red_to_blue) * 2.0,
        ],
        np.float32,
    )
    seam_h, seam_w = seamless.shape[:2]
    tex_scale = 0.32
    sx = np.fmod((x / PITCH_FULL_HALF_W * 0.5 + 0.5) * seam_w * 18.0 * tex_scale, seam_w)
    sy = np.fmod((y / PITCH_FULL_HALF_H * 0.5 + 0.5) * seam_h * 12.0 * tex_scale, seam_h)
    sx = np.broadcast_to(sx, (overlay_h, overlay_w))
    sy = np.broadcast_to(sy, (overlay_h, overlay_w))
    x1 = np.floor(sx).astype(np.int64)
    y1 = np.floor(sy).astype(np.int64)
    x2 = (x1 + 1) % seam_w
    y2 = (y1 + 1) % seam_h
    bx = (sx - x1)[..., None]
    by = (sy - y1)[..., None]
    tex = (seamless[y1, x1] * (1 - bx) + seamless[y1, x2] * bx) * (1 - by) + (
        seamless[y2, x1] * (1 - bx) + seamless[y2, x2] * bx
    ) * by
    color = base * 0.7 + tex * 0.3
    distance = np.sqrt((x / PITCH_HALF_W) ** 2 + (y / PITCH_HALF_H) ** 2)
    darkness = 1.0 - np.power(np.clip(distance * 0.7, 0.0, 1.0), 1.5) * 0.18
    color = color * darkness[..., None]
    alpha = overlay[..., 3:4] / 256.0
    color = np.clip(color * (1.0 - alpha) + overlay[..., :3] * alpha, 0, 255)
    diffuse = Image.fromarray(color.astype(np.uint8)[::-1], 'RGB')
    diffuse = diffuse.resize((width, height), Image.Resampling.BOX)

    nx = (
        np.arange(normal_width, dtype=np.float64) + 0.5
    ) / normal_width * 2 * PITCH_FULL_HALF_W - PITCH_FULL_HALF_W
    ny = (
        PITCH_FULL_HALF_H
        - (np.arange(normal_height, dtype=np.float64) + 0.5) / normal_height * 2 * PITCH_FULL_HALF_H
    )
    inside = (np.abs(nx)[None, :] < PITCH_HALF_W) & (np.abs(ny)[:, None] < PITCH_HALF_H)
    normal = np.zeros((normal_height, normal_width, 3))
    normal[..., 2] = 1.0
    band_y = _grass_direction(ny / 11.0)[:, None] * 0.1
    band_x = _grass_direction(nx / 11.0)[None, :] * 0.12
    normal[..., 0] += np.where(inside, band_y, 0.0)
    normal[..., 1] += np.where(inside, band_x, 0.0)
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    encoded = np.clip(np.floor((normal * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
    return diffuse, Image.fromarray(encoded, 'RGB')


def _curve(source, bias):
    return (np.sin((source - 0.5) * math.pi) * 0.5 + 0.5) * bias + source * (1.0 - bias)


def _grass_direction(coord):
    bias = coord - np.floor(coord)
    bias = np.sin(bias * 2 * math.pi) * 0.5 + 0.5
    for _ in range(5):
        bias = _curve(bias, 1.0)
    return bias * 2.0 - 1.0


# Checks.


def read_glb(path):
    """Reads a glb into its JSON document and binary chunk."""
    with open(path, 'rb') as handle:
        data = handle.read()
    magic, version, length = struct.unpack_from('<4sII', data, 0)
    if magic != b'glTF' or version != 2 or length != len(data):
        raise ValueError(f'{path}: bad glb header')
    json_length, json_type = struct.unpack_from('<I4s', data, 12)
    if json_type != b'JSON':
        raise ValueError(f'{path}: first chunk is not JSON')
    doc = json.loads(data[20 : 20 + json_length])
    offset = 20 + json_length
    bin_length, bin_type = struct.unpack_from('<I4s', data, offset)
    if bin_type != b'BIN\x00' or offset + 8 + bin_length != len(data):
        raise ValueError(f'{path}: bad binary chunk')
    return doc, data[offset + 8 : offset + 8 + bin_length]


def read_accessor(doc, binary, index):
    """Returns an accessor's data as a numpy array of shape [count, components]."""
    accessor = doc['accessors'][index]
    view = doc['bufferViews'][accessor['bufferView']]
    dtype = {FLOAT: np.float32, UBYTE: np.uint8, USHORT: np.uint16, UINT: np.uint32}[
        accessor['componentType']
    ]
    width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[accessor['type']]
    start = view['byteOffset'] + accessor.get('byteOffset', 0)
    count = accessor['count']
    size = count * width * np.dtype(dtype).itemsize
    if start % np.dtype(dtype).itemsize or start + size > view['byteOffset'] + view['byteLength']:
        raise ValueError(f'accessor {index} lies outside its buffer view')
    if view['byteOffset'] + view['byteLength'] > len(binary):
        raise ValueError(f'buffer view of accessor {index} lies outside the buffer')
    return np.frombuffer(binary, dtype, count * width, start).reshape(count, width)


def node_matrix(node):
    """Returns a node's local 4x4 matrix from its TRS."""
    matrix = np.eye(4)
    if 'rotation' in node:
        matrix[:3, :3] = quat_matrix(np.array(node['rotation']))
    if 'scale' in node:
        matrix[:3, :3] = matrix[:3, :3] * np.array(node['scale'])
    if 'translation' in node:
        matrix[:3, 3] = node['translation']
    return matrix


def world_matrices(doc):
    """Returns every node's world matrix from the scene's roots."""
    worlds = [None] * len(doc['nodes'])

    def visit(index, parent):
        worlds[index] = parent @ node_matrix(doc['nodes'][index])
        for child in doc['nodes'][index].get('children', []):
            visit(child, worlds[index])

    for root in doc['scenes'][doc.get('scene', 0)]['nodes']:
        visit(root, np.eye(4))
    return worlds


def check_glb(path, fullbody=None):
    """Checks a glb's structure, its accessor bounds, weights and inverse bind matrices.

    For the player it also skins the mesh into GRF's base pose and compares the vertices that
    follow one bone with fullbody.ase, which GRF modelled in that pose.
    """
    doc, binary = read_glb(path)
    name = os.path.basename(path)
    for index, accessor in enumerate(doc['accessors']):
        data = read_accessor(doc, binary, index)
        if 'min' in accessor:
            if not (
                np.allclose(data.min(axis=0), accessor['min'], atol=0)
                and np.allclose(data.max(axis=0), accessor['max'], atol=0)
            ):
                raise ValueError(f'{name}: accessor {index} bounds differ from its data')
        if accessor['componentType'] == FLOAT and not np.all(np.isfinite(data)):
            raise ValueError(f'{name}: accessor {index} holds a non-finite number')
    for mesh in doc['meshes']:
        for primitive in mesh['primitives']:
            attributes = primitive['attributes']
            count = doc['accessors'][attributes['POSITION']]['count']
            for accessor in attributes.values():
                if doc['accessors'][accessor]['count'] != count:
                    raise ValueError(f'{name}: attribute counts differ')
            indices = read_accessor(doc, binary, primitive['indices'])
            if len(indices) % 3 or indices.max() >= count:
                raise ValueError(f'{name}: indices outside the vertices')
            normals = read_accessor(doc, binary, attributes['NORMAL'])
            if np.abs(np.linalg.norm(normals, axis=1) - 1).max() > 1e-4:
                raise ValueError(f'{name}: normals are not unit length')
            if 'WEIGHTS_0' in attributes:
                weights = read_accessor(doc, binary, attributes['WEIGHTS_0'])
                joints = read_accessor(doc, binary, attributes['JOINTS_0'])
                if np.abs(weights.sum(axis=1) - 1).max() > 1e-6 or weights.min() < 0:
                    raise ValueError(f'{name}: weights do not sum to one')
                if joints.max() >= len(doc['skins'][0]['joints']):
                    raise ValueError(f'{name}: joint index outside the skin')
                for row, used in zip(joints, weights > 0, strict=True):
                    if len(set(row[used].tolist())) != int(used.sum()):
                        raise ValueError(f'{name}: a vertex lists a joint twice')
    for image in doc.get('images', []):
        view = doc['bufferViews'][image['bufferView']]
        data = binary[view['byteOffset'] : view['byteOffset'] + view['byteLength']]
        _decode(data)
    report = {'file': name}
    if 'skins' in doc:
        skin = doc['skins'][0]
        worlds = world_matrices(doc)
        inverse = read_accessor(doc, binary, skin['inverseBindMatrices'])
        inverse = inverse.reshape(-1, 4, 4).transpose(0, 2, 1)
        for joint, matrix in zip(skin['joints'], inverse, strict=True):
            if np.abs(matrix @ worlds[joint] - np.eye(4)).max() > 1e-6:
                raise ValueError(f'{name}: inverse bind of joint {joint} misses identity')
        if fullbody is not None:
            report['base_pose_error_m'] = _base_pose_error(doc, binary, inverse, fullbody)
    return report


def _decode(data):
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def _base_pose_error(doc, binary, inverse, fullbody):
    attributes = doc['meshes'][0]['primitives'][0]['attributes']
    positions = read_accessor(doc, binary, attributes['POSITION']).astype(np.float64)
    joints = read_accessor(doc, binary, attributes['JOINTS_0'])
    weights = read_accessor(doc, binary, attributes['WEIGHTS_0']).astype(np.float64)
    pose = []
    for node in range(len(NODES)):
        world = np.eye(4)
        world[:3, :3] = quat_matrix(fullbody['base_rotations'][node])
        world[:3, 3] = fullbody['base_positions'][node]
        pose.append(world @ inverse[node])
    pose = np.array(pose)
    homogeneous = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
    skinned = np.zeros((len(positions), 3))
    for slot in range(4):
        moved = np.einsum('nij,nj->ni', pose[joints[:, slot]], homogeneous)[:, :3]
        skinned += moved * weights[:, slot : slot + 1]
    single = weights[:, 1] == 0
    error = np.linalg.norm(skinned - fullbody['sources'], axis=1)
    if error[single].max() > 1e-5:
        raise ValueError('player.glb does not skin back to fullbody.ase in the base pose')
    return {
        'single_bone_max': float(error[single].max()),
        'blended_max': float(error[~single].max()),
        'blended_mean': float(error[~single].mean()),
    }


# The run.


def build(media, databases, out):
    """Writes every asset and the manifest, then checks the glb files."""
    os.makedirs(out, exist_ok=True)
    textures = os.path.join(media, 'objects', 'players', 'textures')
    files = []

    def record(file, role, sources, **details):
        entry = {'file': file, 'role': role, 'source': sources}
        entry.update(details)
        files.append(entry)

    fullbody = prepare_fullbody(media)
    write_player(fullbody, os.path.join(out, 'player.glb'))
    group_counts = {}
    for group, mesh_name, _ in fullbody['groups']:
        group_counts.setdefault(group, []).append(mesh_name)
    record(
        'player.glb',
        'player',
        [
            'objects/players/models/fullbody.ase',
            'objects/players/player.object',
            'animations/base.anim.util',
            'animations/straight.anim.util',
        ],
        skinned=True,
        joints=NODES,
        vertices=len(fullbody['positions']),
        triangles=int(sum(len(t) for _, _, t in fullbody['groups'])),
        materials={group: group_counts[group] for group in GROUP_ORDER},
        material_textures={
            'kit': 'a texture painted from kit_mask.png, or kit_goalkeeper.webp',
            'skin': 'skin_0N.png for the player, N in 1 to 4',
            'shoe': 'shoe.webp',
            'sole': 'shoe_sole.png',
        },
        normal_textures={'kit': 'kit_normal.webp'},
        bind_pose='each joint at its straight position with identity world rotation',
    )

    hair_entries = {}
    for style in HAIRSTYLES:
        _, meshes = load_ase(
            os.path.join(media, 'objects', 'players', 'hairstyles', style + '.ase')
        )
        merged = merge_parts(meshes)
        file = f'hair_{style}.glb'
        write_static(
            os.path.join(out, file),
            'hair_' + style,
            [(merged, 0)],
            [{'name': 'hair', 'extras': {'texture': 'hair_<color>.png'}}],
        )
        hair_entries[style] = file
        record(
            file,
            'hair',
            [f'objects/players/hairstyles/{style}.ase'],
            attach_joint='neck',
            attach_index=NODES.index('neck'),
            offset={'translation': [0, 0, 0], 'rotation': [0, 0, 0, 1]},
            material_textures={'hair': 'hair_<color>.png'},
        )

    _, meshes = load_ase(os.path.join(media, 'objects', 'balls', 'generic.ase'))
    ball_texture = _encode(open_rgb(os.path.join(media, 'objects', 'balls', 'ball.bmp')), 'webp')
    merged = merge_parts(meshes)
    ball_radius = float(np.linalg.norm(merged['positions'], axis=1).max())
    write_static(
        os.path.join(out, 'ball.glb'),
        'ball',
        [(merged, 0)],
        [{'name': 'ball', 'texture': (ball_texture, 'image/webp')}],
    )
    record(
        'ball.glb',
        'ball',
        ['objects/balls/generic.ase', 'objects/balls/ball.bmp'],
        origin='the ball centre, place it at the recorded ball position and rotation',
        radius_m=round(ball_radius, 4),
        embedded_textures=['ball.bmp as WebP'],
    )

    _, meshes = load_ase(os.path.join(media, 'objects', 'stadiums', 'goals.ase'))
    net = Image.open(os.path.join(media, 'textures', 'stadium', 'goalnetting.bmp'))
    net = _encode(net.convert('RGBA'), 'png')
    frame_parts = [mesh for mesh in meshes if mesh['material'] == 0]
    net_parts = [mesh for mesh in meshes if mesh['material'] == 1]
    frame = merge_parts(frame_parts)
    netting = merge_parts(net_parts)
    write_static(
        os.path.join(out, 'goal.glb'),
        'goals',
        [(frame, 0), (netting, 1)],
        [
            {'name': 'frame'},
            {
                'name': 'net',
                'texture': (net, 'image/png'),
                'alphaMode': 'MASK',
                'alphaCutoff': 0.5,
                'doubleSided': True,
            },
        ],
    )
    record(
        'goal.glb',
        'goals',
        ['objects/stadiums/goals.ase', 'textures/stadium/goalnetting.bmp'],
        placement='both goals in world coordinates of the engine frame, place at the origin',
        frame_bounds_m=_bounds(frame['positions']),
        net_bounds_m=_bounds(netting['positions']),
        embedded_textures=['goalnetting.bmp as PNG'],
        note='GRF deforms the netting where the ball touches it, the mesh is the rest shape',
    )

    mask, labels, hem, sock_top = kit_mask(media, fullbody, 512)
    # The kits of GRF's team database belong to real clubs, so only the goalkeeper's ships.
    for file, source, role in (
        (
            'kit_goalkeeper.webp',
            os.path.join(textures, 'goalie_kit.bmp'),
            'kit of both goalkeepers',
        ),
    ):
        kit = open_rgb(source)
        save_image(kit, os.path.join(out, file))
        record(
            file,
            role,
            [_relative(source, media, databases)],
            srgb=True,
            applies_to='material kit of player.glb',
            region_colors=kit_colors(kit, labels),
        )
    save_image(mask, os.path.join(out, 'kit_mask.png'))
    record(
        'kit_mask.png',
        'kit recolor mask',
        ['objects/players/textures/kit_template.bmp', 'objects/players/models/fullbody.ase'],
        channels={'r': 'shirt and collar', 'g': 'shorts', 'b': 'socks'},
        use='albedo = r * shirt + g * shorts + b * socks, in sRGB, on the kit material',
        template_rows={'shirt_hem': hem, 'socks_top': sock_top, 'of': 1024},
    )
    normal = Image.open(os.path.join(textures, 'kit_UVWnormal.bmp')).convert('RGB')
    save_image(
        normal.resize((512, 512), Image.Resampling.BOX),
        os.path.join(out, 'kit_normal.webp'),
        lossless=True,
    )
    record(
        'kit_normal.webp',
        'kit normal map',
        ['objects/players/textures/kit_UVWnormal.bmp'],
        srgb=False,
        applies_to='material kit of player.glb, tangent space',
    )

    skin_colors = []
    for tone in range(1, 5):
        image = open_rgb(os.path.join(textures, f'skin0{tone}.bmp')).convert('RGB')
        save_image(image, os.path.join(out, f'skin_0{tone}.png'))
        skin_colors.append(list(image.getpixel((0, 0))))
        record(
            f'skin_0{tone}.png',
            'skin tone',
            [f'objects/players/textures/skin0{tone}.bmp'],
            srgb=True,
            color=skin_colors[-1],
            applies_to='material skin of player.glb',
        )
    for color in HAIR_COLORS:
        image = open_rgb(os.path.join(textures, 'hair', color + '.bmp')).convert('RGB')
        save_image(image, os.path.join(out, f'hair_{color}.png'))
        record(
            f'hair_{color}.png',
            'hair color',
            [f'objects/players/textures/hair/{color}.bmp'],
            srgb=True,
            applies_to='material hair of hair_<style>.glb',
        )
    shoe = open_rgb(os.path.join(textures, 'shoe.bmp'))
    save_image(shoe.resize((256, 256), Image.Resampling.LANCZOS), os.path.join(out, 'shoe.webp'))
    record(
        'shoe.webp',
        'shoe',
        ['objects/players/textures/shoe.bmp'],
        srgb=True,
        applies_to='material shoe of player.glb',
    )
    save_image(
        open_rgb(os.path.join(textures, 'shoe_sole.bmp')), os.path.join(out, 'shoe_sole.png')
    )
    record(
        'shoe_sole.png',
        'shoe sole',
        ['objects/players/textures/shoe_sole.bmp'],
        srgb=True,
        applies_to='material sole of player.glb',
    )

    diffuse, pitch_normal = bake_pitch(media, 2048, 1024, 1024, 512)
    save_image(diffuse, os.path.join(out, 'pitch.webp'))
    pitch_extent = {
        'x': [-PITCH_FULL_HALF_W, PITCH_FULL_HALF_W],
        'y': [-PITCH_FULL_HALF_H, PITCH_FULL_HALF_H],
    }
    record(
        'pitch.webp',
        'pitch albedo with line markings',
        ['textures/pitch/seamlessgrass08.bmp', 'textures/pitch/overlay.bmp'],
        srgb=True,
        extent_m=pitch_extent,
        orientation='north up, top row at y = 40, left column at x = -60',
        note='replaces pitch_01 to pitch_04, which GRF overwrites with this procedural bake',
    )
    save_image(pitch_normal, os.path.join(out, 'pitch_normal.webp'), lossless=True)
    record(
        'pitch_normal.webp',
        'pitch normal map, the mowing bands',
        ['onthepitch/proceduralpitch.cpp GetPitchNormalColor'],
        srgb=False,
        extent_m=pitch_extent,
        orientation='north up, as pitch.webp',
        encoding='rgb = (n + 1) / 2 with r along +x, g along +y, b along +z of the engine',
    )
    grass = open_rgb(os.path.join(media, 'textures', 'pitch', 'seamlessgrass08.bmp'))
    save_image(grass.resize((512, 512), Image.Resampling.LANCZOS), os.path.join(out, 'grass.webp'))
    record(
        'grass.webp',
        'tiling grass detail',
        ['textures/pitch/seamlessgrass08.bmp'],
        srgb=True,
        tile_m=[
            round(2 * PITCH_FULL_HALF_W / (18 * 0.32), 4),
            round(2 * PITCH_FULL_HALF_H / (12 * 0.32), 4),
        ],
    )

    stadium = os.path.join(media, 'textures', 'stadium')
    save_image(open_rgb(os.path.join(stadium, 'crowd01.bmp')), os.path.join(out, 'crowd.webp'))
    record('crowd.webp', 'crowd, tiling, with alpha', ['textures/stadium/crowd01.bmp'], srgb=True)
    save_image(
        open_rgb(os.path.join(stadium, 'crowd01_normal.bmp')).convert('RGB'),
        os.path.join(out, 'crowd_normal.webp'),
        lossless=True,
    )
    record(
        'crowd_normal.webp', 'crowd normal map', ['textures/stadium/crowd01_normal.bmp'], srgb=False
    )
    for file, source in (
        ('adboard_polygon.webp', 'ad_polygon01.bmp'),
        ('adboard_your_ad_here.webp', 'ad_your_ad_here.bmp'),
    ):
        path = os.path.join(media, 'textures', 'adboards', source)
        save_image(open_rgb(path).convert('RGB'), os.path.join(out, file))
        record(file, 'adboard', [f'textures/adboards/{source}'], srgb=True)

    checks = [check_glb(os.path.join(out, 'player.glb'), fullbody)]
    for entry in files:
        if entry['file'].endswith('.glb') and entry['file'] != 'player.glb':
            checks.append(check_glb(os.path.join(out, entry['file'])))

    manifest = {
        'version': 1,
        'generator': 'ci/render_assets.py',
        'source': {
            'project': 'Google Research Football',
            'path': 'third_party/gfootball_engine/data/media',
            'license': 'Unlicense, public domain, third_party/gfootball_engine/LICENSE',
        },
        'frame': {
            'axes': 'engine frame, x along the pitch, y across it, z up, right handed',
            'units': 'meters',
            'gltf': 'every glb stays in the engine frame and is not turned to y up, so the viewer '
            'maps engine (x, y, z) to three (x, z, -y) once at the scene root, a turn of '
            '-90 degrees about x',
            'quaternions': 'x, y, z, w',
            'texture_coordinates': 'glTF, origin at the top left texel',
        },
        'skeleton': {
            'nodes': NODES,
            'parents': PARENTS,
            'rest_translation_local': fullbody['local_positions'].round(6).tolist(),
            'rest_translation_world': fullbody['straight_positions'].round(6).tolist(),
            'rest_rotation': 'identity for every node, local and world',
            'player_object_rotations': fullbody['rest_rotations'].round(6).tolist(),
            'note': "GRF's animations replace every local rotation of player.object, so the "
            'rotations written there never reach a pose. The bind pose of player.glb is '
            "straight.anim.util, player.object's positions with identity rotations, and "
            'joint world matrices from a recording, times the inverse bind matrices, give '
            "GRF's skinning exactly.",
        },
        'player_scale': {
            'reference_height_m': DEFAULT_PLAYER_HEIGHT,
            'rule': "GRF scales the skinned body by height / 1.92 about the player's ground "
            'point, the xy of node player, and moves the hair to the scaled neck '
            'without scaling the hair itself',
        },
        'appearance': {
            'players': [
                {
                    'index': index,
                    'skin': f'skin_0{skin}.png',
                    'hair': hair_entries[style],
                    'hair_color': f'hair_{color}.png',
                    'height_m': height,
                }
                for index, (skin, style, color, height) in enumerate(PLAYERS)
            ],
            'same_for_both_teams': True,
            'goalkeepers': 'kit_goalkeeper.webp',
            'teams': [
                {
                    'side': 'left',
                    'name': 'Frequentists United',
                    'radar_color': [255, 100, 100],
                    'radar_secondary': [100, 100, 255],
                },
                {
                    'side': 'right',
                    'name': 'Real Bayesians',
                    'radar_color': [255, 255, 255],
                    'radar_secondary': [50, 50, 126],
                },
            ],
        },
        'pitch': {
            'goal_lines_x': [-PITCH_HALF_W, PITCH_HALF_W],
            'touch_lines_y': [-PITCH_HALF_H, PITCH_HALF_H],
            'full_extent_m': pitch_extent,
            'specular': 'GRF bakes 2 plus 18 times the Perlin noise, without the noise 11 of 255',
        },
        'files': files,
        'checks': checks,
    }
    for entry in files:
        entry['bytes'] = os.path.getsize(os.path.join(out, entry['file']))
    text = json.dumps(manifest, indent=2)
    # Lists of numbers and of short names stay on one line, which keeps the file readable.
    text = re.sub(
        r'\[\s+((?:-?[0-9.eE+-]+|"[a-z_]+")(?:,\s+(?:-?[0-9.eE+-]+|"[a-z_]+"))*)\s+\]',
        lambda match: '[' + re.sub(r',\s+', ', ', match.group(1)) + ']',
        text,
    )
    with open(os.path.join(out, 'manifest.json'), 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(text + '\n')
    return manifest, fullbody


def _bounds(positions):
    return {
        'min': positions.astype(np.float64).min(axis=0).round(4).tolist(),
        'max': positions.astype(np.float64).max(axis=0).round(4).tolist(),
    }


def _encode(image, kind):
    buffer = io.BytesIO()
    if kind == 'png':
        image.save(buffer, format='PNG', optimize=True)
    else:
        image.save(buffer, format='WEBP', quality=85, method=6)
    return buffer.getvalue()


def _relative(path, media, databases):
    for root, prefix in ((media, ''), (databases, '../databases/')):
        relative = os.path.relpath(path, root)
        if not relative.startswith('..'):
            return prefix + relative.replace(os.sep, '/')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('media', help="GRF's third_party/gfootball_engine/data/media")
    parser.add_argument('--databases', help="GRF's data/databases, default next to media")
    parser.add_argument('--out', default=DEFAULT_OUT, help='the assets directory to write')
    args = parser.parse_args(argv)
    media = os.path.abspath(args.media)
    databases = args.databases or os.path.join(os.path.dirname(media), 'databases')
    manifest, fullbody = build(media, databases, os.path.abspath(args.out))
    total = sum(entry['bytes'] for entry in manifest['files'])
    for entry in manifest['files']:
        print(f'{entry["bytes"]:>9}  {entry["file"]}')
    print(f'{total:>9}  total, {fullbody["multi_bone"]} blended vertices')
    for check in manifest['checks']:
        print('checked', json.dumps(check))
    return 0


if __name__ == '__main__':
    sys.exit(main())
