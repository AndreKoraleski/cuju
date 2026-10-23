# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Compares two builds byte for byte, and says where they differ.

Usage: `python ci/compare_builds.py <first> <second>`, two directories of wheels and sdists, such
as two `cibuildwheel` output directories, or two files. Inside an archive it compares member by
member, and inside an ELF library section by section. Exits with 1 unless both are identical.
"""

import hashlib
import os
import struct
import sys
import tarfile
import zipfile


def read(path):
    with open(path, 'rb') as handle:
        return handle.read()


def archive_members(path):
    """Returns `{name: (bytes, metadata)}` of a wheel or an sdist, or `None` for another file."""
    if path.endswith(('.whl', '.zip')):
        with zipfile.ZipFile(path) as archive:
            return {
                info.filename: (archive.read(info), (info.date_time, info.external_attr))
                for info in archive.infolist()
            }
    if path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(path) as archive:
            return {
                member.name: (
                    archive.extractfile(member).read() if member.isfile() else b'',
                    (member.mtime, member.mode, member.uid, member.gid, member.uname, member.gname),
                )
                for member in archive.getmembers()
            }
    return None


def elf_sections(data):
    """Returns `{name: bytes}` of the sections of a 64-bit little-endian ELF file, or `None`."""
    if data[:4] != b'\x7fELF' or data[4] != 2 or data[5] != 1:
        return None
    (offset,) = struct.unpack_from('<Q', data, 0x28)
    entry_size, count, names_index = struct.unpack_from('<HHH', data, 0x3A)
    headers = [
        struct.unpack_from('<IIQQQQIIQQ', data, offset + i * entry_size) for i in range(count)
    ]
    names_offset = headers[names_index][4]

    def name(start):
        start += names_offset
        return data[start : data.index(b'\0', start)].decode(errors='replace')

    # A section without bytes in the file (`SHT_NOBITS`, type 8) compares by its size alone.
    return {
        name(header[0]): header[5].to_bytes(8, 'little')
        if header[1] == 8
        else data[header[4] : header[4] + header[5]]
        for header in headers
    }


def explain_elf(first, second):
    """Returns the lines that name the sections in which two ELF files differ."""
    one, other = elf_sections(first), elf_sections(second)
    if one is None or other is None:
        return []
    lines = [f'section {name} only in one' for name in sorted(set(one) ^ set(other))]
    lines += [
        f'section {name} differs'
        for name in sorted(set(one) & set(other))
        if one[name] != other[name]
    ]
    return lines or ['the ELF headers differ']


def explain(first_path, second_path, first, second):
    """Returns the lines that say where two different files differ."""
    one, other = archive_members(first_path), archive_members(second_path)
    if one is None or other is None:
        return explain_elf(first, second)
    lines = [f'{name}: only in one' for name in sorted(set(one) ^ set(other))]
    for name in sorted(set(one) & set(other)):
        if one[name][0] != other[name][0]:
            lines.append(f'{name}: the bytes differ')
            lines += [f'    {line}' for line in explain_elf(one[name][0], other[name][0])]
        elif one[name][1] != other[name][1]:
            lines.append(f'{name}: the metadata differs, {one[name][1]} against {other[name][1]}')
    if not lines and list(one) != list(other):
        lines.append('the members come in another order')
    return lines or ['the same members with the same metadata, in another archive encoding']


def compare(first, second):
    """Prints how two files, or two directories file by file, compare; returns whether equal."""
    if os.path.isfile(first) and os.path.isfile(second):
        pairs = [(os.path.basename(first), first, second)]
    else:
        names = sorted(set(os.listdir(first)) | set(os.listdir(second)))
        pairs = [(name, os.path.join(first, name), os.path.join(second, name)) for name in names]
    equal = True
    for name, one, other in pairs:
        if not (os.path.isfile(one) and os.path.isfile(other)):
            print(f'{name}: in one build only')
            equal = False
            continue
        first_bytes, second_bytes = read(one), read(other)
        if first_bytes == second_bytes:
            print(f'{name}: identical, sha256 {hashlib.sha256(first_bytes).hexdigest()}')
            continue
        equal = False
        print(f'{name}: DIFFERENT')
        for line in explain(one, other, first_bytes, second_bytes):
            print(f'    {line}')
    return equal


def main():
    if len(sys.argv) != 3:
        sys.exit('usage: python ci/compare_builds.py <first> <second>')
    sys.exit(0 if compare(sys.argv[1], sys.argv[2]) else 1)


if __name__ == '__main__':
    main()
