# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Procedural mesh generation for primitive shapes.

Generates Panda3D GeomNode objects for basic shapes (sphere, cube, cone,
cylinder) with correct normals and UV coordinates. These are used for
Habitat-compatible object primitives and test fixtures.
"""

from __future__ import annotations

import math

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LColor,
    NodePath,
)


def make_sphere(
    name: str = "sphere",
    radius: float = 1.0,
    slices: int = 32,
    stacks: int = 16,
    color: tuple = (0.8, 0.8, 0.8, 1.0),
) -> NodePath:
    """Create a UV sphere mesh with normals.

    Parameters
    ----------
    name : str
        Node name.
    radius : float
        Sphere radius.
    slices : int
        Number of longitudinal divisions.
    stacks : int
        Number of latitudinal divisions.
    color : tuple
        RGBA color.

    Returns
    -------
    NodePath wrapping a GeomNode.
    """
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    col = GeomVertexWriter(vdata, "color")
    tris = GeomTriangles(Geom.UHStatic)

    lc = LColor(*color)

    # Generate vertices
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        for j in range(slices + 1):
            theta = 2 * math.pi * j / slices
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            nx, ny, nz = (
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi),
            )
            vertex.addData3(x, y, z)
            normal.addData3(nx, ny, nz)
            col.addData4(lc)

    # Generate triangles
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            tris.addVertices(a, b, a + 1)
            tris.addVertices(a + 1, b, b + 1)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def make_cube(
    name: str = "cube",
    size: float = 1.0,
    color: tuple = (0.8, 0.8, 0.8, 1.0),
) -> NodePath:
    """Create a cube mesh with face normals.

    Parameters
    ----------
    name : str
        Node name.
    size : float
        Half-extent of the cube.
    color : tuple
        RGBA color.

    Returns
    -------
    NodePath wrapping a GeomNode.
    """
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    col = GeomVertexWriter(vdata, "color")
    tris = GeomTriangles(Geom.UHStatic)

    lc = LColor(*color)
    s = size

    # 6 faces, 4 vertices each, with correct face normals
    faces = [
        # (normal, vertices)
        ((0, 0, 1), [(-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]),
        ((0, 0, -1), [(-s, s, -s), (s, s, -s), (s, -s, -s), (-s, -s, -s)]),
        ((0, 1, 0), [(-s, s, -s), (-s, s, s), (s, s, s), (s, s, -s)]),
        ((0, -1, 0), [(-s, -s, s), (-s, -s, -s), (s, -s, -s), (s, -s, s)]),
        ((1, 0, 0), [(s, -s, s), (s, -s, -s), (s, s, -s), (s, s, s)]),
        ((-1, 0, 0), [(-s, -s, -s), (-s, -s, s), (-s, s, s), (-s, s, -s)]),
    ]

    idx = 0
    for n, verts in faces:
        for v in verts:
            vertex.addData3(*v)
            normal.addData3(*n)
            col.addData4(lc)
        tris.addVertices(idx, idx + 1, idx + 2)
        tris.addVertices(idx, idx + 2, idx + 3)
        idx += 4

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def make_cone(
    name: str = "cone",
    radius: float = 1.0,
    height: float = 2.0,
    slices: int = 32,
    color: tuple = (0.8, 0.8, 0.8, 1.0),
) -> NodePath:
    """Create a cone mesh.

    The cone apex is at (0, 0, height) and the base center at origin.
    """
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal_w = GeomVertexWriter(vdata, "normal")
    col = GeomVertexWriter(vdata, "color")
    tris = GeomTriangles(Geom.UHStatic)

    lc = LColor(*color)
    slope = radius / height

    # Apex vertex
    vertex.addData3(0, 0, height)
    normal_w.addData3(0, 0, 1)
    col.addData4(lc)
    apex_idx = 0

    # Base ring
    for i in range(slices + 1):
        theta = 2 * math.pi * i / slices
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        # Normal for cone side
        nx = math.cos(theta)
        ny = math.sin(theta)
        nz = slope
        ln = math.sqrt(nx * nx + ny * ny + nz * nz)
        vertex.addData3(x, y, 0)
        normal_w.addData3(nx / ln, ny / ln, nz / ln)
        col.addData4(lc)

    # Side triangles
    for i in range(slices):
        tris.addVertices(apex_idx, i + 1, i + 2)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def make_cylinder(
    name: str = "cylinder",
    radius: float = 1.0,
    height: float = 2.0,
    slices: int = 32,
    color: tuple = (0.8, 0.8, 0.8, 1.0),
) -> NodePath:
    """Create a cylinder mesh along the Z axis."""
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal_w = GeomVertexWriter(vdata, "normal")
    col = GeomVertexWriter(vdata, "color")
    tris = GeomTriangles(Geom.UHStatic)

    lc = LColor(*color)

    # Two rings (bottom and top)
    for ring in range(2):
        z = ring * height
        for i in range(slices + 1):
            theta = 2 * math.pi * i / slices
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            vertex.addData3(x, y, z)
            normal_w.addData3(math.cos(theta), math.sin(theta), 0)
            col.addData4(lc)

    # Side quads
    for i in range(slices):
        a = i
        b = i + 1
        c = i + slices + 1
        d = i + slices + 2
        tris.addVertices(a, c, b)
        tris.addVertices(b, c, d)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


# Registry of primitive shape generators
PRIMITIVE_GENERATORS = {
    "sphere": make_sphere,
    "cube": make_cube,
    "cone": make_cone,
    "cylinder": make_cylinder,
    # Habitat compatibility names
    "coneSolid": make_cone,
    "cubeSolid": make_cube,
    "sphereSolid": make_sphere,
    "cylinderSolid": make_cylinder,
}
