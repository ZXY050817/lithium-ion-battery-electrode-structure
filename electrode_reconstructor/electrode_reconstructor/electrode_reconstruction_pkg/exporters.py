from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from xml.sax.saxutils import escape as xml_escape
import zipfile

import numpy as np

from .particles import particle_size_rank_value
from .shared import (
    DEFAULT_PARTICLE_HISTOGRAM_BIN_WIDTH_UM,
    EPS,
    Float3,
    Int3,
    ParticleRecord,
    PHASE_CBD,
    PHASE_ELECTROLYTE,
    STEP_FACE_SPECS,
)

def excel_column_name(index: int) -> str:
    if index < 0:
        raise ValueError("Excel column index must be non-negative.")
    letters: List[str] = []
    column_number = index + 1
    while column_number > 0:
        column_number, remainder = divmod(column_number - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))

def excel_cell_xml(row_number: int, column_index: int, value: object) -> str:
    cell_ref = f"{excel_column_name(column_index)}{row_number}"
    if value == "":
        return ""
    if isinstance(value, bool):
        return f'<c r="{cell_ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, int):
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    if isinstance(value, float):
        if math.isfinite(value):
            return f'<c r="{cell_ref}"><v>{value:.15g}</v></c>'
        value = str(value)
    text = xml_escape(str(value))
    return (
        f'<c r="{cell_ref}" t="inlineStr">'
        f'<is><t xml:space="preserve">{text}</t></is>'
        f"</c>"
    )

def write_particle_records_excel(records: Iterable[ParticleRecord], path: Path) -> None:
    headers: List[object] = [
        "particle_id",
        "scale_factor",
        "semi_axis_a_um",
        "semi_axis_b_um",
        "semi_axis_c_um",
        "semi_axis_a_vox",
        "semi_axis_b_vox",
        "semi_axis_c_vox",
        "rot_x_deg",
        "rot_y_deg",
        "rot_z_deg",
        "analytic_volume_um3",
        "template_voxels",
        "placed",
        "center_x_vox",
        "center_y_vox",
        "center_z_vox",
        "attempts",
        "overlap_ratio",
    ]

    rows: List[List[object]] = [headers]
    for record in records:
        center_x: object = ""
        center_y: object = ""
        center_z: object = ""
        if record.center_vox is not None:
            center_x, center_y, center_z = record.center_vox

        rows.append(
            [
                record.particle_id,
                record.scale_factor,
                record.semi_axes_um[0],
                record.semi_axes_um[1],
                record.semi_axes_um[2],
                record.semi_axes_vox[0],
                record.semi_axes_vox[1],
                record.semi_axes_vox[2],
                record.angles_deg[0],
                record.angles_deg[1],
                record.angles_deg[2],
                record.analytic_volume_um3,
                record.template_voxels,
                int(record.placed),
                center_x,
                center_y,
                center_z,
                record.attempts,
                record.overlap_ratio,
            ]
        )

    last_column = excel_column_name(len(headers) - 1)
    dimension_ref = f"A1:{last_column}{len(rows)}"
    sheet_rows: List[str] = []
    for row_index, row in enumerate(rows, start=1):
        cell_xml = "".join(
            excel_cell_xml(row_index, column_index, value)
            for column_index, value in enumerate(row)
        )
        sheet_rows.append(f'<row r="{row_index}">{cell_xml}</row>')

    worksheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension_ref}"/>'
        "<sheetData>"
        f"{''.join(sheet_rows)}"
        "</sheetData>"
        "</worksheet>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Particles" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        "</Relationships>"
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        "</Types>"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", content_types_xml)
        workbook.writestr("_rels/.rels", root_rels_xml)
        workbook.writestr("xl/workbook.xml", workbook_xml)
        workbook.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        workbook.writestr("xl/styles.xml", styles_xml)
        workbook.writestr("xl/worksheets/sheet1.xml", worksheet_xml)

def write_particle_size_histogram_svg(
    records: Sequence[ParticleRecord],
    path: Path,
    bin_width_um: float = DEFAULT_PARTICLE_HISTOGRAM_BIN_WIDTH_UM,
) -> None:
    placed_diameters = [
        particle_size_rank_value(record)
        for record in records
        if record.placed
    ]
    if not placed_diameters:
        return

    values = np.asarray(placed_diameters, dtype=np.float64)
    bin_width = max(EPS, float(bin_width_um))
    value_min = 0.0
    value_max = float(np.max(values))
    value_max = max(bin_width, math.ceil(value_max / bin_width) * bin_width)
    edges = np.arange(value_min, value_max + bin_width * 0.5, bin_width)
    if edges.size < 2:
        edges = np.asarray([0.0, bin_width], dtype=np.float64)

    counts, edges = np.histogram(values, bins=edges)
    bin_count = int(len(counts))
    max_count = max(1, int(np.max(counts)))

    width = 920
    height = 560
    margin_left = 86
    margin_right = 34
    margin_top = 52
    margin_bottom = 86
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    bar_gap = 3.0
    bar_width = plot_width / int(bin_count)

    def x_pos(index: int) -> float:
        return margin_left + index * bar_width

    axis_y = margin_top + plot_height
    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial, sans-serif; fill:#222;} .axis{stroke:#222;stroke-width:1.5;} .tick{stroke:#555;stroke-width:1;} .bar{fill:#4c78a8;} .grid{stroke:#e5e5e5;stroke-width:1;}</style>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="20">Placed Particle Size Distribution</text>',
        f'<text x="{width / 2:.1f}" y="50" text-anchor="middle" font-size="13">major-axis diameter, bin width={bin_width:.2f} um, n={values.size}</text>',
    ]

    for tick in range(0, 6):
        count_value = max_count * tick / 5.0
        y = axis_y - plot_height * tick / 5.0
        svg_parts.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}"/>')
        svg_parts.append(f'<line class="tick" x1="{margin_left - 5}" y1="{y:.1f}" x2="{margin_left}" y2="{y:.1f}"/>')
        svg_parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{count_value:.0f}</text>')

    for index, count in enumerate(counts):
        bar_height = plot_height * float(count) / float(max_count)
        x = x_pos(index) + bar_gap / 2.0
        y = axis_y - bar_height
        svg_parts.append(
            f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" '
            f'width="{max(1.0, bar_width - bar_gap):.1f}" height="{bar_height:.1f}"/>'
        )

    tick_count = min(10, max(2, int(math.floor(value_max / bin_width)) + 1))
    for tick in range(tick_count):
        value = value_max * tick / float(max(1, tick_count - 1))
        x = margin_left + plot_width * value / max(EPS, value_max)
        svg_parts.append(f'<line class="tick" x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" y2="{axis_y + 5}"/>')
        svg_parts.append(f'<text x="{x:.1f}" y="{axis_y + 24}" text-anchor="middle" font-size="12">{value:.2f}</text>')

    svg_parts.extend(
        [
            f'<line class="axis" x1="{margin_left}" y1="{axis_y}" x2="{width - margin_right}" y2="{axis_y}"/>',
            f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{axis_y}"/>',
            f'<text x="{width / 2:.1f}" y="{height - 26}" text-anchor="middle" font-size="15">Particle major-axis diameter (um)</text>',
            f'<text x="24" y="{margin_top + plot_height / 2:.1f}" transform="rotate(-90 24 {margin_top + plot_height / 2:.1f})" text-anchor="middle" font-size="15">Count</text>',
            f'<text x="{width - margin_right}" y="{height - 8}" text-anchor="end" font-size="11">mean={float(np.mean(values)):.3g} um, std={float(np.std(values, ddof=1)) if values.size > 1 else 0.0:.3g} um</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(svg_parts), encoding="utf-8")

def write_phase_slices_txt(phase_labels: np.ndarray, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for z_index in range(phase_labels.shape[2]):
            slice_xy = phase_labels[:, :, z_index]
            for x_index in range(slice_xy.shape[0]):
                row = " ".join(str(int(value)) for value in slice_xy[x_index, :])
                handle.write(row)
                handle.write("\n")
            if z_index + 1 < phase_labels.shape[2]:
                handle.write("\n")

def encode_phase_labels_three_channel(phase_labels: np.ndarray) -> np.ndarray:
    if phase_labels.ndim != 3:
        raise ValueError("phase_labels must be a 3D array.")
    if np.any((phase_labels < PHASE_ELECTROLYTE) | (phase_labels > PHASE_CBD)):
        raise ValueError("phase_labels contains unsupported phase values.")

    # Per-voxel one-hot vectors: electrolyte -> (1,0,0), active -> (0,1,0), CBD -> (0,0,1).
    return np.eye(3, dtype=np.uint8)[phase_labels]

def write_ascii_triangle(
    handle, normal: Float3, v0: Float3, v1: Float3, v2: Float3
) -> None:
    handle.write(
        f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n"
    )
    handle.write("    outer loop\n")
    handle.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
    handle.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
    handle.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
    handle.write("    endloop\n")
    handle.write("  endfacet\n")

def write_ascii_stl_from_voxels(
    occupied: np.ndarray,
    voxel_size_um: float,
    output_path: Path,
    solid_name: str = "active_particle_phase",
) -> None:
    face_specs = [
        ((1.0, 0.0, 0.0), (1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        ((-1.0, 0.0, 0.0), (-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
        ((0.0, 1.0, 0.0), (0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
        ((0.0, -1.0, 0.0), (0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        ((0.0, 0.0, 1.0), (0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
        ((0.0, 0.0, -1.0), (0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
    ]

    shape = occupied.shape
    with output_path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"solid {solid_name}\n")
        for x, y, z in np.argwhere(occupied):
            x0 = float(x) * voxel_size_um
            y0 = float(y) * voxel_size_um
            z0 = float(z) * voxel_size_um

            for normal, direction, quad in face_specs:
                nx = int(x) + direction[0]
                ny = int(y) + direction[1]
                nz = int(z) + direction[2]
                if (
                    0 <= nx < shape[0]
                    and 0 <= ny < shape[1]
                    and 0 <= nz < shape[2]
                    and occupied[nx, ny, nz]
                ):
                    continue

                vertices: List[Float3] = []
                for vx, vy, vz in quad:
                    vertices.append(
                        (
                            x0 + float(vx) * voxel_size_um,
                            y0 + float(vy) * voxel_size_um,
                            z0 + float(vz) * voxel_size_um,
                        )
                    )
                write_ascii_triangle(
                    handle, normal, vertices[0], vertices[1], vertices[2]
                )
                write_ascii_triangle(
                    handle, normal, vertices[0], vertices[2], vertices[3]
                )
        handle.write(f"endsolid {solid_name}\n")

def format_step_real(value: float) -> str:
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    if text == "-0.0":
        return "0.0"
    return text

def step_string_literal(value: str) -> str:
    return value.replace("'", "''")

def collect_exposed_voxel_quads(
    occupied: np.ndarray,
) -> Tuple[List[Int3], List[Tuple[int, int, int, int]]]:
    shape = occupied.shape
    vertex_map: Dict[Int3, int] = {}
    vertices: List[Int3] = []
    faces: List[Tuple[int, int, int, int]] = []

    for x, y, z in np.argwhere(occupied):
        for direction, quad in STEP_FACE_SPECS:
            nx = int(x) + direction[0]
            ny = int(y) + direction[1]
            nz = int(z) + direction[2]
            if (
                0 <= nx < shape[0]
                and 0 <= ny < shape[1]
                and 0 <= nz < shape[2]
                and occupied[nx, ny, nz]
            ):
                continue

            face_vertex_ids: List[int] = []
            for vx, vy, vz in quad:
                key = (int(x) + vx, int(y) + vy, int(z) + vz)
                vertex_id = vertex_map.get(key)
                if vertex_id is None:
                    vertex_id = len(vertices)
                    vertex_map[key] = vertex_id
                    vertices.append(key)
                face_vertex_ids.append(vertex_id)
            if len(face_vertex_ids) != 4:
                raise RuntimeError("STEP face extraction produced a non-quad face.")
            face_quad: Tuple[int, int, int, int] = (
                face_vertex_ids[0],
                face_vertex_ids[1],
                face_vertex_ids[2],
                face_vertex_ids[3],
            )
            faces.append(face_quad)

    return vertices, faces

class StepEntityWriter:
    def __init__(self) -> None:
        self.entities: List[str] = []

    def add(self, entity: str) -> int:
        self.entities.append(entity)
        return len(self.entities)

    @staticmethod
    def ref(entity_id: int) -> str:
        return f"#{entity_id}"

def write_faceted_step_from_voxels(
    occupied: np.ndarray, voxel_size_um: float, output_path: Path
) -> Tuple[int, int]:
    vertices_lattice, faces = collect_exposed_voxel_quads(occupied)
    if not faces:
        raise ValueError("No active-particle surface was found for STEP export.")

    writer = StepEntityWriter()
    part_name = step_string_literal(output_path.stem)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    uncertainty_value = max(voxel_size_um * 0.5, EPS)

    app_context = writer.add(
        "APPLICATION_CONTEXT('automotive_design')"
    )
    writer.add(
        "APPLICATION_PROTOCOL_DEFINITION('international standard',"
        f"'automotive_design',2001,{writer.ref(app_context)})"
    )
    product_context = writer.add(
        f"PRODUCT_CONTEXT('',{writer.ref(app_context)},'mechanical')"
    )
    product_definition_context = writer.add(
        f"PRODUCT_DEFINITION_CONTEXT('part definition',{writer.ref(app_context)},'design')"
    )
    product = writer.add(
        f"PRODUCT('{part_name}','{part_name}','',({writer.ref(product_context)}))"
    )
    product_definition_formation = writer.add(
        "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE('1','',"
        f"{writer.ref(product)},.NOT_KNOWN.)"
    )
    product_definition = writer.add(
        f"PRODUCT_DEFINITION('design','',{writer.ref(product_definition_formation)},"
        f"{writer.ref(product_definition_context)})"
    )
    product_definition_shape = writer.add(
        f"PRODUCT_DEFINITION_SHAPE('','',{writer.ref(product_definition)})"
    )

    length_unit = writer.add(
        "(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MICRO.,.METRE.))"
    )
    plane_angle_unit = writer.add(
        "(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))"
    )
    solid_angle_unit = writer.add(
        "(NAMED_UNIT(*) SOLID_ANGLE_UNIT() SI_UNIT($,.STERADIAN.))"
    )
    uncertainty = writer.add(
        "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE("
        f"{format_step_real(uncertainty_value)}),{writer.ref(length_unit)},"
        "'distance_accuracy_value','confusion accuracy')"
    )
    representation_context = writer.add(
        "(GEOMETRIC_REPRESENTATION_CONTEXT(3) "
        f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({writer.ref(uncertainty)})) "
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({writer.ref(length_unit)},"
        f"{writer.ref(plane_angle_unit)},{writer.ref(solid_angle_unit)})) "
        "REPRESENTATION_CONTEXT('Context #1','3D context with units'))"
    )

    point_ids: Dict[Int3, int] = {}
    face_ids: List[int] = []

    def get_point_id(coord: Int3) -> int:
        point_id = point_ids.get(coord)
        if point_id is None:
            point_id = writer.add(
                "CARTESIAN_POINT('',("
                f"{format_step_real(coord[0] * voxel_size_um)},"
                f"{format_step_real(coord[1] * voxel_size_um)},"
                f"{format_step_real(coord[2] * voxel_size_um)}))"
            )
            point_ids[coord] = point_id
        return point_id

    for face in faces:
        face_coords = [vertices_lattice[index] for index in face]
        poly_loop_points = ",".join(
            writer.ref(get_point_id(coord)) for coord in face_coords
        )
        poly_loop_id = writer.add(f"POLY_LOOP('',({poly_loop_points}))")
        face_bound_id = writer.add(
            f"FACE_OUTER_BOUND('',{writer.ref(poly_loop_id)},.T.)"
        )
        face_ids.append(
            writer.add(
                f"FACE('',({writer.ref(face_bound_id)}))"
            )
        )

    closed_shell_id = writer.add(
        "CLOSED_SHELL('',(" + ",".join(writer.ref(face_id) for face_id in face_ids) + "))"
    )
    solid_brep_id = writer.add(
        f"FACETED_BREP('{part_name}',{writer.ref(closed_shell_id)})"
    )
    shape_representation_id = writer.add(
        "SHAPE_REPRESENTATION('',("
        f"{writer.ref(solid_brep_id)}),{writer.ref(representation_context)})"
    )
    writer.add(
        "SHAPE_DEFINITION_REPRESENTATION("
        f"{writer.ref(product_definition_shape)},{writer.ref(shape_representation_id)})"
    )

    with output_path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ISO-10303-21;\n")
        handle.write("HEADER;\n")
        handle.write("FILE_DESCRIPTION(('Voxelized active particle phase'),'2;1');\n")
        handle.write(
            "FILE_NAME("
            f"'{part_name}.step','{timestamp}',('Codex'),('OpenAI'),"
            "'Codex voxel STEP exporter','Codex','');\n"
        )
        handle.write("FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n")
        handle.write("ENDSEC;\n")
        handle.write("DATA;\n")
        for index, entity in enumerate(writer.entities, start=1):
            handle.write(f"#{index}={entity};\n")
        handle.write("ENDSEC;\n")
        handle.write("END-ISO-10303-21;\n")

    return len(vertices_lattice), len(faces)
