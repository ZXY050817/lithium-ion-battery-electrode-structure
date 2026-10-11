#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
import re
from typing import List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape as xml_escape
import zipfile

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires numpy. Install it with: pip install numpy"
    ) from exc


DEFAULT_SOLID_VALUE = 1
DEFAULT_LIQUID_VALUES = (0, 2)
DEFAULT_UNKNOWN_CLUSTER_THRESHOLD = 0.2
NEIGHBOR_OFFSETS_26 = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
)


@dataclass
class SpecificSurfaceAreaResult:
    shape_xyz: Tuple[int, int, int]
    voxel_size_um: float
    solid_voxel_count: int
    interface_face_count: int
    total_interface_area_um2: float
    solid_volume_um3: float
    specific_surface_area_inv_um: float


@dataclass
class ClusterInfo:
    volume_voxels: int
    touches_boundary: bool


@dataclass
class AgglomerationResult:
    particle_cluster_count: int
    total_particle_volume_voxels: int
    largest_cluster_volume_voxels: int
    largest_cluster_touches_boundary: bool
    unknown_cluster_count: int
    unknown_cluster_volume_voxels: int
    internal_cluster_count: int
    known_cluster_volume_voxels: int
    unknown_cluster_volume_fraction: float
    unknown_cluster_threshold: float
    representative_sample: bool
    normalized_average_cluster_volume: Optional[float]


@dataclass
class ElectronicConnectivityResult:
    connected_particle_voxel_count: int
    total_particle_voxel_count: int
    electronic_connectivity: float


@dataclass
class ElectrolyteConnectivityResult:
    connected_electrolyte_voxel_count: int
    total_electrolyte_voxel_count: int
    electrolyte_connectivity: float
    percolating: bool


@dataclass
class EquivalentParticleDiameterResult:
    average_cluster_volume_voxels: float
    average_cluster_volume_um3: float
    equivalent_particle_diameter_um: float


@dataclass
class TortuosityResult:
    shortest_path_length_um: float
    straight_distance_um: float
    tortuosity_tau: float


@dataclass
class PhaseVolumeFractionResult:
    total_voxel_count: int
    electrolyte_voxel_count: int
    particle_voxel_count: int
    cbd_voxel_count: int
    electrolyte_volume_fraction: float
    particle_volume_fraction: float
    cbd_volume_fraction: float


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_ROOT = DEFAULT_PROJECT_ROOT / "Structure_Data"
STRUCTURE_FACTOR_JSON_SUFFIX = "_structure_factors.json"
MULTITASK_FACTOR_NAMES: Tuple[str, ...] = (
    "particle_volume_fraction",
    "cbd_volume_fraction",
    "specific_surface_area_inv_um",
    "electronic_connectivity",
    "electrolyte_connectivity",
    "equivalent_particle_diameter_um",
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
        return f'<c r="{cell_ref}"><v>{value:.15g}</v></c>'
    text = xml_escape(str(value))
    return (
        f'<c r="{cell_ref}" t="inlineStr">'
        f'<is><t xml:space="preserve">{text}</t></is>'
        f"</c>"
    )


def write_rows_to_excel(rows: Sequence[Sequence[object]], sheet_name: str, path: Path) -> None:
    if not rows:
        raise ValueError("Excel output rows must not be empty.")
    max_columns = max(len(row) for row in rows)
    last_column = excel_column_name(max_columns - 1)
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
        f'<sheets><sheet name="{xml_escape(sheet_name)}" sheetId="1" r:id="rId1"/></sheets>'
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


def default_output_xlsx_path(input_txt: Path) -> Path:
    parent_name = input_txt.parent.name
    if re.fullmatch(r"\d{8}_\d+", parent_name):
        return input_txt.parent / f"structure_factor_{parent_name}.xlsx"
    return input_txt.with_name(f"structure_factor_{input_txt.stem}.xlsx")


def default_output_json_path(input_txt: Path) -> Path:
    input_name = input_txt.name
    if input_name.endswith("_phase_slices.txt"):
        output_name = input_name.replace(
            "_phase_slices.txt",
            STRUCTURE_FACTOR_JSON_SUFFIX,
        )
        return input_txt.with_name(output_name)
    return input_txt.with_name(f"{input_txt.stem}{STRUCTURE_FACTOR_JSON_SUFFIX}")


def build_multitask_structure_factor_payload(
    phase_volume_fractions: PhaseVolumeFractionResult,
    result: SpecificSurfaceAreaResult,
    electronic_connectivity: ElectronicConnectivityResult,
    electrolyte_connectivity: ElectrolyteConnectivityResult,
    equivalent_particle_diameter: EquivalentParticleDiameterResult,
) -> dict[str, float]:
    payload: dict[str, float] = {
        "particle_volume_fraction": float(phase_volume_fractions.particle_volume_fraction),
        "cbd_volume_fraction": float(phase_volume_fractions.cbd_volume_fraction),
        "specific_surface_area_inv_um": float(result.specific_surface_area_inv_um),
        "electronic_connectivity": float(electronic_connectivity.electronic_connectivity),
        "electrolyte_connectivity": float(electrolyte_connectivity.electrolyte_connectivity),
        "equivalent_particle_diameter_um": float(
            equivalent_particle_diameter.equivalent_particle_diameter_um
        ),
    }
    for factor_name in MULTITASK_FACTOR_NAMES:
        factor_value = payload[factor_name]
        if not math.isfinite(factor_value):
            raise ValueError(
                f"Structure-factor key '{factor_name}' is not finite and cannot be written to JSON."
            )
    return payload


def write_multitask_structure_factor_json(output_json: Path, payload: dict[str, float]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def structure_display_name(input_txt: Path) -> str:
    parent_name = input_txt.parent.name
    if re.fullmatch(r"\d{8}_\d+", parent_name):
        return parent_name
    return input_txt.stem


def write_specific_surface_area_excel(
    input_txt: Path,
    output_xlsx: Path,
    result: SpecificSurfaceAreaResult,
    phase_volume_fractions: PhaseVolumeFractionResult,
    agglomeration: AgglomerationResult,
    electronic_connectivity: ElectronicConnectivityResult,
    electrolyte_connectivity: ElectrolyteConnectivityResult,
    equivalent_particle_diameter: EquivalentParticleDiameterResult,
    tortuosity: TortuosityResult,
    solid_value: int,
    liquid_values: Sequence[int],
) -> None:
    normalized_cluster_volume: object
    if agglomeration.normalized_average_cluster_volume is None:
        normalized_cluster_volume = "不适用 / N/A"
    else:
        normalized_cluster_volume = agglomeration.normalized_average_cluster_volume
    tortuosity_value: object
    if math.isinf(tortuosity.tortuosity_tau):
        tortuosity_value = "inf"
    else:
        tortuosity_value = tortuosity.tortuosity_tau

    rows: List[List[object]] = [
        ["名称 / Name", "数值 / Value"],
        ["输入结构 / Input Structure", structure_display_name(input_txt)],
        [
            "电解液相体积分数 / Electrolyte Volume Fraction",
            phase_volume_fractions.electrolyte_volume_fraction,
        ],
        [
            "颗粒相体积分数 / Particle Volume Fraction",
            phase_volume_fractions.particle_volume_fraction,
        ],
        [
            "CBD相体积分数 / CBD Volume Fraction",
            phase_volume_fractions.cbd_volume_fraction,
        ],
        ["固体体素数量 / Solid Voxel Count", result.solid_voxel_count],
        ["总界面面积 (um^2) / Total Interfacial Area (um^2)", result.total_interface_area_um2],
        ["总固体体积 (um^3) / Total Solid Volume (um^3)", result.solid_volume_um3],
        ["比表面积 (1/um) / Specific Surface Area (1/um)", result.specific_surface_area_inv_um],
        ["总集群数量 / Total Cluster Count", agglomeration.particle_cluster_count],
        [
            "最大颗粒集群体积 (voxel) / Largest Particle Cluster Volume (voxel)",
            agglomeration.largest_cluster_volume_voxels,
        ],
        ["未知颗粒集群数量 / Unknown Particle Cluster Count", agglomeration.unknown_cluster_count],
        [
            "未知颗粒集群总体积 (voxel) / Total Unknown Particle Cluster Volume (voxel)",
            agglomeration.unknown_cluster_volume_voxels,
        ],
        ["未知颗粒体积分数 / Unknown Particle Volume Fraction", agglomeration.unknown_cluster_volume_fraction],
        [
            "归一化平均团簇体积 / Normalized Average Cluster Volume",
            normalized_cluster_volume,
        ],
        [
            "电子连接度 / Electronic Connectivity",
            electronic_connectivity.electronic_connectivity,
        ],
        [
            "电解液相连通率 / Electrolyte Connectivity",
            electrolyte_connectivity.electrolyte_connectivity,
        ],
        [
            "是否贯通 / Percolating",
            "是 / Yes" if electrolyte_connectivity.percolating else "否 / No",
        ],
        [
            "曲折度τ / Tortuosity τ",
            tortuosity_value,
        ],
        [
            "等效当量颗粒粒径 (um) / Equivalent Particle Diameter (um)",
            equivalent_particle_diameter.equivalent_particle_diameter_um,
        ],
    ]
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    write_rows_to_excel(rows, "SpecificSurfaceArea", output_xlsx)


def parse_slice_block(block_text: str, slice_index: int) -> np.ndarray:
    rows: List[List[int]] = []
    expected_width: int | None = None

    for row_index, raw_line in enumerate(block_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        values = line.split()
        if expected_width is None:
            expected_width = len(values)
        elif len(values) != expected_width:
            raise ValueError(
                f"Slice {slice_index} has inconsistent row widths at row {row_index}."
            )

        try:
            rows.append([int(value) for value in values])
        except ValueError as exc:
            raise ValueError(
                f"Slice {slice_index} row {row_index} contains a non-integer value."
            ) from exc

    if not rows:
        raise ValueError(f"Slice {slice_index} is empty.")

    return np.asarray(rows, dtype=np.int8)


def load_phase_slices_txt(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8-sig")
    blocks = [block for block in text.split("\n\n") if block.strip()]
    if not blocks:
        raise ValueError("The phase-slice text file is empty.")

    slices = [
        parse_slice_block(block_text, slice_index)
        for slice_index, block_text in enumerate(blocks, start=1)
    ]
    reference_shape = slices[0].shape
    for slice_index, slice_xy in enumerate(slices[1:], start=2):
        if slice_xy.shape != reference_shape:
            raise ValueError(
                f"Slice {slice_index} shape {slice_xy.shape} does not match "
                f"the first slice shape {reference_shape}."
            )

    return np.stack(slices, axis=2)


def count_solid_liquid_interface_faces(
    phase_labels: np.ndarray,
    solid_value: int,
    liquid_values: Sequence[int],
) -> int:
    liquid_values_array = np.asarray(tuple(liquid_values), dtype=phase_labels.dtype)
    interface_face_count = 0

    for axis in range(3):
        left_slices = [slice(None), slice(None), slice(None)]
        right_slices = [slice(None), slice(None), slice(None)]
        left_slices[axis] = slice(0, -1)
        right_slices[axis] = slice(1, None)

        left = phase_labels[tuple(left_slices)]
        right = phase_labels[tuple(right_slices)]

        left_is_solid = left == solid_value
        right_is_solid = right == solid_value
        left_is_liquid = np.isin(left, liquid_values_array)
        right_is_liquid = np.isin(right, liquid_values_array)

        interface_mask = (left_is_solid & right_is_liquid) | (
            right_is_solid & left_is_liquid
        )
        interface_face_count += int(np.count_nonzero(interface_mask))

    return interface_face_count


def label_particle_clusters_26(
    particle_mask: np.ndarray,
) -> List[ClusterInfo]:
    shape = particle_mask.shape
    visited = np.zeros(shape, dtype=bool)
    clusters: List[ClusterInfo] = []

    particle_coords = np.argwhere(particle_mask)
    for coord in particle_coords:
        start = (int(coord[0]), int(coord[1]), int(coord[2]))
        if visited[start]:
            continue

        stack: List[Tuple[int, int, int]] = [start]
        visited[start] = True
        volume_voxels = 0
        touches_boundary = False

        while stack:
            x, y, z = stack.pop()
            volume_voxels += 1
            if (
                x == 0
                or x == shape[0] - 1
                or y == 0
                or y == shape[1] - 1
                or z == 0
                or z == shape[2] - 1
            ):
                touches_boundary = True

            for dx, dy, dz in NEIGHBOR_OFFSETS_26:
                nx = x + dx
                ny = y + dy
                nz = z + dz
                if (
                    0 <= nx < shape[0]
                    and 0 <= ny < shape[1]
                    and 0 <= nz < shape[2]
                    and particle_mask[nx, ny, nz]
                    and not visited[nx, ny, nz]
                ):
                    visited[nx, ny, nz] = True
                    stack.append((nx, ny, nz))

        clusters.append(
            ClusterInfo(
                volume_voxels=volume_voxels,
                touches_boundary=touches_boundary,
            )
        )

    return clusters


def compute_particle_agglomeration(
    phase_labels: np.ndarray,
    particle_value: int = DEFAULT_SOLID_VALUE,
    unknown_cluster_threshold: float = DEFAULT_UNKNOWN_CLUSTER_THRESHOLD,
) -> AgglomerationResult:
    if not (0.0 <= unknown_cluster_threshold <= 1.0):
        raise ValueError("unknown_cluster_threshold must be within [0, 1].")

    particle_mask = phase_labels == particle_value
    clusters = label_particle_clusters_26(particle_mask)
    if not clusters:
        raise ValueError("No particle clusters were found in the reconstructed structure.")

    largest_cluster_index, largest_cluster = max(
        enumerate(clusters),
        key=lambda item: item[1].volume_voxels,
    )
    remaining_clusters = [
        cluster
        for cluster_index, cluster in enumerate(clusters)
        if cluster_index != largest_cluster_index
    ]
    unknown_clusters = [
        cluster for cluster in remaining_clusters if cluster.touches_boundary
    ]
    internal_clusters = [
        cluster for cluster in remaining_clusters if not cluster.touches_boundary
    ]

    total_particle_volume_voxels = sum(
        cluster.volume_voxels for cluster in clusters
    )
    particle_cluster_count = len(clusters)
    unknown_cluster_count = len(unknown_clusters)
    unknown_cluster_volume_voxels = sum(
        cluster.volume_voxels for cluster in unknown_clusters
    )
    internal_cluster_count = len(internal_clusters)
    known_cluster_volume_voxels = total_particle_volume_voxels - unknown_cluster_volume_voxels
    unknown_cluster_volume_fraction = (
        unknown_cluster_volume_voxels / total_particle_volume_voxels
    )
    representative_sample = (
        unknown_cluster_volume_fraction <= unknown_cluster_threshold
    )

    normalized_average_cluster_volume: Optional[float] = None
    if representative_sample and largest_cluster.volume_voxels > 0:
        average_known_cluster_volume = known_cluster_volume_voxels / (
            internal_cluster_count + 1
        )
        normalized_average_cluster_volume = (
            average_known_cluster_volume / largest_cluster.volume_voxels
        )

    return AgglomerationResult(
        particle_cluster_count=particle_cluster_count,
        total_particle_volume_voxels=total_particle_volume_voxels,
        largest_cluster_volume_voxels=largest_cluster.volume_voxels,
        largest_cluster_touches_boundary=largest_cluster.touches_boundary,
        unknown_cluster_count=unknown_cluster_count,
        unknown_cluster_volume_voxels=unknown_cluster_volume_voxels,
        internal_cluster_count=internal_cluster_count,
        known_cluster_volume_voxels=known_cluster_volume_voxels,
        unknown_cluster_volume_fraction=float(unknown_cluster_volume_fraction),
        unknown_cluster_threshold=float(unknown_cluster_threshold),
        representative_sample=representative_sample,
        normalized_average_cluster_volume=normalized_average_cluster_volume,
    )


def compute_phase_volume_fractions(
    phase_labels: np.ndarray,
) -> PhaseVolumeFractionResult:
    total_voxel_count = int(phase_labels.size)
    if total_voxel_count <= 0:
        raise ValueError("phase_labels must contain at least one voxel.")

    electrolyte_voxel_count = int(np.count_nonzero(phase_labels == 0))
    particle_voxel_count = int(np.count_nonzero(phase_labels == 1))
    cbd_voxel_count = int(np.count_nonzero(phase_labels == 2))

    return PhaseVolumeFractionResult(
        total_voxel_count=total_voxel_count,
        electrolyte_voxel_count=electrolyte_voxel_count,
        particle_voxel_count=particle_voxel_count,
        cbd_voxel_count=cbd_voxel_count,
        electrolyte_volume_fraction=float(electrolyte_voxel_count / total_voxel_count),
        particle_volume_fraction=float(particle_voxel_count / total_voxel_count),
        cbd_volume_fraction=float(cbd_voxel_count / total_voxel_count),
    )


def compute_specific_surface_area(
    phase_labels: np.ndarray,
    voxel_size_um: float,
    solid_value: int = DEFAULT_SOLID_VALUE,
    liquid_values: Sequence[int] = DEFAULT_LIQUID_VALUES,
) -> SpecificSurfaceAreaResult:
    if voxel_size_um <= 0.0:
        raise ValueError("voxel_size_um must be positive.")

    solid_voxel_count = int(np.count_nonzero(phase_labels == solid_value))
    if solid_voxel_count <= 0:
        raise ValueError("No solid voxels were found in the reconstructed structure.")

    interface_face_count = count_solid_liquid_interface_faces(
        phase_labels=phase_labels,
        solid_value=solid_value,
        liquid_values=liquid_values,
    )
    face_area_um2 = voxel_size_um ** 2
    total_interface_area_um2 = interface_face_count * face_area_um2
    solid_volume_um3 = solid_voxel_count * (voxel_size_um ** 3)
    specific_surface_area_inv_um = total_interface_area_um2 / solid_volume_um3

    return SpecificSurfaceAreaResult(
        shape_xyz=(
            int(phase_labels.shape[0]),
            int(phase_labels.shape[1]),
            int(phase_labels.shape[2]),
        ),
        voxel_size_um=float(voxel_size_um),
        solid_voxel_count=solid_voxel_count,
        interface_face_count=interface_face_count,
        total_interface_area_um2=float(total_interface_area_um2),
        solid_volume_um3=float(solid_volume_um3),
        specific_surface_area_inv_um=float(specific_surface_area_inv_um),
    )


def compute_electronic_connectivity(
    phase_labels: np.ndarray,
    particle_value: int = DEFAULT_SOLID_VALUE,
    conductive_values: Sequence[int] = (1, 2),
    collector_axis: int = 2,
    collector_index: int = 0,
) -> ElectronicConnectivityResult:
    if collector_axis != 2:
        raise ValueError("This implementation currently supports z-direction collectors only (collector_axis=2).")

    conductive_mask = np.isin(
        phase_labels, np.asarray(tuple(conductive_values), dtype=phase_labels.dtype)
    )
    particle_mask = phase_labels == particle_value
    total_particle_voxel_count = int(np.count_nonzero(particle_mask))
    if total_particle_voxel_count <= 0:
        raise ValueError("No particle voxels were found in the reconstructed structure.")

    shape = phase_labels.shape
    face_mask = conductive_mask[:, :, collector_index]
    seed_count = int(np.count_nonzero(face_mask))
    initial_capacity = max(1024, seed_count)
    stack_x = np.empty(initial_capacity, dtype=np.int32)
    stack_y = np.empty(initial_capacity, dtype=np.int32)
    stack_z = np.empty(initial_capacity, dtype=np.int32)
    stack_size = 0

    connected_particle_voxel_count = 0
    for x in range(shape[0]):
        face_row = face_mask[x]
        for y in range(shape[1]):
            if face_row[y]:
                face_row[y] = False
                if stack_size >= stack_x.shape[0]:
                    new_capacity = stack_x.shape[0] * 2
                    new_stack_x = np.empty(new_capacity, dtype=np.int32)
                    new_stack_y = np.empty(new_capacity, dtype=np.int32)
                    new_stack_z = np.empty(new_capacity, dtype=np.int32)
                    new_stack_x[:stack_size] = stack_x[:stack_size]
                    new_stack_y[:stack_size] = stack_y[:stack_size]
                    new_stack_z[:stack_size] = stack_z[:stack_size]
                    stack_x, stack_y, stack_z = new_stack_x, new_stack_y, new_stack_z
                stack_x[stack_size] = x
                stack_y[stack_size] = y
                stack_z[stack_size] = collector_index
                stack_size += 1
                if particle_mask[x, y, collector_index]:
                    connected_particle_voxel_count += 1

    while stack_size > 0:
        stack_size -= 1
        x = int(stack_x[stack_size])
        y = int(stack_y[stack_size])
        z = int(stack_z[stack_size])
        for dx, dy, dz in NEIGHBOR_OFFSETS_26:
            nx = x + dx
            ny = y + dy
            nz = z + dz
            if (
                0 <= nx < shape[0]
                and 0 <= ny < shape[1]
                and 0 <= nz < shape[2]
                and conductive_mask[nx, ny, nz]
            ):
                conductive_mask[nx, ny, nz] = False
                if stack_size >= stack_x.shape[0]:
                    new_capacity = stack_x.shape[0] * 2
                    new_stack_x = np.empty(new_capacity, dtype=np.int32)
                    new_stack_y = np.empty(new_capacity, dtype=np.int32)
                    new_stack_z = np.empty(new_capacity, dtype=np.int32)
                    new_stack_x[:stack_size] = stack_x[:stack_size]
                    new_stack_y[:stack_size] = stack_y[:stack_size]
                    new_stack_z[:stack_size] = stack_z[:stack_size]
                    stack_x, stack_y, stack_z = new_stack_x, new_stack_y, new_stack_z
                stack_x[stack_size] = nx
                stack_y[stack_size] = ny
                stack_z[stack_size] = nz
                stack_size += 1
                if particle_mask[nx, ny, nz]:
                    connected_particle_voxel_count += 1

    electronic_connectivity = (
        connected_particle_voxel_count / total_particle_voxel_count
    )
    return ElectronicConnectivityResult(
        connected_particle_voxel_count=connected_particle_voxel_count,
        total_particle_voxel_count=total_particle_voxel_count,
        electronic_connectivity=float(electronic_connectivity),
    )


def compute_electrolyte_connectivity(
    phase_labels: np.ndarray,
    electrolyte_value: int = 0,
    inlet_axis: int = 2,
    inlet_index: int = 0,
) -> ElectrolyteConnectivityResult:
    if inlet_axis != 2:
        raise ValueError(
            "This implementation currently supports z-direction inlet faces only (inlet_axis=2)."
        )

    electrolyte_mask = phase_labels == electrolyte_value
    total_electrolyte_voxel_count = int(np.count_nonzero(electrolyte_mask))
    if total_electrolyte_voxel_count <= 0:
        raise ValueError("No electrolyte voxels were found in the reconstructed structure.")

    shape = phase_labels.shape
    face_mask = electrolyte_mask[:, :, inlet_index]
    seed_count = int(np.count_nonzero(face_mask))
    initial_capacity = max(1024, seed_count)
    stack_x = np.empty(initial_capacity, dtype=np.int32)
    stack_y = np.empty(initial_capacity, dtype=np.int32)
    stack_z = np.empty(initial_capacity, dtype=np.int32)
    stack_size = 0

    connected_electrolyte_voxel_count = 0
    for x in range(shape[0]):
        face_row = face_mask[x]
        for y in range(shape[1]):
            if face_row[y]:
                face_row[y] = False
                if stack_size >= stack_x.shape[0]:
                    new_capacity = stack_x.shape[0] * 2
                    new_stack_x = np.empty(new_capacity, dtype=np.int32)
                    new_stack_y = np.empty(new_capacity, dtype=np.int32)
                    new_stack_z = np.empty(new_capacity, dtype=np.int32)
                    new_stack_x[:stack_size] = stack_x[:stack_size]
                    new_stack_y[:stack_size] = stack_y[:stack_size]
                    new_stack_z[:stack_size] = stack_z[:stack_size]
                    stack_x, stack_y, stack_z = new_stack_x, new_stack_y, new_stack_z
                stack_x[stack_size] = x
                stack_y[stack_size] = y
                stack_z[stack_size] = inlet_index
                stack_size += 1
                connected_electrolyte_voxel_count += 1

    outlet_index = shape[2] - 1
    percolating = False
    while stack_size > 0:
        stack_size -= 1
        x = int(stack_x[stack_size])
        y = int(stack_y[stack_size])
        z = int(stack_z[stack_size])
        if z == outlet_index:
            percolating = True
        for dx, dy, dz in NEIGHBOR_OFFSETS_26:
            nx = x + dx
            ny = y + dy
            nz = z + dz
            if (
                0 <= nx < shape[0]
                and 0 <= ny < shape[1]
                and 0 <= nz < shape[2]
                and electrolyte_mask[nx, ny, nz]
            ):
                electrolyte_mask[nx, ny, nz] = False
                if stack_size >= stack_x.shape[0]:
                    new_capacity = stack_x.shape[0] * 2
                    new_stack_x = np.empty(new_capacity, dtype=np.int32)
                    new_stack_y = np.empty(new_capacity, dtype=np.int32)
                    new_stack_z = np.empty(new_capacity, dtype=np.int32)
                    new_stack_x[:stack_size] = stack_x[:stack_size]
                    new_stack_y[:stack_size] = stack_y[:stack_size]
                    new_stack_z[:stack_size] = stack_z[:stack_size]
                    stack_x, stack_y, stack_z = new_stack_x, new_stack_y, new_stack_z
                stack_x[stack_size] = nx
                stack_y[stack_size] = ny
                stack_z[stack_size] = nz
                stack_size += 1
                connected_electrolyte_voxel_count += 1

    electrolyte_connectivity = (
        connected_electrolyte_voxel_count / total_electrolyte_voxel_count
    )
    return ElectrolyteConnectivityResult(
        connected_electrolyte_voxel_count=connected_electrolyte_voxel_count,
        total_electrolyte_voxel_count=total_electrolyte_voxel_count,
        electrolyte_connectivity=float(electrolyte_connectivity),
        percolating=percolating,
    )


def compute_tortuosity(
    phase_labels: np.ndarray,
    voxel_size_um: float,
    electrolyte_connectivity: ElectrolyteConnectivityResult,
    electrolyte_value: int = 0,
    inlet_axis: int = 2,
) -> TortuosityResult:
    if inlet_axis != 2:
        raise ValueError(
            "This implementation currently supports z-direction inlet faces only (inlet_axis=2)."
        )
    if voxel_size_um <= 0.0:
        raise ValueError("voxel_size_um must be positive.")

    straight_distance_um = max(phase_labels.shape[2] - 1, 0) * voxel_size_um
    if not electrolyte_connectivity.percolating:
        return TortuosityResult(
            shortest_path_length_um=math.inf,
            straight_distance_um=float(straight_distance_um),
            tortuosity_tau=math.inf,
        )

    electrolyte_mask = phase_labels == electrolyte_value
    if not np.any(electrolyte_mask):
        raise ValueError("No electrolyte voxels were found in the reconstructed structure.")

    outlet_index = phase_labels.shape[2] - 1
    seed_coords = np.argwhere(electrolyte_mask[:, :, 0])
    if seed_coords.size == 0:
        return TortuosityResult(
            shortest_path_length_um=math.inf,
            straight_distance_um=float(straight_distance_um),
            tortuosity_tau=math.inf,
        )

    if straight_distance_um == 0.0:
        return TortuosityResult(
            shortest_path_length_um=0.0,
            straight_distance_um=0.0,
            tortuosity_tau=1.0,
        )

    neighbor_steps = tuple(
        (
            dx,
            dy,
            dz,
            voxel_size_um * math.sqrt(float(dx * dx + dy * dy + dz * dz)),
        )
        for dx, dy, dz in NEIGHBOR_OFFSETS_26
    )

    distances = np.full(phase_labels.shape, math.inf, dtype=np.float64)
    visited = np.zeros(phase_labels.shape, dtype=bool)
    queue: List[Tuple[float, int, int, int]] = []

    for coord in seed_coords:
        x = int(coord[0])
        y = int(coord[1])
        distances[x, y, 0] = 0.0
        heapq.heappush(queue, (0.0, x, y, 0))

    shape = phase_labels.shape
    while queue:
        current_distance, x, y, z = heapq.heappop(queue)
        if visited[x, y, z]:
            continue
        if current_distance > distances[x, y, z]:
            continue
        visited[x, y, z] = True

        if z == outlet_index:
            return TortuosityResult(
                shortest_path_length_um=float(current_distance),
                straight_distance_um=float(straight_distance_um),
                tortuosity_tau=float(current_distance / straight_distance_um),
            )

        for dx, dy, dz, step_distance in neighbor_steps:
            nx = x + dx
            ny = y + dy
            nz = z + dz
            if (
                0 <= nx < shape[0]
                and 0 <= ny < shape[1]
                and 0 <= nz < shape[2]
                and electrolyte_mask[nx, ny, nz]
                and not visited[nx, ny, nz]
            ):
                candidate_distance = current_distance + step_distance
                if candidate_distance < distances[nx, ny, nz]:
                    distances[nx, ny, nz] = candidate_distance
                    heapq.heappush(queue, (candidate_distance, nx, ny, nz))

    return TortuosityResult(
        shortest_path_length_um=math.inf,
        straight_distance_um=float(straight_distance_um),
        tortuosity_tau=math.inf,
    )


def compute_equivalent_particle_diameter(
    agglomeration: AgglomerationResult,
    voxel_size_um: float,
) -> EquivalentParticleDiameterResult:
    if voxel_size_um <= 0.0:
        raise ValueError("voxel_size_um must be positive.")
    if agglomeration.particle_cluster_count <= 0:
        raise ValueError("particle_cluster_count must be positive.")

    average_cluster_volume_voxels = (
        agglomeration.total_particle_volume_voxels / agglomeration.particle_cluster_count
    )
    average_cluster_volume_um3 = average_cluster_volume_voxels * (voxel_size_um ** 3)
    equivalent_particle_diameter_um = 2.0 * math.pow(
        average_cluster_volume_um3 * 3.0 / (4.0 * math.pi),
        1.0 / 3.0,
    )
    return EquivalentParticleDiameterResult(
        average_cluster_volume_voxels=float(average_cluster_volume_voxels),
        average_cluster_volume_um3=float(average_cluster_volume_um3),
        equivalent_particle_diameter_um=float(equivalent_particle_diameter_um),
    )


def compute_all_structure_factors(
    input_txt: Path,
    voxel_size_um: float,
    solid_value: int,
    liquid_values: Sequence[int],
    unknown_cluster_threshold: float,
) -> Tuple[
    SpecificSurfaceAreaResult,
    PhaseVolumeFractionResult,
    AgglomerationResult,
    ElectronicConnectivityResult,
    ElectrolyteConnectivityResult,
    EquivalentParticleDiameterResult,
    TortuosityResult,
]:
    phase_labels = load_phase_slices_txt(input_txt)
    phase_volume_fractions = compute_phase_volume_fractions(phase_labels)
    result = compute_specific_surface_area(
        phase_labels=phase_labels,
        voxel_size_um=voxel_size_um,
        solid_value=solid_value,
        liquid_values=liquid_values,
    )
    agglomeration = compute_particle_agglomeration(
        phase_labels=phase_labels,
        particle_value=solid_value,
        unknown_cluster_threshold=unknown_cluster_threshold,
    )
    electronic_connectivity = compute_electronic_connectivity(
        phase_labels=phase_labels,
        particle_value=solid_value,
        conductive_values=(solid_value, 2),
    )
    electrolyte_connectivity = compute_electrolyte_connectivity(
        phase_labels=phase_labels,
        electrolyte_value=0,
    )
    tortuosity = compute_tortuosity(
        phase_labels=phase_labels,
        voxel_size_um=voxel_size_um,
        electrolyte_connectivity=electrolyte_connectivity,
        electrolyte_value=0,
    )
    equivalent_particle_diameter = compute_equivalent_particle_diameter(
        agglomeration=agglomeration,
        voxel_size_um=voxel_size_um,
    )
    return (
        result,
        phase_volume_fractions,
        agglomeration,
        electronic_connectivity,
        electrolyte_connectivity,
        equivalent_particle_diameter,
        tortuosity,
    )


def process_single_structure_file(
    input_txt: Path,
    voxel_size_um: float,
    solid_value: int,
    liquid_values: Sequence[int],
    unknown_cluster_threshold: float,
    output_xlsx: Optional[Path] = None,
) -> Tuple[
    Path,
    SpecificSurfaceAreaResult,
    PhaseVolumeFractionResult,
    AgglomerationResult,
    ElectronicConnectivityResult,
    ElectrolyteConnectivityResult,
    EquivalentParticleDiameterResult,
    TortuosityResult,
]:
    (
        result,
        phase_volume_fractions,
        agglomeration,
        electronic_connectivity,
        electrolyte_connectivity,
        equivalent_particle_diameter,
        tortuosity,
    ) = compute_all_structure_factors(
        input_txt=input_txt,
        voxel_size_um=voxel_size_um,
        solid_value=solid_value,
        liquid_values=liquid_values,
        unknown_cluster_threshold=unknown_cluster_threshold,
    )
    resolved_output_xlsx = (
        output_xlsx if output_xlsx is not None else default_output_xlsx_path(input_txt)
    )
    write_specific_surface_area_excel(
        input_txt=input_txt,
        output_xlsx=resolved_output_xlsx,
        result=result,
        phase_volume_fractions=phase_volume_fractions,
        agglomeration=agglomeration,
        electronic_connectivity=electronic_connectivity,
        electrolyte_connectivity=electrolyte_connectivity,
        equivalent_particle_diameter=equivalent_particle_diameter,
        tortuosity=tortuosity,
        solid_value=solid_value,
        liquid_values=liquid_values,
    )
    return (
        resolved_output_xlsx,
        result,
        phase_volume_fractions,
        agglomeration,
        electronic_connectivity,
        electrolyte_connectivity,
        equivalent_particle_diameter,
        tortuosity,
    )


def collect_phase_slice_txt_files(batch_root: Path) -> List[Path]:
    return sorted(
        path for path in batch_root.rglob("*_phase_slices.txt") if path.is_file()
    )


def prompt_text(prompt: str, default: str = "") -> str:
    if default:
        raw = input(f"{prompt} [{default}]: ").strip()
        return raw if raw else default
    return input(f"{prompt}: ").strip()


def configure_interactively(args: argparse.Namespace) -> argparse.Namespace:
    if args.input_txt is not None or args.batch_root is not None:
        return args

    print("Select analysis mode:")
    print("1. single file")
    print("2. batch directory scan")
    while True:
        raw_choice = input("Mode [1]: ").strip().lower()
        choice = raw_choice if raw_choice else "1"
        if choice in {"1", "single", "s"}:
            input_text = prompt_text("Input TXT path")
            if not input_text:
                raise ValueError("Input TXT path is required.")
            args.input_txt = Path(input_text)
            if args.output_xlsx is None:
                output_text = prompt_text(
                    "Output Excel path (blank for default)", ""
                )
                if output_text:
                    args.output_xlsx = Path(output_text)
            return args
        if choice in {"2", "batch", "b"}:
            batch_root_text = prompt_text(
                "Batch root directory",
                str(DEFAULT_BATCH_ROOT),
            )
            if not batch_root_text:
                raise ValueError("Batch root directory is required.")
            args.batch_root = Path(batch_root_text)
            while True:
                batch_output_format_text = prompt_text(
                    "Batch output format (json/xlsx/both)",
                    str(getattr(args, "batch_output_format", "json")),
                ).strip().lower()
                if not batch_output_format_text:
                    batch_output_format_text = "json"
                if batch_output_format_text in {"json", "xlsx", "both"}:
                    args.batch_output_format = batch_output_format_text
                    break
                print("Please enter json, xlsx, or both.")
            return args
        print("Please enter 1 or 2.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a phase-slice TXT file and compute the solid-liquid specific "
            "surface area of the reconstructed electrode."
        )
    )
    parser.add_argument(
        "input_txt",
        type=Path,
        nargs="?",
        help="Path to the phase-slice TXT file produced by electrode_reconstruction.py.",
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        help=(
            "Optional root directory for batch scanning. All *_phase_slices.txt files "
            "under this directory will be processed, and outputs will be written to the "
            "same subdirectory as each TXT file according to --batch-output-format."
        ),
    )
    parser.add_argument(
        "--voxel-size-um",
        type=float,
        default=0.5,
        help="Voxel size h in micrometers. Default: 0.5",
    )
    parser.add_argument(
        "--solid-value",
        type=int,
        default=DEFAULT_SOLID_VALUE,
        help="Phase label treated as solid. Default: 1",
    )
    parser.add_argument(
        "--liquid-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_LIQUID_VALUES),
        help="Phase labels treated as electrolyte/liquid. Default: 0 2",
    )
    parser.add_argument(
        "--unknown-cluster-threshold",
        type=float,
        default=DEFAULT_UNKNOWN_CLUSTER_THRESHOLD,
        help="Maximum allowed unknown-cluster volume fraction for a representative sample. Default: 0.2",
    )
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        help=(
            "Optional Excel output path. If omitted, the workbook is written "
            "to the same directory as the input TXT."
        ),
    )
    parser.add_argument(
        "--batch-output-format",
        choices=("json", "xlsx", "both"),
        default="json",
        help=(
            "Batch mode output format. "
            "'json' writes *_structure_factors.json sidecars for multitask training, "
            "'xlsx' writes Excel only, and 'both' writes both. Default: json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = configure_interactively(parse_args())

    if args.input_txt is not None and args.batch_root is not None:
        raise ValueError("Specify either input_txt or --batch-root, not both.")

    if args.batch_root is not None:
        if args.output_xlsx is not None:
            raise ValueError("--output-xlsx cannot be used together with --batch-root.")
        write_batch_json = args.batch_output_format in {"json", "both"}
        write_batch_excel = args.batch_output_format in {"xlsx", "both"}
        batch_root = args.batch_root
        if not batch_root.exists():
            raise FileNotFoundError(f"Batch root does not exist: {batch_root}")
        txt_files = collect_phase_slice_txt_files(batch_root)
        if not txt_files:
            raise ValueError(
                f"No *_phase_slices.txt files were found under: {batch_root}"
            )

        print(f"Batch root directory: {batch_root}")
        print(f"Batch output format: {args.batch_output_format}")
        print(f"Detected structure TXT files: {len(txt_files)}")
        success_count = 0
        failure_count = 0
        last_reported_percent = 0

        for index, input_txt in enumerate(txt_files, start=1):
            try:
                (
                    result,
                    phase_volume_fractions,
                    agglomeration,
                    electronic_connectivity,
                    electrolyte_connectivity,
                    equivalent_particle_diameter,
                    tortuosity,
                ) = compute_all_structure_factors(
                    input_txt=input_txt,
                    voxel_size_um=args.voxel_size_um,
                    solid_value=args.solid_value,
                    liquid_values=args.liquid_values,
                    unknown_cluster_threshold=args.unknown_cluster_threshold,
                )
                output_xlsx: Optional[Path] = None
                output_json: Optional[Path] = None
                if write_batch_excel:
                    output_xlsx = default_output_xlsx_path(input_txt)
                    write_specific_surface_area_excel(
                        input_txt=input_txt,
                        output_xlsx=output_xlsx,
                        result=result,
                        phase_volume_fractions=phase_volume_fractions,
                        agglomeration=agglomeration,
                        electronic_connectivity=electronic_connectivity,
                        electrolyte_connectivity=electrolyte_connectivity,
                        equivalent_particle_diameter=equivalent_particle_diameter,
                        tortuosity=tortuosity,
                        solid_value=args.solid_value,
                        liquid_values=args.liquid_values,
                    )
                if write_batch_json:
                    output_json = default_output_json_path(input_txt)
                    payload = build_multitask_structure_factor_payload(
                        phase_volume_fractions=phase_volume_fractions,
                        result=result,
                        electronic_connectivity=electronic_connectivity,
                        electrolyte_connectivity=electrolyte_connectivity,
                        equivalent_particle_diameter=equivalent_particle_diameter,
                    )
                    write_multitask_structure_factor_json(output_json, payload)
                success_count += 1
            except Exception as exc:
                failure_count += 1
                print(f"[Failed] {input_txt} -> {exc}")
                output_xlsx = None
                output_json = None

            percent_complete = float(index) / float(len(txt_files)) * 100.0
            report_percent = int(percent_complete // 10) * 10
            if index == len(txt_files):
                report_percent = 100
            if report_percent >= 10 and report_percent > last_reported_percent:
                print(
                    f"[Batch] {index}/{len(txt_files)} ({report_percent}%) "
                    f"| success={success_count} | failed={failure_count}"
                )
                last_reported_percent = report_percent

            if output_xlsx is not None:
                print(f"[Written] {output_xlsx}")
            if output_json is not None:
                print(f"[Written] {output_json}")

        print(f"Batch completed. Success: {success_count}, Failed: {failure_count}")
        return 0 if failure_count == 0 else 1

    if args.input_txt is None:
        raise ValueError("An input TXT path or --batch-root is required.")

    (
        output_xlsx,
        result,
        phase_volume_fractions,
        agglomeration,
        electronic_connectivity,
        electrolyte_connectivity,
        equivalent_particle_diameter,
        tortuosity,
    ) = process_single_structure_file(
        input_txt=args.input_txt,
        voxel_size_um=args.voxel_size_um,
        solid_value=args.solid_value,
        liquid_values=args.liquid_values,
        unknown_cluster_threshold=args.unknown_cluster_threshold,
        output_xlsx=args.output_xlsx,
    )

    print(f"Input TXT: {args.input_txt}")
    print(f"Output Excel: {output_xlsx}")
    print(f"Voxel grid shape (x, y, z): {result.shape_xyz}")
    print(f"Voxel size h (um): {result.voxel_size_um:.10g}")
    print(f"Solid phase value: {args.solid_value}")
    print(f"Liquid phase values: {list(args.liquid_values)}")
    print(
        "Electrolyte volume fraction: "
        f"{phase_volume_fractions.electrolyte_volume_fraction:.10g} "
        f"({phase_volume_fractions.electrolyte_voxel_count} voxels)"
    )
    print(
        "Particle volume fraction: "
        f"{phase_volume_fractions.particle_volume_fraction:.10g} "
        f"({phase_volume_fractions.particle_voxel_count} voxels)"
    )
    print(
        "CBD volume fraction: "
        f"{phase_volume_fractions.cbd_volume_fraction:.10g} "
        f"({phase_volume_fractions.cbd_voxel_count} voxels)"
    )
    print(f"Solid voxel count: {result.solid_voxel_count}")
    print(f"Solid volume (um^3): {result.solid_volume_um3:.10g}")
    print(f"Solid-liquid interface face count: {result.interface_face_count}")
    print(f"Total solid-liquid interface area (um^2): {result.total_interface_area_um2:.10g}")
    print(
        "Specific surface area (interface area / solid volume, 1/um): "
        f"{result.specific_surface_area_inv_um:.10g}"
    )
    print(f"Particle cluster count (26-neighborhood): {agglomeration.particle_cluster_count}")
    print(f"Largest cluster volume (voxels): {agglomeration.largest_cluster_volume_voxels}")
    print(f"Unknown cluster count: {agglomeration.unknown_cluster_count}")
    print(f"Unknown cluster volume (voxels): {agglomeration.unknown_cluster_volume_voxels}")
    print(f"Internal cluster count: {agglomeration.internal_cluster_count}")
    print(f"Known cluster volume (voxels): {agglomeration.known_cluster_volume_voxels}")
    print(
        "Unknown cluster volume fraction (V_unknown / V_total): "
        f"{agglomeration.unknown_cluster_volume_fraction:.10g}"
    )
    print(
        "Representative sample under threshold "
        f"{agglomeration.unknown_cluster_threshold:.10g}: "
        f"{agglomeration.representative_sample}"
    )
    if agglomeration.normalized_average_cluster_volume is None:
        print(
            "Normalized average cluster volume: discarded "
            "(unknown-cluster volume fraction exceeds threshold)"
        )
    else:
        print(
            "Normalized average cluster volume: "
            f"{agglomeration.normalized_average_cluster_volume:.10g}"
        )
    print(
        "Electronically connected particle voxels from collector (z=0): "
        f"{electronic_connectivity.connected_particle_voxel_count}"
    )
    print(
        "Electronic connectivity: "
        f"{electronic_connectivity.electronic_connectivity:.10g}"
    )
    print(
        "Connected electrolyte voxels from z=0: "
        f"{electrolyte_connectivity.connected_electrolyte_voxel_count}"
    )
    print(
        "Electrolyte connectivity: "
        f"{electrolyte_connectivity.electrolyte_connectivity:.10g}"
    )
    print(
        "Electrolyte percolating from z=0 to z=max: "
        f"{electrolyte_connectivity.percolating}"
    )
    if math.isinf(tortuosity.tortuosity_tau):
        print("Tortuosity tau: inf")
    else:
        print(f"Tortuosity tau: {tortuosity.tortuosity_tau:.10g}")
    print(
        "Equivalent particle diameter (um): "
        f"{equivalent_particle_diameter.equivalent_particle_diameter_um:.10g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
