from __future__ import annotations
from pathlib import Path
from subprocess import run
from datetime import datetime
from typing import Self
import re
import numpy as np
import numpy.typing as npt


class UPFv1:
    """Class for reading and storing some of the information in old
    versions of UPF pseudopotentials (UPFv1). Not all information is
    retained because the intention of this class is to grab the
    information that is either not copied over to a UPFv2 file or is
    subjected to rounding errors.
    (because apparently 16 decimals is one too many to copy over...)
    """

    def __init__(
        self,
        pp_path: Path,
        pp_info: str,
        pp_header: str,
        pp_r: list,
        pp_rab: list,
        pp_local: list,
        pp_betas: dict[int, dict],
        pp_pswfcs: dict[str, list],
        pp_rhoatom: list,
    ):
        self.pp_path = pp_path
        self.pp_info = pp_info
        self.pp_header = pp_header
        self.pp_r = pp_r
        self.pp_rab = pp_rab
        self.pp_local = pp_local
        self.pp_betas = pp_betas
        self.pp_pswfcs = pp_pswfcs
        self.pp_rhoatom = pp_rhoatom


    @classmethod
    def from_upf_file(cls, upf_file: Path) -> Self:
        """Read in an old-format UPF file."""
        with open(upf_file, "r") as upf:
            upf_str = upf.read()

        section_blocks = list(re.finditer(r"(<.+?>)", upf_str))

        delimiters = re.compile("(<|>|/)")
        pp_betas = {}
        for section_index, section in enumerate(section_blocks):
            # Skip over section ends
            if "/" in section.group():
                continue

            section_start = section.start()
            section_end = section_blocks[section_index+1].end()

            match delimiters.sub("", section.group()):
                case "PP_INFO":
                    pp_info = upf_str[section_start:section_end].splitlines()[1:-1]
                case "PP_HEADER":
                    pp_header = upf_str[section_start:section_end].splitlines()[1:-1]
                case "PP_MESH":
                    continue
                case "PP_R":
                    pp_r = cls.parse_pp_generic_grid(
                        upf_str[section_start:section_end]
                    )
                case "PP_RAB":
                    pp_rab = cls.parse_pp_generic_grid(
                        upf_str[section_start:section_end]
                    )
                case "PP_LOCAL":
                    pp_local = cls.parse_pp_generic_grid(
                        upf_str[section_start:section_end]
                    )
                case "PP_NONLOCAL":
                    continue
                case "PP_BETA":
                    grid_data, index, angular_momentum, grid_points = cls.parse_pp_beta(
                        upf_str[section_start:section_end]
                    )
                    pp_betas[index] = {
                        "grid_data": grid_data,
                        "angular_momentum": angular_momentum,
                        "grid_points": grid_points,
                    }
                case "PP_DIJ": # This is converted w/out rounding by upfconv.x
                    continue
                case "PP_PSWFC":
                    pp_pswfcs = cls.parse_pp_pswfc(
                        upf_str[section_start:section_end]
                    )
                case "PP_RHOATOM":
                    pp_rhoatom = cls.parse_pp_generic_grid(
                        upf_str[section_start:section_end]
                    )
                case "PP_ADDINFO":
                    # This is not described anywhere.
                    # QE's upfconv.x converts this into a PP_SPIN_ORB block, but that
                    # block is also not described anywhere...
                    continue
            #end match
        #end for
        return cls(
            pp_info,
            pp_header,
            pp_r,
            pp_rab,
            pp_local,
            pp_betas,
            pp_pswfcs,
            pp_rhoatom,
        )

    @staticmethod
    def parse_pp_generic_grid(grid_str: str) -> list[str]:
        """Parse grid data from a UPFv1 block like ``PP_RHOATOM``."""

        grid_lines = grid_str.splitlines()[1:-1]
        grid_data = []
        for line in grid_lines:
            grid_data.extend(line.strip().split())

        return grid_data

    @staticmethod
    def parse_pp_beta(grid_str: str) -> list[str]:
        """Parse grid data from a UPFv1 ``PP_BETA`` block."""

        grid_lines = grid_str.splitlines()[1:-1]

        pp_beta_info = grid_lines.pop(0).strip().split()
        index = int(pp_beta_info[0])
        angular_momentum = int(pp_beta_info[1])
        grid_points = int(grid_lines.pop(0).strip())

        grid_data = []
        for line in grid_lines:
            grid_data.extend(line.strip().split())

        return grid_data, index, angular_momentum, grid_points

    @staticmethod
    def parse_pp_pswfc(grid_str: str) -> list[str]:
        """Parse grid data from a UPFv1 ``PP_PSWFC`` block."""

        grid_lines = grid_str.splitlines()[1:-1]

        wfcs = {}
        wfc_block = "None"
        for line in grid_lines:
            line = line.strip()
            if line.startswith(("S", "P", "D", "F")):
                wfc_block = line.split()[0]
                if wfc_block in list(wfcs.keys()):
                    wfc_block = wfc_block + "2"
                wfcs[wfc_block] = []
            else:
                wfcs[wfc_block].extend(line.split())

        return wfcs


    def get_info(self, info: str) -> str | datetime:
        """Get information from ``PP_INFO``.
        
        Parameters
        ----------
        info : {"opium_version", "generation_date"}
        """

        match info:
            case "opium_version":
                version = [
                    line for line in self.pp_info if line.startswith("#Opium version")
                ][0]
                return version.split(":")[-1]
            case "generation_date":
                date = [
                    line for line in self.pp_info if line.startswith("#Execution Date")
                ][0]
                return datetime.strptime(
                    date.split(":", 1)[-1],
                    "%a %b %d %H:%M:%S %Y"
                )# Thu Jan  4 11:15:10 2024


    def fix_upfconv_output(self, upfv2_file: Path, output_file: Path):
        """Fix a UPFv2 file generated by ``upfconv.x`` with the data
        from the original UPFv1 file and write to ``output_file``.
        """

        with open(upfv2_file, "r") as upf2:
            upf2_str = upf2.read()

        section_blocks = list(re.finditer(r"(<.+?>)", upf2_str))

        new_upf2_lines = []
        for section_index, section in enumerate(section_blocks):

            section_start = section.start()
            section_end = section_blocks[section_index+1].end()

            match section.group().split(maxsplit=1)[0]:
                case "<PP_INFO>":
                    info_lines = upf2_str[section_start:section_end].splitlines()
                    new_upf2_lines.extend(self.populate_info(info_lines))
                case "<PP_HEADER":
                    ...

        return new_upf2_lines


    def populate_info(self, info_lines: list[str]) -> list[str]:
        """Take in a ``PP_INFO`` section from ``upfconv.x`` and populate
        it with the data from ``self``.
        """
        new_info = []
        for line in info_lines:
            if line.strip().startswith("Generated by new atomic code"):
                op_version = self.get_info("opium_version")
                new_info.append(
                    f"    Generated using OPIUM v{op_version}\n"
                )
            elif line.strip().startswith("Generation date:"):
                gen_date = self.get_info("generation_date")
                gen_date = gen_date.strftime(
                    "%a %d %b %Y, %H:%M:%S"
                )
                new_info.append(
                    f"    Generation date: {gen_date}\n"
                )
            elif line.strip().startswith("Generation configuration: not available"):
                new_info.append(
                    "    <PP_INPUTFILE>\n"
                )
                # Dump the Opium input directly into this block
                new_info.extend(self.pp_info)

                new_info.append(
                    "    </PP_INPUTFILE>\n"
                )
            else:
                new_info.append(line+"\n")

        return new_info


def calc_quantum_number_n(
    orbital_filling: dict[str, float],
    pseudo_wavefunction: npt.ArrayLike,
    angular_momentum: int,
) -> int:
    """Attempt to calculate the principle quantum number for a
    pseudo-wavefunction by counting the number of nodes.

    Parameters
    ----------
    orbital_filling : dict[str, float]
        A dictionary where the keys are strings of the quantum numbers
        and the values are their occupancies.

        This would look like ``{"100": 2.0, "200": 2.0, "210": 6.0}``
        for fully filled 1S, 2S, and 2P orbitals.
    pseudo_wavefunction : ArrayLike of float
        A linear array of the pseudo-wavefunction from the ``PP_PSWFC``
        block in a UPF pseudopotential file.
    angular_momentum : int
        The angular momentum of the provided pseudo wavefunction.
    """




