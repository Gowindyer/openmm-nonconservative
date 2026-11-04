#!/usr/bin/env python3
"""Inventory a serialized CG System and benchmark it on OpenMM's CPU platform.

The timed region contains only integrator steps.  Context construction, initial
state evaluation, and warmup are reported separately from throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import openmm as mm
from openmm import app, unit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def force_inventory(system: mm.System) -> list[dict[str, object]]:
    forces = []
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        record: dict[str, object] = {
            "index": index,
            "class": type(force).__name__,
            "group": force.getForceGroup(),
        }
        if isinstance(force, mm.CustomNonbondedForce):
            record.update(
                method=force.getNonbondedMethod(),
                cutoff_nm=force.getCutoffDistance().value_in_unit(unit.nanometer),
                exclusions=force.getNumExclusions(),
                interaction_groups=force.getNumInteractionGroups(),
                per_particle_parameters=force.getNumPerParticleParameters(),
                global_parameters=force.getNumGlobalParameters(),
                tabulated_functions=force.getNumTabulatedFunctions(),
                switching=force.getUseSwitchingFunction(),
            )
        elif isinstance(force, mm.NonbondedForce):
            record.update(
                method=force.getNonbondedMethod(),
                cutoff_nm=force.getCutoffDistance().value_in_unit(unit.nanometer),
                exceptions=force.getNumExceptions(),
            )
        forces.append(record)
    return forces


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return number


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-xml", required=True, type=Path)
    parser.add_argument("--pdb", type=Path, help="Coordinates for a timed CPU run")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--warmup-steps", type=nonnegative_int, default=100)
    parser.add_argument("--steps", type=positive_int, default=1000)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--friction-per-ps", type=float, default=1.0)
    parser.add_argument("--timestep-ps", type=float, default=0.01)
    parser.add_argument("--seed", type=nonnegative_int, default=2026)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if not args.inventory_only and args.pdb is None:
        parser.error("--pdb is required for a timed run")
    if args.temperature_k <= 0 or args.friction_per_ps < 0 or args.timestep_ps <= 0:
        parser.error("temperature and timestep must be positive; friction must be nonnegative")
    return args


def timed_run(system: mm.System, positions, args: argparse.Namespace) -> dict[str, float]:
    integrator = mm.LangevinMiddleIntegrator(
        args.temperature_k * unit.kelvin,
        args.friction_per_ps / unit.picosecond,
        args.timestep_ps * unit.picoseconds,
    )
    integrator.setRandomNumberSeed(args.seed)
    platform = mm.Platform.getPlatformByName("CPU")
    start = perf_counter()
    context = mm.Context(system, integrator, platform, {"Threads": str(args.threads)})
    context_seconds = perf_counter() - start
    context.setPositions(positions)
    context.setVelocitiesToTemperature(args.temperature_k * unit.kelvin, args.seed)
    initial_kj_mol = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    integrator.step(args.warmup_steps)
    start = perf_counter()
    integrator.step(args.steps)
    step_seconds = perf_counter() - start
    del context, integrator
    return {
        "context_seconds": context_seconds,
        "initial_potential_kj_mol": initial_kj_mol,
        "step_seconds": step_seconds,
        "steps_per_second": args.steps / step_seconds,
    }


def main() -> None:
    args = arguments()
    system = mm.XmlSerializer.deserialize(args.system_xml.read_text(encoding="utf-8"))
    result: dict[str, object] = {
        "openmm_version": mm.version.version,
        "system_xml_sha256": sha256(args.system_xml),
        "particles": system.getNumParticles(),
        "forces": force_inventory(system),
    }
    if not args.inventory_only:
        assert args.pdb is not None
        pdb = app.PDBFile(str(args.pdb))
        if pdb.topology.getNumAtoms() != system.getNumParticles():
            raise ValueError("PDB atom count does not match System particle count")
        runs = [timed_run(system, pdb.positions, args) for _ in range(args.repeats)]
        result.update(
            pdb_sha256=sha256(args.pdb),
            platform="CPU",
            threads=args.threads,
            integrator="LangevinMiddleIntegrator",
            temperature_k=args.temperature_k,
            friction_per_ps=args.friction_per_ps,
            timestep_ps=args.timestep_ps,
            seed=args.seed,
            warmup_steps=args.warmup_steps,
            timed_steps=args.steps,
            runs=runs,
            median_steps_per_second=median(run["steps_per_second"] for run in runs),
        )
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is None:
        print(output, end="")
    else:
        args.output_json.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
