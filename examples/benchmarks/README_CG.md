# Coarse-grained CPU baseline

`cg_cpu_baseline.py` inventories a serialized OpenMM `System` and measures CPU integration throughput with matching PDB coordinates. It records input hashes, force classes, relevant cutoff metadata, context construction time, and repeated step timing in JSON. Context creation and warmup are outside the timed interval.

Generate the System XML and PDB with the intended coarse-grained model builder, then run:

```bash
python examples/benchmarks/cg_cpu_baseline.py \
  --system-xml /path/to/system.xml \
  --pdb /path/to/positions.pdb \
  --threads 1 --warmup-steps 1000 --steps 10000 --repeats 3 \
  --output-json cg_cpu_baseline.json
```

Set temperature, friction, time step, and random seed explicitly when they differ from the command defaults. Record the model builder revision, CPU model and affinity, input preparation, and the output JSON with each result. Keep those settings and the serialized System identical when comparing a future distributed platform with this baseline.

To inspect force coverage without creating a `Context`, pass `--inventory-only` and omit `--pdb`. The output identifies every force class. A force absent from a future MPI platform's compatibility table must be rejected during its `Context` preflight.

The openMICRON iSoLF POPC example is an initial smoke fixture: its 640-bead saved System includes a membrane barostat, center-of-mass remover, bonded terms, and a tabulated `CustomNonbondedForce`. Its size is suitable for checking inputs and force inventory. Performance scaling needs substantially larger systems and separate measurement of communication and load balance.
