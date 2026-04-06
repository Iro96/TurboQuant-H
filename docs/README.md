# Docs Index

This folder is the newcomer-friendly map for the TurboQuant-H project.

## Read These First

1. [Project Map](./project-map.md)
2. [Runtime Flow](./runtime-flow.md)
3. [Compression Guide](./compression-guide.md)

## What Each Doc Helps With

- `project-map.md`: understand the repository layout and what each top-level file or folder is for.
- `runtime-flow.md`: follow one benchmark run from the wrapper script to the final printed report.
- `compression-guide.md`: understand the compression subsystem and which module owns which responsibility.

## Fast Mental Model

- `turboquant_h_smollm_benchmark.py` is the compatibility entrypoint.
- `src/turboquant_h/cli.py` turns CLI args into config objects.
- `src/turboquant_h/benchmark.py` runs the model, generation loop, and compression benchmark.
- `src/turboquant_h/compression/` contains the KV-cache compression logic.
- `src/turboquant_h/reporting.py` formats the final terminal output.
- `tests/` covers small, deterministic pieces of the project.

## Suggested First Steps For A New Contributor

1. Read [Project Map](./project-map.md) to learn the repo shape.
2. Read [Runtime Flow](./runtime-flow.md) to see how execution moves across modules.
3. Read [Compression Guide](./compression-guide.md) before editing anything under `compression/`.
4. Run `python turboquant_h_smollm_benchmark.py --help` to see the exposed interface.
