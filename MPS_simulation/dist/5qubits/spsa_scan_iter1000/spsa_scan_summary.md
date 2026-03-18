# SPSA Parameter Scan

## Setup
- Learning rates: `[0.002, 0.005, 0.01]`
- SPSA c values: `[0.01, 0.02, 0.05]`
- Iterations per run: `1000`
- Shared SPSA seed: `1234`

## Best By Final Residual
- Learning rate: `0.01`
- SPSA c: `0.05`
- Final residual: `0.989178582751`
- Final global cost: `0.490373245332`
- Final relative solution error: `0.994740066025`
- Run report: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/5qubits/spsa_scan_iter1000/spsa_lr0p01_c0p05_iter1000_report.md`

## Ranking
1. `lr=0.01`, `c=0.05`: residual=`0.989178582751`, cost=`0.490373245332`, rel_err=`0.994740066025`
2. `lr=0.002`, `c=0.05`: residual=`0.990714434991`, cost=`0.492418357864`, rel_err=`0.992953418678`
3. `lr=0.002`, `c=0.02`: residual=`0.991818457829`, cost=`0.493389738826`, rel_err=`0.994586709426`
4. `lr=0.002`, `c=0.01`: residual=`0.992111838955`, cost=`0.493647947685`, rel_err=`0.994957166025`
5. `lr=0.01`, `c=0.02`: residual=`0.994779045924`, cost=`0.496334712393`, rel_err=`0.998223040333`
6. `lr=0.005`, `c=0.05`: residual=`0.996874378018`, cost=`0.497546073558`, rel_err=`0.999839466478`
7. `lr=0.01`, `c=0.01`: residual=`0.9980480312`, cost=`0.500323499751`, rel_err=`1.00123299119`
8. `lr=0.005`, `c=0.01`: residual=`1.0012042761`, cost=`0.501513245568`, rel_err=`1.00122763665`
9. `lr=0.005`, `c=0.02`: residual=`1.00480448254`, cost=`0.505744233202`, rel_err=`1.00802773313`

## Heatmap
- Residual heatmap: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/5qubits/spsa_scan_iter1000/spsa_scan_residual_heatmap.png`
