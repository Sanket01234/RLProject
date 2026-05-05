import csv, numpy as np

rows = []
with open('logs/figure8_20260505_225721.csv') as f:
    for r in csv.DictReader(f):
        rows.append({k: float(v) for k, v in r.items()})

print("Step  | z_true | z_ref | pos_err | u_T    | u_roll | u_pitch")
print("-" * 70)
for r in rows:
    s = int(r['step'])
    if 200 <= s <= 345 and s % 5 == 0:
        zt = r['z_true']
        zr = r['z_ref']
        pe = r['pos_err']
        ut = r['u_T']
        ur = r['u_roll']
        up = r['u_pitch']
        print(f"  {s:>4} | {zt:6.3f} | {zr:5.2f}  | {pe:6.3f}  | {ut:6.2f} | {ur:+6.3f} | {up:+6.3f}")

print(f"\nTotal logged steps: {len(rows)}")

# Check where z drops below 0.5
low_z = [r for r in rows if r['z_true'] < 0.5]
if low_z:
    first = int(low_z[0]['step'])
    print(f"First z < 0.5m at step {first}")

# Check where pos_err spikes
high_err = [r for r in rows if r['pos_err'] > 0.5]
if high_err:
    first = int(high_err[0]['step'])
    print(f"First pos_err > 0.5m at step {first}")
