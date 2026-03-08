import os
import json
from trajectory import fit_trajectory, predict_plate_crossing

def main():
    sample_path = '/Users/peter/mlbstats/pitch_sample/sample_pitches.json'
    if not os.path.exists(sample_path):
        print(f"Error: {sample_path} not found.")
        return

    with open(sample_path, 'r') as f:
        pitches = json.load(f)

    print(f"--- Pitch Path Bayesian Refinement Demo ---")
    print(f"Processing {len(pitches)} pitches...\n")
    
    results = []
    for i, pitch in enumerate(pitches):
        print(f"[{i+1}/{len(pitches)}] {pitch['pitcher']} -> {pitch['batter']} ({pitch['pitch_type']})")
        
        # Original estimation
        x_orig, z_orig = predict_plate_crossing(pitch['parameters'])
        err_x_orig = x_orig - pitch['actual_plate']['x']
        err_z_orig = z_orig - pitch['actual_plate']['z']
        
        # Bayesian Refinement
        smoothed = fit_trajectory(pitch['parameters'], pitch['actual_plate'])
        
        # Refined estimation
        refined_params = pitch['parameters'].copy()
        refined_params.update(smoothed)
        x_ref, z_ref = predict_plate_crossing(refined_params, time=smoothed['time'])
        err_x_ref = x_ref - pitch['actual_plate']['x']
        err_z_ref = z_ref - pitch['actual_plate']['z']
        
        print(f"  Initial Error: x={err_x_orig:+.4f}, z={err_z_orig:+.4f}")
        print(f"  Refined Error: x={err_x_ref:+.4f}, z={err_z_ref:+.4f}")
        print(f"  Accel Adjust:  ax={smoothed['ax']-pitch['parameters']['ax']:+.2f}, "
              f"ay={smoothed['ay']-pitch['parameters']['ay']:+.2f}, "
              f"az={smoothed['az']-pitch['parameters']['az']:+.2f}\n")
        
        results.append({
            'metadata': {
                'pitcher': pitch['pitcher'],
                'batter': pitch['batter'],
                'type': pitch['pitch_type']
            },
            'original': pitch['parameters'],
            'refined': smoothed,
            'plate_actual': pitch['actual_plate']
        })

    output_path = '/Users/peter/mlbstats/pitchpath/pymc_smoother/results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {output_path}")

if __name__ == "__main__":
    main()
