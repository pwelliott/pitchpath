import json
import math
import os

def calculate_flight_path(params, actual_plate, pitcher, batter, pitch_type):
    # Parameters from Statcast
    x0, y0, z0 = params['x0'], params['y0'], params['z0']
    vx0, vy0, vz0 = params['vx0'], params['vy0'], params['vz0']
    ax, ay, az = params['ax'], params['ay'], params['az']
    
    # Calculate time to plate (solve y(t) = 0 for 50ft to 0ft)
    # y(t) = y0 + vy0*t + 0.5*ay*t^2 = 0
    # Use quadratic formula: t = (-vy0 - sqrt(vy0^2 - 2*ay*y0)) / ay
    # Note: vy0 is usually negative (moving toward 0), ay is usually positive (drag)
    
    discriminant = vy0**2 - 2 * ay * y0
    if discriminant < 0:
        return None
        
    t_plate = (-vy0 - math.sqrt(discriminant)) / ay
    
    # Calculate crossing coordinates
    x_plate = x0 + vx0 * t_plate + 0.5 * ax * t_plate**2
    z_plate = z0 + vz0 * t_plate + 0.5 * az * t_plate**2
    
    print(f"--- Pitch: {pitcher} to {batter} ({pitch_type}) ---")
    print(f"Flight Time: {t_plate:.4f}s")
    print(f"Estimated: x={x_plate:.3f}, z={z_plate:.3f}")
    if actual_plate['x'] is not None:
        print(f"Actual:    x={actual_plate['x']:.3f}, z={actual_plate['z']:.3f}")
        print(f"Error:     x={abs(x_plate - actual_plate['x']):.4f}, z={abs(z_plate - actual_plate['z']):.4f}")
    print("")
    
    return {
        't_plate': t_plate,
        'estimated_plate': {'x': x_plate, 'z': z_plate},
        'actual_plate': actual_plate
    }

if __name__ == "__main__":
    data_path = "/Users/peter/mlbstats/pitch_sample/sample_pitches.json"
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Run fetch_samples.py first.")
        exit(1)
        
    with open(data_path, 'r') as f:
        pitches = json.load(f)
        
    results = []
    for pitch in pitches:
        # Some pitches might have missing data
        if any(v is None for v in pitch['parameters'].values()):
            continue
            
        res = calculate_flight_path(
            pitch['parameters'], 
            pitch['actual_plate'],
            pitch['pitcher'],
            pitch['batter'],
            pitch['pitch_type']
        )
        if res:
            results.append(res)
            
    print(f"Processed {len(results)} pitches.")
