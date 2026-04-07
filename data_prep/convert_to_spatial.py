import numpy as np
import argparse
from pathlib import Path

def convert_time_to_spatial(npz_path, v_kmh, spatial_step=0.25, skip_time=2.0, out_path=None):
    v_ms = v_kmh / 3.6
    
    data = np.load(npz_path, allow_pickle=True)
    out_dict = dict(data)
    t_step = data['dt']
    N_t = len(data['X'])
    time_array = np.arange(N_t) * t_step
    
    valid_idx = time_array >= skip_time
    valid_time = time_array[valid_idx]
    
    start_distance = valid_time[0] * v_ms
    end_distance = valid_time[-1] * v_ms
    
    spatial_queries = np.arange(start_distance, end_distance, spatial_step)
    query_times = spatial_queries / v_ms
    
    print(f"Interpolating from {N_t} time steps to {len(spatial_queries)} spatial steps...")
    
    for k in data.files:
        val = data[k]
        # FIX: Also accept arrays that are exactly N_t + 1 in length
        if isinstance(val, np.ndarray) and val.ndim >= 1 and (len(val) == N_t or len(val) == N_t + 1):
            
            # Trim off-by-one difference before interpolating
            current_len = min(len(val), N_t)
            trimmed_val = val[:current_len]
            
            if trimmed_val.ndim == 1:
                interp_val = np.interp(query_times, valid_time, trimmed_val[valid_idx])
            else:
                interp_val = np.zeros((len(spatial_queries), trimmed_val.shape[1]), dtype=trimmed_val.dtype)
                for c in range(trimmed_val.shape[1]):
                    interp_val[:, c] = np.interp(query_times, valid_time, trimmed_val[valid_idx, c])
            out_dict[k] = interp_val

    out_dict['spatial_s'] = spatial_queries
    out_dict['v_kmh'] = np.array([v_kmh])
    
    if out_path is None:
        out_path = str(npz_path).replace('.npz', '_spatial.npz')
        
    np.savez_compressed(out_path, **out_dict)
    print(f"Saved spatial data to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', default='results/default_project/高速客车-外部导入-vehicle-standard-20260322_065703/files/simulation_result.npz', type=str)
    parser.add_argument('--v_kmh', type=float, default=215.0)
    args = parser.parse_args()
    convert_time_to_spatial(args.npz, args.v_kmh)
