
import numpy as np
import matplotlib.pyplot as plt
import os
import re

class ResultPlotter:
    """仿真结果绘图与保存模块"""
    
    @staticmethod
    def setup_style():
        """Configure publication-style plots with English labels and Times New Roman."""
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['lines.linewidth'] = 1.3
        plt.rcParams['grid.alpha'] = 0.35
        plt.rcParams['grid.linestyle'] = '--'

    @staticmethod
    def _safe_col(arr, col, length):
        """Safely fetch a column from 2D arrays; return zeros if unavailable."""
        if arr is None or arr.ndim != 2 or arr.shape[0] != length or arr.shape[1] <= col:
            return np.zeros(length)
        return arr[:, col]

    @staticmethod
    def _fft_amplitude(signal, dt):
        """One-sided FFT amplitude spectrum."""
        signal = np.asarray(signal)
        signal = signal - np.mean(signal)
        n = signal.size
        if n < 4:
            return np.array([0.0]), np.array([0.0])

        window = np.hanning(n)
        yw = signal * window
        amp = np.abs(np.fft.rfft(yw)) * 2.0 / np.sum(window)
        freq = np.fft.rfftfreq(n, d=dt)
        return freq, amp

    @staticmethod
    def _safe_1d(arr):
        """Return flattened 1D ndarray; empty if invalid."""
        if arr is None:
            return np.array([])
        out = np.asarray(arr).reshape(-1)
        return out

    @staticmethod
    def _sanitize_name(name):
        """Sanitize folder/file names for cross-platform filesystem safety."""
        if name is None:
            return 'Unknown'
        text = str(name).strip()
        text = re.sub(r'[\\/:*?"<>|]+', '_', text)
        text = re.sub(r'\s+', '_', text)
        return text if text else 'Unknown'

    @staticmethod
    def save_data(run_name, X, V, A, spy_dict, dt, idx_car_start, idx_car_end=None, save_dof_mode='full', results_root='results'):
        """Save simulation data under results/<run_name>/files/simulation_result.npz."""
        safe_run_name = ResultPlotter._sanitize_name(run_name)
        run_dir = os.path.join(results_root, safe_run_name)
        files_dir = os.path.join(run_dir, 'files')
        os.makedirs(files_dir, exist_ok=True)
        filepath = os.path.join(files_dir, 'simulation_result.npz')

        mode = str(save_dof_mode).lower().strip()
        if mode not in ('full', 'vehicle'):
            raise ValueError(f"未知的保存模式: {save_dof_mode}，仅支持 full/vehicle")

        if mode == 'vehicle':
            if idx_car_end is None:
                idx_car_end = idx_car_start + 35
            X_save = X[:, idx_car_start:idx_car_end]
            V_save = V[:, idx_car_start:idx_car_end]
            A_save = A[:, idx_car_start:idx_car_end]
            idx_car_start_save = 0
            idx_car_end_save = X_save.shape[1]
        else:
            X_save, V_save, A_save = X, V, A
            if idx_car_end is None:
                idx_car_end = idx_car_start + 35
            idx_car_start_save = idx_car_start
            idx_car_end_save = idx_car_end
        
        # 巧妙之处：把车体自由度索引与保存模式一并存入 npz，供读取时使用
        np.savez_compressed(
            filepath, 
            X=X_save, V=V_save, A=A_save, dt=dt,
            idx_car_start=idx_car_start_save,
            idx_car_end=idx_car_end_save,
            save_dof_mode=mode,
            **spy_dict
        )
        print(f" -> [数据归档] 运行目录: {run_dir}")
        print(f" -> [数据归档] 保存模式: {mode}")
        print(f" -> [数据归档] 仿真结果已成功保存至: {filepath}")
        return filepath

    @staticmethod
    def plot_core_responses(Nt, dt, A, spy_dict, idx_car_start, save_dir=None, show=True):
        """
        Plot extended dynamic responses using saved NPZ structures.
        Optionally save figures to disk and suppress GUI display.
        """
        ResultPlotter.setup_style()
        t_axis = np.arange(Nt) * dt

        # Frequently used channels
        total_v = spy_dict.get('TotalVerticalForce')
        total_l = spy_dict.get('TotalLateralForce')
        total_v_p2 = spy_dict.get('TotalVerticalForce_Point2')
        total_l_p2 = spy_dict.get('TotalLateralForce_Point2')
        yixi_z = spy_dict.get('Yixi_Force_z')
        erxi_z = spy_dict.get('Erxi_Force_z')
        fv_fastener = spy_dict.get('FV_Fastener')
        fl_fastener = spy_dict.get('FL_Fastener')

        body_ay = A[:, idx_car_start] if A.shape[1] > idx_car_start else np.zeros(Nt)
        body_az = A[:, idx_car_start + 1] if A.shape[1] > (idx_car_start + 1) else np.zeros(Nt)

        fig, axes = plt.subplots(3, 3, figsize=(18, 12), constrained_layout=True)
        fig.suptitle('Vehicle-Track Coupled Dynamics: Extended Response Overview', fontsize=16, fontweight='bold')

        # (1) Carbody acceleration
        axes[0, 0].plot(t_axis, body_az, label='Carbody vertical accel $a_z$', color='#1f77b4')
        axes[0, 0].plot(t_axis, body_ay, label='Carbody lateral accel $a_y$', color='#ff7f0e', alpha=0.85)
        axes[0, 0].set_title('Carbody Acceleration Time Histories')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Acceleration (m/s²)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 轮轨力绘图采用“点1 + 点2”的合力（若点2不存在或全零则自动退化为点1）
        total_v_plot = total_v
        if (
            total_v is not None and total_v_p2 is not None
            and total_v.ndim == 2 and total_v_p2.ndim == 2
            and total_v.shape == total_v_p2.shape
            and total_v.shape[0] == Nt
            and np.any(np.abs(total_v_p2) > 1e-9)
        ):
            total_v_plot = total_v + total_v_p2

        total_l_plot = total_l
        if (
            total_l is not None and total_l_p2 is not None
            and total_l.ndim == 2 and total_l_p2.ndim == 2
            and total_l.shape == total_l_p2.shape
            and total_l.shape[0] == Nt
            and np.any(np.abs(total_l_p2) > 1e-9)
        ):
            total_l_plot = total_l + total_l_p2

        # (2) Wheel-rail vertical contact force (axle-1)
        fz_l = ResultPlotter._safe_col(total_v_plot, 0, Nt) / 1000.0
        fz_r = ResultPlotter._safe_col(total_v_plot, 1, Nt) / 1000.0
        axes[0, 1].plot(t_axis, fz_l, label='Left wheel (axle-1)', color='#2a9d8f')
        axes[0, 1].plot(t_axis, fz_r, label='Right wheel (axle-1)', color='#e76f51', alpha=0.9)
        axes[0, 1].plot(t_axis, fz_l + fz_r, label='Axle-1 sum (L+R)', color='#264653', linestyle='--', alpha=0.9)
        axes[0, 1].set_title('Wheel-Rail Vertical Contact Force (Combined)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Force (kN)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 若单轮存在较多零值，提示“单侧失载”而非整轴无接触
        z_l = np.mean(np.abs(fz_l) < 1e-9)
        z_r = np.mean(np.abs(fz_r) < 1e-9)
        if (z_l > 0.05) or (z_r > 0.05):
            axes[0, 1].text(
                0.02, 0.96,
                f'Zero-ratio: L={z_l*100:.1f}%, R={z_r*100:.1f}%',
                transform=axes[0, 1].transAxes,
                fontsize=9,
                va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.65, edgecolor='gray')
            )

        # (3) Wheel-rail lateral contact force (axle-1)
        axes[0, 2].plot(t_axis, ResultPlotter._safe_col(total_l_plot, 0, Nt) / 1000.0, label='Left wheel (axle-1)', color='#6a4c93')
        axes[0, 2].plot(t_axis, ResultPlotter._safe_col(total_l_plot, 1, Nt) / 1000.0, label='Right wheel (axle-1)', color='#1982c4', alpha=0.9)
        axes[0, 2].set_title('Wheel-Rail Lateral Contact Force (Combined)')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Force (kN)')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # (4) Primary suspension vertical force
        axes[1, 0].plot(t_axis, ResultPlotter._safe_col(yixi_z, 0, Nt) / 1000.0, label='Primary spring #1', color='#43aa8b')
        if yixi_z is not None and yixi_z.ndim == 2 and yixi_z.shape[0] == Nt:
            axes[1, 0].plot(t_axis, np.mean(yixi_z, axis=1) / 1000.0, label='Primary average (all)', color='#577590', alpha=0.85)
        axes[1, 0].set_title('Primary Suspension Vertical Force')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Force (kN)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # (5) Secondary suspension vertical force
        axes[1, 1].plot(t_axis, ResultPlotter._safe_col(erxi_z, 0, Nt) / 1000.0, label='Secondary spring #1', color='#f8961e')
        if erxi_z is not None and erxi_z.ndim == 2 and erxi_z.shape[0] == Nt:
            axes[1, 1].plot(t_axis, np.mean(erxi_z, axis=1) / 1000.0, label='Secondary average (all)', color='#f3722c', alpha=0.85)
        axes[1, 1].set_title('Secondary Suspension Vertical Force')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Force (kN)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # (6) Two-point contact contribution (if active)
        p2_left = ResultPlotter._safe_col(total_v_p2, 0, Nt) / 1000.0
        p2_right = ResultPlotter._safe_col(total_v_p2, 1, Nt) / 1000.0
        if np.any(np.abs(p2_left) > 1e-9) or np.any(np.abs(p2_right) > 1e-9):
            axes[1, 2].plot(t_axis, p2_left, label='Point-2 left wheel', color='#9d4edd')
            axes[1, 2].plot(t_axis, p2_right, label='Point-2 right wheel', color='#7b2cbf', alpha=0.9)
            axes[1, 2].set_title('Two-Point Contact Vertical Force')
            axes[1, 2].set_ylabel('Force (kN)')
        else:
            axes[1, 2].text(0.5, 0.5, 'No significant point-2 contact detected', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Two-Point Contact Status')
            axes[1, 2].set_ylabel('')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].legend(loc='upper right') if axes[1, 2].lines else None
        axes[1, 2].grid(True)

        # (7) Fastener force envelope
        if fv_fastener is not None and fv_fastener.ndim == 2 and fv_fastener.shape[0] == Nt:
            axes[2, 0].plot(t_axis, np.mean(np.abs(fv_fastener), axis=1) / 1000.0, label='Vertical fastener mean |F|', color='#118ab2')
        if fl_fastener is not None and fl_fastener.ndim == 2 and fl_fastener.shape[0] == Nt:
            axes[2, 0].plot(t_axis, np.mean(np.abs(fl_fastener), axis=1) / 1000.0, label='Lateral fastener mean |F|', color='#ef476f')
        axes[2, 0].set_title('Fastener Force Envelope')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Mean absolute force (kN)')
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        # (8) Frequency domain: body vertical acceleration
        freq_az, amp_az = ResultPlotter._fft_amplitude(body_az, dt)
        fmax = 100.0
        mask = freq_az <= fmax
        axes[2, 1].plot(freq_az[mask], amp_az[mask], color='#073b4c', label='|FFT($a_z$)|')
        axes[2, 1].set_title('Spectrum of Carbody Vertical Acceleration')
        axes[2, 1].set_xlabel('Frequency (Hz)')
        axes[2, 1].set_ylabel('Amplitude')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        # (9) Frequency domain: axle-1 left vertical contact force
        force_left = ResultPlotter._safe_col(total_v_plot, 0, Nt) / 1000.0
        freq_fz, amp_fz = ResultPlotter._fft_amplitude(force_left, dt)
        mask2 = freq_fz <= fmax
        axes[2, 2].plot(freq_fz[mask2], amp_fz[mask2], color='#264653', label='|FFT($F_z$)|')
        axes[2, 2].set_title('Spectrum of Vertical Contact Force (Axle-1 Left)')
        axes[2, 2].set_xlabel('Frequency (Hz)')
        axes[2, 2].set_ylabel('Amplitude (kN)')
        axes[2, 2].legend()
        axes[2, 2].grid(True)

        saved_paths = []
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            png_path = os.path.join(save_dir, 'extended_response_overview.png')
            svg_path = os.path.join(save_dir, 'extended_response_overview.svg')
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            fig.savefig(svg_path, bbox_inches='tight')
            saved_paths.extend([png_path, svg_path])
            print(f" -> Figures saved: {png_path}")
            print(f" -> Figures saved: {svg_path}")

        # --------------------- Figure 2: Irregularity + Track Profiles ---------------------
        fig2, axes2 = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
        fig2.suptitle('Track Irregularity and Plan-Profile Overview', fontsize=16, fontweight='bold')

        # Data extract
        irre_s = ResultPlotter._safe_1d(spy_dict.get('Irre_distance_m'))
        bz_l = ResultPlotter._safe_1d(spy_dict.get('Irre_bz_L_ref'))
        bz_r = ResultPlotter._safe_1d(spy_dict.get('Irre_bz_R_ref'))
        by_l = ResultPlotter._safe_1d(spy_dict.get('Irre_by_L_ref'))
        by_r = ResultPlotter._safe_1d(spy_dict.get('Irre_by_R_ref'))

        s_rel = ResultPlotter._safe_1d(spy_dict.get('Track_rel_mileage_m'))
        k_prof = ResultPlotter._safe_1d(spy_dict.get('Track_curvature_1pm'))
        h_prof = ResultPlotter._safe_1d(spy_dict.get('Track_cant_m'))
        g_prof = ResultPlotter._safe_1d(spy_dict.get('Track_gradient'))
        z_prof = ResultPlotter._safe_1d(spy_dict.get('Track_vertical_profile_m'))

        # (1) Vertical irregularity
        n1 = min(len(irre_s), len(bz_l), len(bz_r))
        if n1 > 1:
            axes2[0, 0].plot(irre_s[:n1], bz_l[:n1] * 1000.0, label='Left rail vertical irregularity', color='#1f77b4')
            axes2[0, 0].plot(irre_s[:n1], bz_r[:n1] * 1000.0, label='Right rail vertical irregularity', color='#d62728', alpha=0.85)
            axes2[0, 0].set_xlabel('Distance (m)')
            axes2[0, 0].set_ylabel('Irregularity (mm)')
            axes2[0, 0].set_title('Vertical Irregularity')
            axes2[0, 0].legend()
        else:
            axes2[0, 0].text(0.5, 0.5, 'No vertical irregularity data', ha='center', va='center', transform=axes2[0, 0].transAxes)
            axes2[0, 0].set_title('Vertical Irregularity')
        axes2[0, 0].grid(True)

        # (2) Lateral irregularity
        n2 = min(len(irre_s), len(by_l), len(by_r))
        if n2 > 1:
            axes2[0, 1].plot(irre_s[:n2], by_l[:n2] * 1000.0, label='Left rail lateral irregularity', color='#2ca02c')
            axes2[0, 1].plot(irre_s[:n2], by_r[:n2] * 1000.0, label='Right rail lateral irregularity', color='#9467bd', alpha=0.85)
            axes2[0, 1].set_xlabel('Distance (m)')
            axes2[0, 1].set_ylabel('Irregularity (mm)')
            axes2[0, 1].set_title('Lateral Irregularity')
            axes2[0, 1].legend()
        else:
            axes2[0, 1].text(0.5, 0.5, 'No lateral irregularity data', ha='center', va='center', transform=axes2[0, 1].transAxes)
            axes2[0, 1].set_title('Lateral Irregularity')
        axes2[0, 1].grid(True)

        # (3) Plan profile (curvature + cant)
        n3 = min(len(s_rel), len(k_prof), len(h_prof))
        if n3 > 1:
            ax31 = axes2[1, 0]
            ax32 = ax31.twinx()
            l1, = ax31.plot(s_rel[:n3], k_prof[:n3], color='#ff7f0e', label='Curvature')
            l2, = ax32.plot(s_rel[:n3], h_prof[:n3] * 1000.0, color='#17becf', label='Cant')
            ax31.set_xlabel('Relative mileage (m)')
            ax31.set_ylabel('Curvature (1/m)', color='#ff7f0e')
            ax32.set_ylabel('Cant (mm)', color='#17becf')
            ax31.set_title('Plan Profile (Curvature & Cant)')
            ax31.legend([l1, l2], ['Curvature', 'Cant'], loc='upper right')
            ax31.grid(True)
        else:
            axes2[1, 0].text(0.5, 0.5, 'No plan profile data', ha='center', va='center', transform=axes2[1, 0].transAxes)
            axes2[1, 0].set_title('Plan Profile (Curvature & Cant)')
            axes2[1, 0].grid(True)

        # (4) Longitudinal profile (gradient + integrated elevation)
        n4 = min(len(s_rel), len(g_prof), len(z_prof))
        if n4 > 1:
            ax41 = axes2[1, 1]
            ax42 = ax41.twinx()
            l3, = ax41.plot(s_rel[:n4], g_prof[:n4] * 1000.0, color='#8c564b', label='Gradient')
            l4, = ax42.plot(s_rel[:n4], z_prof[:n4] * 1000.0, color='#e377c2', label='Integrated elevation')
            ax41.set_xlabel('Relative mileage (m)')
            ax41.set_ylabel('Gradient (‰)', color='#8c564b')
            ax42.set_ylabel('Elevation (mm)', color='#e377c2')
            ax41.set_title('Longitudinal Profile (Gradient & Elevation)')
            ax41.legend([l3, l4], ['Gradient', 'Integrated elevation'], loc='upper right')
            ax41.grid(True)
        else:
            axes2[1, 1].text(0.5, 0.5, 'No longitudinal profile data', ha='center', va='center', transform=axes2[1, 1].transAxes)
            axes2[1, 1].set_title('Longitudinal Profile (Gradient & Elevation)')
            axes2[1, 1].grid(True)

        if save_dir:
            png_path2 = os.path.join(save_dir, 'track_irregularity_and_profile.png')
            svg_path2 = os.path.join(save_dir, 'track_irregularity_and_profile.svg')
            fig2.savefig(png_path2, dpi=300, bbox_inches='tight')
            fig2.savefig(svg_path2, bbox_inches='tight')
            saved_paths.extend([png_path2, svg_path2])
            print(f" -> Figures saved: {png_path2}")
            print(f" -> Figures saved: {svg_path2}")

        if show:
            plt.show()
        else:
            plt.close(fig)
            plt.close(fig2)

        return saved_paths