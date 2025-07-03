import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class ReferencePath:
    def __init__(self, complexity=3, t_max=135, dt=0.01):
        self.complexity = complexity
        self.t_max = t_max
        self.dt = dt
    
    def smooth_poly7_transition(self, t, t_start, duration=4.0):
        """
        Create extremely smooth transition using 7th order polynomial
        with zero initial and final velocity, acceleration, and jerk
        
        Increased default transition duration for smoother changes
        """
        if t < t_start:
            return 0.0
        elif t > t_start + duration:
            return 1.0
        else:
            # Normalized time from 0 to 1
            s = (t - t_start) / duration
            # 7th order polynomial with boundary conditions
            return s**4 * (35 - 84*s + 70*s**2 - 20*s**3)
    
    def smooth_derivatives(self, data, window=201, polyorder=3, iterations=3):
        """
        Enhanced smoothing with:
        - Larger window size
        - More smoothing iterations
        - Higher-order polynomial fit
        """
        smoothed = data.copy()
        
        # Ensure window size is odd and not larger than array
        window = min(window, len(data) // 2 * 2 + 1)
        
        # Apply multiple passes of smoothing
        for _ in range(iterations):
            smoothed = savgol_filter(smoothed, window, polyorder)
            
        return smoothed
    
    def generate_path(self):
        # Time parameters - removed transition 1-2 duration
        t = np.arange(0, self.t_max + self.dt, self.dt)
        
        # Define the phase timing parameters explicitly (removed t1_duration)
        self.p1_duration = 5.0    # Vertical climb duration
        self.p2_duration = 80.0   # Cruise duration
        self.t2_duration = 4.0    # Transition 2-3 duration
        
        # Phase start/end times calculated from durations (removed transition 1-2)
        self.p1_start = 0.0
        self.p1_end = self.p1_start + self.p1_duration
        self.p2_start = self.p1_end  # Phase 2 starts immediately after phase 1
        self.p2_end = self.p2_start + self.p2_duration
        self.t2_start = self.p2_end  # Transition 2 starts at end of phase 2
        self.t2_end = self.t2_start + self.t2_duration
        self.p3_start = self.t2_end  # Phase 3 starts at end of transition 2
        
        print(f"Phase timing (seconds):")
        print(f"  Phase 1 (Vertical): {self.p1_start}-{self.p1_end}")
        print(f"  Phase 2 (Cruise): {self.p2_start}-{self.p2_end}")
        print(f"  Transition 2-3: {self.t2_start}-{self.t2_end}")
        print(f"  Phase 3 (Helical): {self.p3_start}-{self.t_max}")
        
        # Path parameters with reduced peak velocities
        climb_height = 5.0
        cruise_speed = 0.8   # Reduced from 1.0
        helix_radius = 4   
        helix_period = 12.0  # Increased period for smoother motion
        helix_angular_vel = 2*np.pi/helix_period
        helix_vertical_speed = 0.4  # Reduced from 0.5
        
        # Initialize arrays
        xd = np.zeros_like(t)
        yd = np.zeros_like(t)
        zd = np.zeros_like(t)
        phase = np.ones_like(t)
        
        # PHASE 1: Smooth Vertical Climb
        climb_mask = t <= self.p1_end
        for i, time in enumerate(t[climb_mask]):
            # Smooth sigmoid-like climb instead of linear
            if self.p1_duration > 0:
                progress = time / self.p1_duration
                # Simple linear climb - no fancy transitions needed for phase 1
                climb_progress = time / self.p1_duration  # Linear from 0 to 1
            else:
                # smooth_progress = 1.0
                climb_progress = 1.0
            # Ensure bounds
            climb_progress = min(1.0, max(0.0, climb_progress))

            xd[i] = 0.0
            yd[i] = 0.0
            zd[i] = climb_height * climb_progress  # Linear climb from 0 to 5m
            phase[i] = 1.0
        
        # PHASE 2: Cruise Phase (X movement) - starts immediately after Phase 1
        cruise_mask = (t > self.p2_start) & (t <= self.t2_start)
        for i, time in enumerate(t[cruise_mask], start=np.sum(climb_mask)):
            phase[i] = 2.0
            xd[i] = cruise_speed * (time - self.p2_start)  # Start X movement from 0
            yd[i] = 0.0  # No Y movement during cruise
            zd[i] = climb_height  # Maintain climb height
        
        # Transition 2-3: Cruise to Helical (X and Y only, Z jumps directly)
        transition_2_3_mask = (t > self.t2_start) & (t <= self.p3_start)
        x_cruise_end = cruise_speed * self.p2_duration  # Total X distance from cruise phase
        for i, time in enumerate(t[transition_2_3_mask], start=np.sum(climb_mask) + np.sum(cruise_mask)):
            transition_progress = self.smooth_poly7_transition(time, self.t2_start, self.t2_duration)
            phase[i] = 2.0 + transition_progress
            
            # Smoother transition to helical trajectory
            helix_time = transition_progress * 0.5 * helix_period
            helix_angle = helix_angular_vel * helix_time
            
            # Blend cruise and helix positions for X and Y only
            target_x = x_cruise_end + helix_radius * np.cos(helix_angle)
            target_y = helix_radius * np.sin(helix_angle)
            
            xd[i] = x_cruise_end * (1 - transition_progress) + target_x * transition_progress
            yd[i] = target_y * transition_progress
            # Z axis: Direct jump to helical Z movement (no transition blending)
            zd[i] = climb_height + helix_vertical_speed * (time - self.t2_start)
        
        # Helical Trajectory Phase
        helix_mask = t > self.p3_start
        for i, time in enumerate(t[helix_mask], start=np.sum(climb_mask) + np.sum(cruise_mask) + np.sum(transition_2_3_mask)):
            phase[i] = 3.0
            helix_time = 0.5 * helix_period + (time - self.p3_start)
            helix_angle = helix_angular_vel * helix_time
            
            xd[i] = x_cruise_end + helix_radius * np.cos(helix_angle)
            yd[i] = helix_radius * np.sin(helix_angle)
            zd[i] = climb_height + helix_vertical_speed * (time - self.t2_start)
        
        # Enhanced smoothing with larger windows and more iterations
        xd = self.smooth_derivatives(xd, window=301, polyorder=3, iterations=3)
        yd = self.smooth_derivatives(yd, window=301, polyorder=3, iterations=3)
        zd = self.smooth_derivatives(zd, window=301, polyorder=3, iterations=3)
        
        # Calculate velocities and accelerations with smoother derivatives
        xdot = np.gradient(xd, self.dt)
        ydot = np.gradient(yd, self.dt)
        zdot = np.gradient(zd, self.dt)
        
        xdot = self.smooth_derivatives(xdot, window=201, polyorder=2)
        ydot = self.smooth_derivatives(ydot, window=201, polyorder=2)
        zdot = self.smooth_derivatives(zdot, window=201, polyorder=2)
        
        xddot = np.gradient(xdot, self.dt)
        yddot = np.gradient(ydot, self.dt)
        zddot = np.gradient(zdot, self.dt)
        
        xddot = self.smooth_derivatives(xddot, window=151, polyorder=2)
        yddot = self.smooth_derivatives(yddot, window=151, polyorder=2)
        zddot = self.smooth_derivatives(zddot, window=151, polyorder=2)
        
        # Create DataFrame
        path_data = pd.DataFrame({
            'time': t,
            'xd': xd,
            'yd': yd,
            'zd': zd,
            'xdot': xdot,
            'ydot': ydot,
            'zdot': zdot,
            'xddot': xddot,
            'yddot': yddot,
            'zddot': zddot,
            'phase': phase
        })
        
        # COMPLEXITY-BASED TRUNCATION
        original_length = len(path_data)
        
        if self.complexity == 1:
            # Phase 1 only: vertical climb + small buffer
            phase_mask = path_data['phase'] <= 1.1
            last_idx = min(np.sum(phase_mask) + 200, len(path_data))  # Larger buffer
            path_data = path_data.iloc[:last_idx].copy()
            print(f"📏 Truncated to complexity 1: {len(path_data)} points (phase 1 only)")
            
        elif self.complexity == 2:
            # Phases 1-2: vertical + cruise (no transition between them)
            phase_mask = path_data['phase'] <= 2.1
            last_idx = min(np.sum(phase_mask) + 200, len(path_data))
            path_data = path_data.iloc[:last_idx].copy()
            print(f"📏 Truncated to complexity 2: {len(path_data)} points (phases 1-2)")
            
        else:
            # Complexity 3: all phases
            print(f"📏 Complexity 3: {len(path_data)} points (all phases)")
        
        # Final acceleration check on truncated data
        final_accelerations = np.sqrt(path_data['xddot']**2 + path_data['yddot']**2 + path_data['zddot']**2)
        final_max_accel = np.max(final_accelerations)
        
        print(f"✅ Final acceleration check: {final_max_accel:.3f} m/s² ")
        
        # Save the path
        path_data.to_csv(f'reference_path_complexity_{self.complexity}.csv', index=False)
        
        # Enhanced visualization
        self.visualize_path(path_data, original_length)
        
        return path_data
    
    def visualize_path(self, path_data, original_length=None):
        """Comprehensive path visualization"""
        import os
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        plt.close('all')  # Close any existing plots
        
        # 2D Position, Velocity, and Acceleration Plots
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        # Position Plots
        position_data = [
            ('X Position', 'xd', 'blue'),
            ('Y Position', 'yd', 'green'),
            ('Z Position', 'zd', 'red')
        ]
        
        for i, (title, column, color) in enumerate(position_data):
            axes[0, i].plot(path_data['time'], path_data[column], color=color, linewidth=2)
            axes[0, i].set_title(title)
            axes[0, i].set_xlabel('Time (s)')
            axes[0, i].set_ylabel('Position (m)')
            axes[0, i].grid(True)
        
        # Velocity Plots
        velocity_data = [
            ('X Velocity', 'xdot', 'blue'),
            ('Y Velocity', 'ydot', 'green'),
            ('Z Velocity', 'zdot', 'red')
        ]
        
        for i, (title, column, color) in enumerate(velocity_data):
            axes[1, i].plot(path_data['time'], path_data[column], color=color, linewidth=2)
            axes[1, i].set_title(title)
            axes[1, i].set_xlabel('Time (s)')
            axes[1, i].set_ylabel('Velocity (m/s)')
            axes[1, i].grid(True)
        
        # Acceleration Plots
        acceleration_data = [
            ('X Acceleration', 'xddot', 'blue'),
            ('Y Acceleration', 'yddot', 'green'),
            ('Z Acceleration', 'zddot', 'red')
        ]
        
        for i, (title, column, color) in enumerate(acceleration_data):
            axes[2, i].plot(path_data['time'], path_data[column], color=color, linewidth=2)
            axes[2, i].set_title(title)
            axes[2, i].set_xlabel('Time (s)')
            axes[2, i].set_ylabel('Acceleration (m/s²)')
            axes[2, i].grid(True)
        
        plt.tight_layout()
        # FIXED: Use proper file path
        plt.savefig(os.path.join('plots', 'reference_path_profiles.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3D Trajectory Plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color trajectory by phase
        phases = np.unique(np.floor(path_data['phase']))
        colors = ['blue', 'green', 'red']
        labels = ['Vertical', 'Cruise', 'Helical']
        
        for p in phases:
            phase_mask = np.floor(path_data['phase']) == p
            p_idx = int(p - 1)
            if p_idx < len(colors):
                ax.plot(
                    path_data.loc[phase_mask, 'xd'], 
                    path_data.loc[phase_mask, 'yd'], 
                    path_data.loc[phase_mask, 'zd'], 
                    color=colors[p_idx], 
                    linewidth=3, 
                    label=f'Phase {int(p)}: {labels[p_idx]}'
                )
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('3D Reference Trajectory')
        ax.legend()
        
        # FIXED: Use proper file path
        plt.savefig(os.path.join('plots', 'reference_path_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Phase Visualization
        plt.figure(figsize=(10, 5))
        plt.plot(path_data['time'], path_data['phase'], linewidth=2)
        plt.title('Reference Path Phases')
        plt.xlabel('Time (s)')
        plt.ylabel('Phase')
        plt.yticks([1, 2, 3], ['Vertical', 'Cruise', 'Helical'])
        plt.grid(True)
        # FIXED: Use proper file path
        plt.savefig(os.path.join('plots', 'reference_path_phases.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Reference path visualization complete:")
        print("  - plots/reference_path_profiles.png")
        print("  - plots/reference_path_3d.png")
        print("  - plots/reference_path_phases.png")

# Test the path generation
if __name__ == "__main__":
    for complexity_level in [1, 2, 3]:
        print(f"\n--- Generating reference path for complexity {complexity_level} ---")
        path_generator = ReferencePath(complexity=complexity_level)
        path_generator.generate_path()