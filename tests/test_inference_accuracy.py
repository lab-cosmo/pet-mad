from pet_mad.calculator import PETMADCalculator
from ase.io import read
import pytest
import numpy as np

from . import DATASET_PATH_MAD_BENCH

VERSIONS = ("1.1.0", "1.0.1")
SUBSETS_NAMES = ['MAD', 'MPtrj', 'MatBench', 'Alexandria', 'OC2020', 'SPICE', 'MD22']

MAX_ERROR_FACTOR = 1.5

PET_MAD_INFERECE_ERRORS = {
    'MAD': [17.6, 65.1],
    'MPtrj': [22.3, 77.9],
    'MatBench': [31.3, 60.1],
    'Alexandria': [49.0, 66.8],
    'OC2020': [18.2, 116.0],
    'SPICE': [3.7, 59.5],
    'MD22': [1.9, 65.6],
}

def apply_bold_color(text, color):
    if color.lower() == 'red':
        return f"\033[1;31m{text}\033[0m"
    elif color.lower() == 'green':
        return f"\033[1;32m{text}\033[0m"
    elif color.lower() == 'yellow':
        return f"\033[1;33m{text}\033[0m"
    elif color.lower() == 'white':
        return f"\033[1;37m{text}\033[0m"

def get_performance_summary(subset_resolved_energy_errors, subset_resolved_force_errors, version):
    summary_lines = []
    summary_lines.append("\n")
    summary_lines.append(apply_bold_color("=" * 96, "white"))
    summary_lines.append(apply_bold_color(f"Performance Summary for PET-MAD v{version}", "white"))
    summary_lines.append(apply_bold_color("=" * 96, "white"))
    
    # Table header
    header = f"{'Subset':<12} {'Energy MAE':<15} {'Force MAE':<15} {'Target Energy MAE':<15} {'Target Force MAE':<15} {'Max Factor':<10} {'Status':<8}"
    summary_lines.append(apply_bold_color(header, "white"))
    summary_lines.append(
        apply_bold_color(
            f"{'':<12} {'(meV/atom)':<15} {'(meV/Å)':<15} {'(meV/atom)':<15} {'(meV/Å)':<15} {'':<10} {'':<8}",
            "white"
        )
    )
    summary_lines.append(apply_bold_color("-" * len(header), "white"))
    
    exceeded_threshold = False
    for subset in SUBSETS_NAMES:
        energy_mae = subset_resolved_energy_errors[subset]
        force_mae = subset_resolved_force_errors[subset]
        energy_threshold = PET_MAD_INFERECE_ERRORS[subset][0] * MAX_ERROR_FACTOR
        force_threshold = PET_MAD_INFERECE_ERRORS[subset][1] * MAX_ERROR_FACTOR
        
        threshold_exceeded = energy_mae > energy_threshold or force_mae > force_threshold
        exceeded_threshold = exceeded_threshold or threshold_exceeded
        
        target_energy = PET_MAD_INFERECE_ERRORS[subset][0]
        target_force = PET_MAD_INFERECE_ERRORS[subset][1]
        
        # Check individual thresholds
        energy_passed = energy_mae <= energy_threshold
        force_passed = force_mae <= force_threshold
        
        # Color coding: green for pass, red for fail
        if threshold_exceeded:
            status = apply_bold_color("FAIL", "red")
        else:
            status = apply_bold_color("PASS", "green")
        
        # Format individual values with conditional coloring and explicit default color resets
        if energy_passed:
            energy_mae_str = apply_bold_color(f"{energy_mae:<15.1f}", "green")
        else:
            energy_mae_str = apply_bold_color(f"{energy_mae:<15.1f}", "red")
            
        if force_passed:
            force_mae_str = apply_bold_color(f"{force_mae:<15.1f}", "green")
        else:
            force_mae_str = apply_bold_color(f"{force_mae:<15.1f}", "red")
        row = ""
        row += apply_bold_color(f"{subset:<12} {energy_mae_str} {force_mae_str} ", "white")
        row += apply_bold_color(f"{target_energy:<15.1f} {target_force:<15.1f} {MAX_ERROR_FACTOR:<10.1f}", "white")
        row += apply_bold_color(f"{status:<8}", "white")
        
        summary_lines.append(row)
    
    summary_lines.append(apply_bold_color("-" * len(header), "white"))
    summary_lines.append(apply_bold_color("=" * 96, "white"))
    
    return exceeded_threshold, "\n".join(summary_lines)

@pytest.mark.parametrize(
    "version",
    VERSIONS,
)
def test_inference_accuracy(version):
    calculator = PETMADCalculator(version=version, device="cpu")
    dataset = read(DATASET_PATH_MAD_BENCH, index=':')
    subset_resolved_energy_errors = {}
    subset_resolved_force_errors = {}
    for subset_name in SUBSETS_NAMES:
        subset = [item for item in dataset if item.info['dataset'] == subset_name]
        num_atoms = np.array([len(item) for item in subset])
        target_energies = np.array([item.get_potential_energy() for item in subset]) / num_atoms
        target_forces = np.concatenate([item.get_forces() for item in subset]).flatten()
        results = calculator.compute_energy(subset, compute_forces_and_stresses=True)
        
        predicted_energies = np.array(results["energy"]) / num_atoms
        predicted_forces = np.concatenate(results["forces"]).flatten()

        energy_mae = np.mean(np.abs(predicted_energies - target_energies)) * 1000
        force_mae = np.mean(np.abs(predicted_forces - target_forces)) * 1000

        subset_resolved_energy_errors[subset_name] = energy_mae
        subset_resolved_force_errors[subset_name] = force_mae
    
    exceeded_threshold, summary = get_performance_summary(
        subset_resolved_energy_errors, 
        subset_resolved_force_errors,
        version
    )
    
    if exceeded_threshold:
        pytest.fail(summary)