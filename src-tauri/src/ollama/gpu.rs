//! GPU memory probing.
//!
//! Placement estimates were previously built on a guess — 8 GB total with 2 GB
//! reserved — and the guess was wrong in a way that changed routing decisions.
//! On the development machine it marked `qwen3-vl:8b` (5.7 GB) as spilled when
//! that model was measured running entirely on the GPU at 21 tok/s, which sent
//! image work to a much smaller model for no reason.
//!
//! So the budget is read from the driver instead. The reserve becomes what is
//! *actually* in use — the compositor, the browser, whatever else is running —
//! plus a small margin, rather than a number chosen in advance.

use super::registry::VramBudget;

const MIB: u64 = 1024 * 1024;

/// Margin left free on top of current usage.
///
/// Covers the desktop growing slightly while a model is loaded. Calibrated
/// against a measured case: `qwen3-vl:8b` occupies 6.3 GB and ran fully on an
/// 8151 MiB card with 1787 MiB already in use, leaving roughly 350 MiB spare.
/// A 512 MiB margin declared that model unusable; 256 MiB reports it as the
/// tight fit it actually is.
const HEADROOM: u64 = 256 * MIB;

/// Read total and in-use VRAM from the driver.
///
/// Returns `None` when no supported GPU tool is available — the caller keeps
/// its previous budget rather than substituting a worse guess.
pub fn probe() -> Option<VramBudget> {
    probe_nvidia()
}

fn probe_nvidia() -> Option<VramBudget> {
    let output = crate::proc::sync_command("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }
    parse_nvidia_smi(&String::from_utf8_lossy(&output.stdout))
}

/// Parse `nvidia-smi` CSV output.
///
/// Split out so the parsing is testable without a GPU present. Multi-GPU hosts
/// report one line per device; the largest card is the one a model would land
/// on, so that is the line used.
pub fn parse_nvidia_smi(text: &str) -> Option<VramBudget> {
    let best = text
        .lines()
        .filter_map(|line| {
            let mut parts = line.split(',');
            let total: u64 = parts.next()?.trim().parse().ok()?;
            let used: u64 = parts.next()?.trim().parse().ok()?;
            Some((total, used))
        })
        .max_by_key(|(total, _)| *total)?;

    let (total_mib, used_mib) = best;
    if total_mib == 0 {
        return None;
    }

    let total_bytes = total_mib * MIB;
    let reserved_bytes = (used_mib * MIB).saturating_add(HEADROOM).min(total_bytes);

    Some(VramBudget { total_bytes, reserved_bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * MIB;

    #[test]
    fn parses_a_single_card() {
        // The development machine's actual reading.
        let budget = parse_nvidia_smi("8151, 1787\n").expect("should parse");
        assert_eq!(budget.total_bytes, 8151 * MIB);
        assert_eq!(budget.reserved_bytes, 1787 * MIB + HEADROOM);
        // ~6.0 GB available, which the earlier fixed 2 GB reserve understated.
        assert!(budget.available() > 5 * GB);
    }

    #[test]
    fn picks_the_largest_card_on_a_multi_gpu_host() {
        let budget = parse_nvidia_smi("8151, 1787\n24576, 500\n").expect("should parse");
        assert_eq!(budget.total_bytes, 24576 * MIB);
    }

    #[test]
    fn tolerates_padded_and_noisy_output() {
        assert!(parse_nvidia_smi("  8151 ,  1787  ").is_some());
        assert!(parse_nvidia_smi("").is_none());
        assert!(parse_nvidia_smi("no devices were found").is_none());
        assert!(parse_nvidia_smi("0, 0").is_none());
    }

    #[test]
    fn reserved_never_exceeds_total() {
        // A nearly-full card must not produce a negative or wrapped budget.
        let budget = parse_nvidia_smi("8151, 8100").expect("should parse");
        assert!(budget.reserved_bytes <= budget.total_bytes);
        assert_eq!(budget.available(), 0);
    }

    #[test]
    fn a_model_measured_on_the_gpu_is_not_reported_as_spilled() {
        // qwen3-vl:8b is 5.7 GB and was measured at 21 tok/s entirely on the
        // GPU, so a budget that calls it spilled is wrong. It is reported as
        // Marginal rather than Resident, which is the honest answer: it fit
        // with roughly 350 MiB to spare and a browser would evict it.
        use super::super::registry::{estimate_placement, Placement};
        let budget = parse_nvidia_smi("8151, 1787").unwrap();
        let (placement, _) =
            estimate_placement(5_700_000_000, Some(65_536), Some(262_144), budget);
        assert_eq!(placement, Placement::Marginal);
    }

    #[test]
    fn a_comfortable_model_is_still_reported_as_resident() {
        // qwen2.5-coder:7b at 4.4 GB, measured at 42-57 tok/s.
        use super::super::registry::{estimate_placement, Placement};
        let budget = parse_nvidia_smi("8151, 1787").unwrap();
        let (placement, _) =
            estimate_placement(4_400_000_000, Some(65_536), Some(32_768), budget);
        assert_eq!(placement, Placement::Resident);
    }
}
