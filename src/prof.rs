//! Feature-gated profiling counters for the search hot path.
//!
//! Build with `--features prof` to enable; without the feature every
//! `prof_scope!` compiles to nothing. Counters are thread-local, so profile
//! with a single-threaded search (`benchmark -n 1`) and call
//! `prof::report()` from that same thread.
//!
//! Sections are nested (e.g. `apply_instructions` time is also counted
//! inside whatever section called it), so read the report as an inclusive
//! call-tree flattened into a table, not as disjoint buckets.

pub const NUM_SECTIONS: usize = 23;

pub mod sec {
    pub const CALIB: usize = 0;
    pub const SELECTION: usize = 1;
    pub const EXPAND: usize = 2;
    pub const ROLLOUT: usize = 3;
    pub const BACKPROP: usize = 4;
    pub const GET_OPTIONS: usize = 5;
    pub const GEN_INS: usize = 6;
    pub const APPLY: usize = 7;
    pub const REVERSE: usize = 8;
    pub const MOVES_FIRST: usize = 9;
    pub const GEN_MOVE: usize = 10;
    pub const GEN_SWITCH: usize = 11;
    pub const BEFORE_MOVE: usize = 12;
    pub const SPECIAL_EFFECT: usize = 13;
    pub const CALC_DAMAGE: usize = 14;
    pub const HIT_MISS: usize = 15;
    pub const RUN_MOVE: usize = 16;
    pub const FROM_DAMAGE: usize = 17;
    pub const SECONDARIES: usize = 18;
    pub const END_OF_TURN: usize = 19;
    pub const COMBINE_DUPES: usize = 20;
    pub const AFTER_MOVE_FINISH: usize = 21;
    pub const STATUS_CONDS: usize = 22;
}

pub static SECTION_NAMES: [&str; NUM_SECTIONS] = [
    "calibration",
    "selection (ucb + apply down path)",
    "expand (geninstr + nodes + sample)",
    "rollout (evaluate)",
    "backprop (scores + reverse up path)",
    "> get_all_options (in selection)",
    "> generate_ins_from_move_pair",
    ">> state.apply_instructions (all)",
    ">> state.reverse_instructions (all)",
    ">> moves_first",
    ">> generate_ins_from_move",
    ">>> gen_ins_from_switch",
    ">>> before_move",
    ">>> choice_special_effect",
    ">>> calculate_damage",
    ">>> check_move_hit_or_miss",
    ">>> run_move",
    ">>>> gen_ins_from_damage",
    ">>>> get_ins_from_secondaries",
    ">> add_end_of_turn_instructions",
    ">> combine_duplicate_instructions",
    ">> after_move_finish",
    ">>> existing_status_conditions",
];

#[cfg(feature = "prof")]
mod imp {
    use super::{sec, NUM_SECTIONS, SECTION_NAMES};
    use std::cell::Cell;
    use std::time::{Duration, Instant};

    thread_local! {
        static CYCLES: [Cell<u64>; NUM_SECTIONS] = std::array::from_fn(|_| Cell::new(0));
        static COUNTS: [Cell<u64>; NUM_SECTIONS] = std::array::from_fn(|_| Cell::new(0));
    }

    #[inline(always)]
    fn rdtsc() -> u64 {
        unsafe { core::arch::x86_64::_rdtsc() }
    }

    pub struct ProfScope {
        section: usize,
        start: u64,
    }

    impl ProfScope {
        #[inline(always)]
        pub fn new(section: usize) -> ProfScope {
            ProfScope {
                section,
                start: rdtsc(),
            }
        }
    }

    impl Drop for ProfScope {
        #[inline(always)]
        fn drop(&mut self) {
            let elapsed = rdtsc().wrapping_sub(self.start);
            CYCLES.with(|c| {
                let cell = &c[self.section];
                cell.set(cell.get() + elapsed);
            });
            COUNTS.with(|c| {
                let cell = &c[self.section];
                cell.set(cell.get() + 1);
            });
        }
    }

    /// Cycles of overhead added by one ProfScope (rdtsc x2 + two TLS adds).
    fn calibrate() -> f64 {
        const N: u64 = 1_000_000;
        let start = rdtsc();
        for _ in 0..N {
            let _s = ProfScope::new(sec::CALIB);
        }
        let total = rdtsc().wrapping_sub(start);
        total as f64 / N as f64
    }

    fn tsc_hz() -> f64 {
        let t0 = Instant::now();
        let c0 = rdtsc();
        while t0.elapsed() < Duration::from_millis(200) {
            std::hint::spin_loop();
        }
        let c1 = rdtsc();
        (c1.wrapping_sub(c0)) as f64 / t0.elapsed().as_secs_f64()
    }

    /// Print per-section totals for the calling thread. `self` cost of one
    /// scope is subtracted from its own total, but a parent section still
    /// includes the overhead of scopes nested inside it — treat small
    /// sections with huge call counts with suspicion.
    pub fn report() {
        let per_scope = calibrate();
        let hz = tsc_hz();
        let total_events: u64 = COUNTS.with(|c| c.iter().map(|x| x.get()).sum());
        println!("\n--- prof report ---");
        println!(
            "scope self-overhead: {:.1} cycles | tsc: {:.2} GHz | total scope events: {} (~{:.0}ms total overhead)",
            per_scope,
            hz / 1e9,
            total_events,
            total_events as f64 * per_scope / hz * 1e3,
        );
        println!(
            "{:<38} {:>13} {:>10} {:>11}",
            "section", "calls", "ms(corr)", "cyc/call"
        );
        CYCLES.with(|cy| {
            COUNTS.with(|ct| {
                for i in 0..NUM_SECTIONS {
                    let calls = ct[i].get();
                    if calls == 0 || i == sec::CALIB {
                        continue;
                    }
                    let raw = cy[i].get() as f64;
                    let corr = (raw - calls as f64 * per_scope).max(0.0);
                    println!(
                        "{:<38} {:>13} {:>10.1} {:>11.0}",
                        SECTION_NAMES[i],
                        calls,
                        corr / hz * 1e3,
                        corr / calls as f64
                    );
                }
            })
        });
    }
}

#[cfg(feature = "prof")]
pub use imp::*;

#[macro_export]
macro_rules! prof_scope {
    ($section:expr) => {
        #[cfg(feature = "prof")]
        let _prof_scope_guard = $crate::prof::ProfScope::new($section);
    };
}
