using System;
using System.Collections.Generic;
using System.IO;
using ElevatorRL.Stats;
using UnityEngine;
using UnityEditor;

namespace ElevatorRL.Editor
{
    /// <summary>
    /// Headless evaluation runner. Runs deterministic episodes under a chosen dispatcher against
    /// a BuildingConfig (inline or a scale-ladder preset asset), drives a StatsCollector, and
    /// writes the three canonical CSV tables. RunSweep is the E1 baseline-characterization driver
    /// (EXPERIMENT_PLAN.md §4): every (preset × dispatcher × pattern × intensity × seed) cell,
    /// aggregated to a summary CSV.
    /// </summary>
    public static class EvalHarness
    {
        /// <summary>A dispatch policy: given the live Building, return one action per car.</summary>
        public delegate int[] Dispatcher(Building b);

        /// <summary>
        /// Builds a FRESH dispatcher for one episode. Required because EtaHeuristic is stateful
        /// (sticky assignments) — reusing one instance across episodes would leak state between
        /// runs. CollectiveLook is stateless and safely reused.
        /// </summary>
        public delegate Dispatcher DispatcherFactory(int numElevators);

        /// <summary>The three implemented baselines (EXPERIMENT_PLAN.md §1), in report order.</summary>
        public static readonly (string name, DispatcherFactory factory)[] Policies =
        {
            ("LOOK",         numCars => ElevatorHeuristics.CollectiveLook),
            ("ETA",          numCars => new EtaHeuristic(numCars).Dispatch),
            ("ETA-Weighted", numCars => new EtaHeuristic(numCars, weightByQueueDepth: true).Dispatch),
        };

        const string PresetFolder = "Assets/ElevatorRL/Config/Presets";

        // NOTE: intensity=1.0 saturates the S preset (8fl/3cars) under UpPeak (~41% rejection —
        // floor-0 arrival rate already exceeds fleet throughput capacity there). Use ~0.5 for a
        // representative, non-saturated comparison; sweep intensity deliberately for the
        // saturation-curve experiments (EXPERIMENT_PLAN.md E1/E3).
        const float SmokeIntensity = 0.5f;

        // ── Ad hoc smoke tests (inline config, no preset asset needed) ─────────

        [MenuItem("Tools/Elevator RL/Run Eval (LOOK smoke test)")]
        static void RunLookSmokeTest()
        {
            var (episode, dir) = RunSingle("LOOK", ElevatorHeuristics.CollectiveLook, "S-smoke",
                8, 3, 8, TrafficPattern.UpPeak, SmokeIntensity, 1, 3600f, 300f, 300f);
            Debug.Log("[Eval] " + Summarize(episode, dir));
        }

        [MenuItem("Tools/Elevator RL/Run Eval (ETA smoke test)")]
        static void RunEtaSmokeTest()
        {
            var eta = new EtaHeuristic(3);
            var (episode, dir) = RunSingle("ETA", eta.Dispatch, "S-smoke",
                8, 3, 8, TrafficPattern.UpPeak, SmokeIntensity, 1, 3600f, 300f, 300f);
            Debug.Log("[Eval] " + Summarize(episode, dir));
        }

        [MenuItem("Tools/Elevator RL/Run Eval (ETA-weighted smoke test)")]
        static void RunEtaWeightedSmokeTest()
        {
            var eta = new EtaHeuristic(3, weightByQueueDepth: true);
            var (episode, dir) = RunSingle("ETA-Weighted", eta.Dispatch, "S-smoke",
                8, 3, 8, TrafficPattern.UpPeak, SmokeIntensity, 1, 3600f, 300f, 300f);
            Debug.Log("[Eval] " + Summarize(episode, dir));
        }

        // Headless validation of the AS1 target-floor sub-controller (EXPERIMENT_PLAN.md §7).
        // Drives the sim with LOOK-chosen targets resolved through TargetFloorControl — the same
        // deterministic sub-controller the RL agent uses in AS1 mode — so we can confirm the
        // target→primitive resolution reproduces LOOK-like behavior WITHOUT entering Play mode.
        // NOTE: this recomputes targets every decision (like LOOK), so it does NOT exercise AS1's
        // commitment-persistence / sparse-decision semantics — that part only lives in the agent
        // and needs a Play-mode run to validate. This validates the resolve/target mechanics only.
        static int[] TargetFloorLookDispatch(Building b)
        {
            var targets = TargetFloorControl.LookTargets(b);
            var act = new int[b.cfg.numElevators];
            for (int i = 0; i < b.cfg.numElevators; i++)
                act[i] = TargetFloorControl.ResolveTowardTarget(b, i, targets[i]);
            return act;
        }

        [MenuItem("Tools/Elevator RL/Validate AS1 (LOOK vs TargetFloorLook, same seed)")]
        static void ValidateTargetFloorMechanics()
        {
            const int seed = 1;
            var (look, lookDir) = RunSingle("LOOK", ElevatorHeuristics.CollectiveLook, "S-smoke",
                8, 3, 8, TrafficPattern.UpPeak, SmokeIntensity, seed, 3600f, 300f, 300f);
            var (tfl, tflDir) = RunSingle("TargetFloorLook", TargetFloorLookDispatch, "S-smoke",
                8, 3, 8, TrafficPattern.UpPeak, SmokeIntensity, seed, 3600f, 300f, 300f);

            Debug.Log("[Eval] " + Summarize(look, lookDir));
            Debug.Log("[Eval] " + Summarize(tfl, tflDir));
            Debug.Log($"[Eval] TargetFloorLook vs LOOK (same seed={seed}): delivered " +
                      $"{look.delivered}→{tfl.delivered}, waitP95 {look.waitP95:0.0}→{tfl.waitP95:0.0}s, " +
                      $"abandoned {look.abandoned}→{tfl.abandoned} — expect broadly similar " +
                      "(same target logic, slightly different resolve).");
        }

        // EXPERIMENT_PLAN.md E2: does the trained rung-S PPO policy (elev-e2-s-ppo-01, 5M steps)
        // match or beat LOOK/ETA under the exact same seed/traffic used for the E1 baselines?
        // Runs all three through the identical RunSingle/RunCore loop (Building + StatsCollector) —
        // PPO's actions come from PpoDispatcher, which reuses the real observation/mask code so
        // this is not a reimplementation of what the agent sees, just a different action source.
        [MenuItem("Tools/Elevator RL/Run E2 Comparison (LOOK vs ETA vs PPO, rung S)")]
        static void RunE2Comparison()
        {
            const int seed = 1;
            const float totalSeconds = 3600f, warmup = 300f, bucket = 300f;
            const int floors = 8, cars = 3, capacity = 8;
            const TrafficPattern pattern = TrafficPattern.UpPeak;

            var (look, lookDir) = RunSingle("LOOK", ElevatorHeuristics.CollectiveLook, "S",
                floors, cars, capacity, pattern, SmokeIntensity, seed, totalSeconds, warmup, bucket);

            var etaHeuristic = new EtaHeuristic(cars);
            var (eta, etaDir) = RunSingle("ETA", etaHeuristic.Dispatch, "S",
                floors, cars, capacity, pattern, SmokeIntensity, seed, totalSeconds, warmup, bucket);

            const string modelPath = "Assets/ElevatorRL/Models/ElevatorController_S_e2.onnx";
            var modelAsset = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(modelPath);
            if (modelAsset == null)
            {
                Debug.LogError($"[Eval] Could not load ModelAsset at {modelPath} — " +
                    "select it in the Project window and check the Inspector imports it as a Model " +
                    "(right-click > Reimport if it shows as a plain file).");
                return;
            }

            // obsSize/branch width must match training exactly — Training.unity's baked
            // BrainParameters for S_BuildingConfig (8 floors, 3 cars, Primitive/AS0): 98.
            using var ppo = new PpoDispatcher(modelAsset, obsSize: 98);
            var (ppoEp, ppoDir) = RunSingle("PPO", ppo.Dispatch, "S",
                floors, cars, capacity, pattern, SmokeIntensity, seed, totalSeconds, warmup, bucket);

            Debug.Log("[Eval] " + Summarize(look, lookDir));
            Debug.Log("[Eval] " + Summarize(eta, etaDir));
            Debug.Log("[Eval] " + Summarize(ppoEp, ppoDir));
        }

        // Multi-seed follow-up to RunE2Comparison: is the single-seed PPO win real, or a draw of
        // that particular seed? Same traffic/rung, seeds 1-5, one aggregate CSV under Runs/ so the
        // per-seed numbers are logged and reproducible, not just Console output.
        [MenuItem("Tools/Elevator RL/Run E2 Sweep (LOOK vs ETA vs PPO, rung S, seeds 1-5)")]
        static void RunE2SweepSeeds() => RunScaleLadderSweep("S", 8, 3, 8,
            "Assets/ElevatorRL/Models/ElevatorController_S_e2.onnx", obsSize: 98);

        [MenuItem("Tools/Elevator RL/Run E3 Sweep (LOOK vs ETA vs PPO, rung M, seeds 1-5)")]
        static void RunE3SweepM() => RunScaleLadderSweep("M", 16, 5, 8,
            "Assets/ElevatorRL/Models/ElevatorController_M_e3.onnx", obsSize: 254);

        // elev-e3-m-ppo-01 resumed from 5M -> 10M steps (same recipe) to test whether M's
        // still-climbing reward curve was "hasn't finished training" rather than a hard
        // architecture ceiling — see EXPERIMENT_PLAN.md E3.
        [MenuItem("Tools/Elevator RL/Run E3 Sweep (LOOK vs ETA vs PPO, rung M, 10M steps, seeds 1-5)")]
        static void RunE3SweepM10M() => RunScaleLadderSweep("M-10M", 16, 5, 8,
            "Assets/ElevatorRL/Models/ElevatorController_M_e3_10m.onnx", obsSize: 254);

        // EXPERIMENT_PLAN.md E6 Architecture A (multi-agent parameter sharing): same protocol as the
        // flat-MLP E3 M sweeps above, but the shared per-car policy runs through
        // MultiAgentPpoDispatcher. obsSize here is the PER-CAR observation size (CarObservationSize),
        // not the whole-fleet size — for M that's 102 (see MultiAgentSetup log / TrainingMultiAgent.unity).
        [MenuItem("Tools/Elevator RL/E6 Multi-Agent/Run Sweep (LOOK vs ETA vs PPO-multi, rung M, seeds 1-5)")]
        static void RunE6ASweepM() => RunScaleLadderSweep("M-e6a", 16, 5, 8,
            "Assets/ElevatorRL/Models/ElevatorController_M_e6a.onnx", obsSize: 102, multiAgent: true);

        // EXPERIMENT_PLAN.md E3: same LOOK/ETA/PPO comparison as E2, generalized across the scale
        // ladder. NOTE intensity is still the fixed SmokeIntensity (0.5), matching E2's methodology
        // for apples-to-apples continuity — NOT each rung's calibrated saturation point (§3: S≈1.33,
        // M≈0.41, ...), so cross-rung comparisons here are at different *relative* loads. Fine for
        // "does PPO beat LOOK on this rung" but not yet the calibrated-intensity headline figure.
        // multiAgent=true uses the E6-A per-car shared policy (MultiAgentPpoDispatcher) and treats
        // obsSize as the per-car observation size; false uses the flat single-agent PpoDispatcher.
        static void RunScaleLadderSweep(string rungName, int floors, int cars, int capacity,
            string modelPath, int obsSize, bool multiAgent = false)
        {
            const float totalSeconds = 3600f, warmup = 300f, bucket = 300f;
            const TrafficPattern pattern = TrafficPattern.UpPeak;
            int[] seeds = { 1, 2, 3, 4, 5 };

            var modelAsset = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(modelPath);
            if (modelAsset == null)
            {
                Debug.LogError($"[Eval] Could not load ModelAsset at {modelPath}.");
                return;
            }
            var ppoMulti = multiAgent ? new MultiAgentPpoDispatcher(modelAsset, obsSize) : null;
            var ppoFlat = multiAgent ? null : new PpoDispatcher(modelAsset, obsSize);
            Dispatcher ppoDispatch = multiAgent ? ppoMulti.Dispatch : ppoFlat.Dispatch;

            var rows = new List<string> {
                "policy,seed,delivered,waitMean,waitP95,waitMax,abandoned,rejected,util,rwTotal"
            };

            foreach (var seed in seeds)
            {
                var (look, _) = RunSingle("LOOK", ElevatorHeuristics.CollectiveLook, rungName,
                    floors, cars, capacity, pattern, SmokeIntensity, seed, totalSeconds, warmup, bucket, quiet: true);
                var etaHeuristic = new EtaHeuristic(cars);
                var (eta, _) = RunSingle("ETA", etaHeuristic.Dispatch, rungName,
                    floors, cars, capacity, pattern, SmokeIntensity, seed, totalSeconds, warmup, bucket, quiet: true);
                var (ppoEp, _) = RunSingle("PPO", ppoDispatch, rungName,
                    floors, cars, capacity, pattern, SmokeIntensity, seed, totalSeconds, warmup, bucket, quiet: true);

                foreach (var (name, e) in new[] { ("LOOK", look), ("ETA", eta), ("PPO", ppoEp) })
                    rows.Add($"{name},{seed},{e.delivered},{e.waitMean:0.00},{e.waitP95:0.00},{e.waitMax:0.00}," +
                             $"{e.abandoned},{e.rejected},{e.utilFleetMean:0.000},{e.rwTotal:0}");

                Debug.Log($"[Eval] rung={rungName} seed={seed} LOOK delivered={look.delivered} waitMean={look.waitMean:0.0}s | " +
                          $"ETA delivered={eta.delivered} waitMean={eta.waitMean:0.0}s | " +
                          $"PPO delivered={ppoEp.delivered} waitMean={ppoEp.waitMean:0.0}s");
            }

            ppoMulti?.Dispose();
            ppoFlat?.Dispose();

            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string outDir = Path.Combine(projectRoot, "Runs", $"{DateTime.Now:yyyyMMdd-HHmmss}-E3-sweep-{rungName}-UpPeak");
            StatsCsv.Write(Path.Combine(outDir, "sweep_summary.csv"), rows[0], rows.GetRange(1, rows.Count - 1));
            Debug.Log($"[Eval] {rungName} sweep complete — {outDir}/sweep_summary.csv");
        }

        [MenuItem("Tools/Elevator RL/Generate wait_hist demo (L UpPeak, LOOK+ETA)")]
        static void GenWaitHistDemo()
        {
            var presets = LoadPresets(new[] { "L" });
            if (presets == null) return;
            var L = presets["L"];
            RunWithPreset("LOOK", ElevatorHeuristics.CollectiveLook, "L", L,
                TrafficPattern.UpPeak, 0.376f, 1, 3600f, 300f, 300f);
            var eta = new EtaHeuristic(L.numElevators);
            RunWithPreset("ETA", eta.Dispatch, "L", L,
                TrafficPattern.UpPeak, 0.376f, 1, 3600f, 300f, 300f);
            Debug.Log("[Eval] wait_hist demo written — L / UpPeak / i0.376, LOOK + ETA");
        }

        [MenuItem("Tools/Elevator RL/Run Comparison (LOOK vs ETA vs ETA-weighted, same seed)")]
        static void RunComparisonUpPeak() => RunComparison(TrafficPattern.UpPeak);

        [MenuItem("Tools/Elevator RL/Run Comparison - Lunch pattern (LOOK vs ETA vs ETA-weighted)")]
        static void RunComparisonLunch() => RunComparison(TrafficPattern.Lunch);

        // Lunch has TWO elevated-demand floors (lobby AND top floor), spread apart in index —
        // unlike UpPeak, where the lobby is both index-0 and the dominant queue, so ascending-
        // index order already coincides with descending-queue-depth order and the weighted
        // variant can never differ from pure ETA. Lunch is the real test of the weighting.
        static void RunComparison(TrafficPattern pattern)
        {
            const int seed = 1;
            var runs = new List<(string policy, EpisodeStats stats, string dir)>();

            var (look, lookDir) = RunSingle("LOOK", ElevatorHeuristics.CollectiveLook, "S-smoke",
                8, 3, 8, pattern, SmokeIntensity, seed, 3600f, 300f, 300f);
            runs.Add(("LOOK", look, lookDir));

            var eta = new EtaHeuristic(3); // stateful (sticky assignments) — one instance per episode
            var (etaStats, etaDir) = RunSingle("ETA", eta.Dispatch, "S-smoke",
                8, 3, 8, pattern, SmokeIntensity, seed, 3600f, 300f, 300f);
            runs.Add(("ETA", etaStats, etaDir));

            var etaW = new EtaHeuristic(3, weightByQueueDepth: true);
            var (etaWStats, etaWDir) = RunSingle("ETA-Weighted", etaW.Dispatch, "S-smoke",
                8, 3, 8, pattern, SmokeIntensity, seed, 3600f, 300f, 300f);
            runs.Add(("ETA-Weighted", etaWStats, etaWDir));

            foreach (var r in runs) Debug.Log("[Eval] " + Summarize(r.stats, r.dir));

            var baseline = look;
            var sb = new System.Text.StringBuilder();
            sb.Append($"[Eval] vs LOOK (pattern={pattern}, seed={seed}, intensity={SmokeIntensity}):");
            foreach (var r in runs)
            {
                if (r.policy == "LOOK") continue;
                sb.Append($"\n  {r.policy}: waitP95 {baseline.waitP95:0.0}s→{r.stats.waitP95:0.0}s ({Pct(baseline.waitP95, r.stats.waitP95)}), " +
                          $"abandoned {baseline.abandoned}→{r.stats.abandoned} ({Pct(baseline.abandoned, r.stats.abandoned)}), " +
                          $"delivered {baseline.delivered}→{r.stats.delivered}");
            }
            Debug.Log(sb.ToString());
        }

        static string Pct(float baseline, float value) =>
            baseline > 1e-6f ? $"{(value - baseline) / baseline * 100f:+0.0;-0.0}%" : "n/a";

        static string Summarize(EpisodeStats e, string runDir) =>
            $"{e.id.runId}: delivered={e.delivered} " +
            $"waitMean={e.waitMean:0.0}s p95={e.waitP95:0.0}s max={e.waitMax:0.0}s " +
            $"abandoned={e.abandoned} rejected={e.rejected} " +
            $"util={e.utilFleetMean:0.00} rwTotal={e.rwTotal:0} → {runDir}";

        // ── E1 sweep: preset assets × policies × patterns × intensities × seeds ─

        [MenuItem("Tools/Elevator RL/Run E1 Sweep - Quick (all rungs, sanity check)")]
        static void RunSweepQuick()
        {
            // Smallest sweep that still touches every rung and every policy: proves the pipeline
            // (esp. Z/H's floorRange + H's variable fleet) runs cleanly everywhere before
            // committing to the full multi-pattern/intensity/seed sweep below.
            RunSweep(
                presetNames: new[] { "S", "M", "L", "Z", "H" },
                patterns: new[] { TrafficPattern.UpPeak },
                intensities: new[] { SmokeIntensity },
                seeds: new[] { 1 },
                totalSeconds: 1200f, warmupSeconds: 120f, bucketSeconds: 300f,
                sweepName: "E1-quick");
        }

        [MenuItem("Tools/Elevator RL/Run E1 Sweep - Full (S-M-L-Z-H x patterns x intensities x seeds)")]
        static void RunSweepFull()
        {
            // NOTE: this is a large, long-running synchronous call (5 rungs x 3 policies x 3
            // patterns x 2 intensities x 3 seeds = 270 episodes at 1 simulated hour each) — it
            // will make the Editor unresponsive for several minutes. Prefer the Quick sweep for
            // interactive iteration; run Full when you're ready to let it churn.
            RunSweep(
                presetNames: new[] { "S", "M", "L", "Z", "H" },
                patterns: new[] { TrafficPattern.UpPeak, TrafficPattern.DownPeak, TrafficPattern.Lunch },
                intensities: new[] { 0.5f, 1.0f },
                seeds: new[] { 1, 2, 3 },
                totalSeconds: 3600f, warmupSeconds: 300f, bucketSeconds: 300f,
                sweepName: "E1-full");
        }

        static Dictionary<string, BuildingConfig> LoadPresets(string[] presetNames)
        {
            var presets = new Dictionary<string, BuildingConfig>();
            foreach (var name in presetNames)
            {
                var path = $"{PresetFolder}/{name}_BuildingConfig.asset";
                var cfg = AssetDatabase.LoadAssetAtPath<BuildingConfig>(path);
                if (cfg == null)
                {
                    Debug.LogError($"[Eval] Preset not found at {path} — run " +
                                    "Tools/Elevator RL/Generate Scale Ladder Presets first.");
                    return null;
                }
                presets[name] = cfg;
            }
            return presets;
        }

        static void RunSweep(string[] presetNames, TrafficPattern[] patterns, float[] intensities,
            int[] seeds, float totalSeconds, float warmupSeconds, float bucketSeconds, string sweepName)
        {
            var presets = LoadPresets(presetNames);
            if (presets == null) return;

            var summaryRows = new List<string>();
            int cellCount = presetNames.Length * Policies.Length * patterns.Length * intensities.Length;
            int cell = 0, totalEpisodes = 0;

            foreach (var presetName in presetNames)
            {
                var presetAsset = presets[presetName];
                foreach (var (policyName, factory) in Policies)
                foreach (var pattern in patterns)
                foreach (var intensity in intensities)
                {
                    cell++;
                    var cellEpisodes = new List<EpisodeStats>(seeds.Length);
                    foreach (var seed in seeds)
                    {
                        var dispatch = factory(presetAsset.numElevators);
                        var (episode, _) = RunWithPreset(policyName, dispatch, presetName, presetAsset,
                            pattern, intensity, seed, totalSeconds, warmupSeconds, bucketSeconds);
                        cellEpisodes.Add(episode);
                        totalEpisodes++;
                    }
                    summaryRows.Add(SummaryRow(presetName, policyName, pattern, intensity,
                        calibratedBase: intensity, intensityMultiplier: 1f, cellEpisodes));
                }
            }

            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string sweepDir = Path.Combine(projectRoot, "Runs", $"{DateTime.Now:yyyyMMdd-HHmmss}-{sweepName}");
            StatsCsv.Write(Path.Combine(sweepDir, "summary.csv"), SummaryHeader, summaryRows);

            Debug.Log($"[Eval] Sweep '{sweepName}' complete: {cell} cells, {totalEpisodes} episodes → " +
                      $"{Path.Combine(sweepDir, "summary.csv")}");
        }

        // ── E1 sweep, calibrated: per (preset, pattern), bisect a base intensity against LOOK ──

        [MenuItem("Tools/Elevator RL/Run E1 Sweep - Calibrated (auto-tuned intensity per rung x pattern)")]
        static void RunSweepCalibratedQuick()
        {
            // calibLo=0.005 (not 0.05): Z/H's true crossing points sit far lower than S/M/L's
            // (~0.006-0.016 vs ~0.1-1.1) because zoned dispatch has much lower effective capacity
            // per unit of nominal intensity — a 0.05 floor was pinning both at a false, degenerate
            // "floor" value rather than their real (much lower) calibration point.
            RunSweepCalibrated(
                presetNames: new[] { "S", "M", "L", "Z", "H" },
                patterns: new[] { TrafficPattern.UpPeak, TrafficPattern.Lunch },
                multiples: new[] { 0.5f, 1.0f, 1.5f },
                seeds: new[] { 1, 2 },
                targetAbandonRate: 0.10f,
                calibIterations: 10, calibLo: 0.005f, calibHi: 4f,
                totalSeconds: 1200f, warmupSeconds: 120f, bucketSeconds: 300f,
                sweepName: "E1-calibrated");
        }

        [MenuItem("Tools/Elevator RL/Diagnose Z-H Calibration Floor (widened search, no full sweep)")]
        static void DiagnoseZHCalibrationFloor()
        {
            // Prior calibration runs pinned at calibLo=0.05 for both Z and H — the bisection
            // math (mid after 8 halvings from lo=0.05 is exactly 0.0654) proves the 10%-abandon
            // target was never met even at the search floor. This checks whether a real crossing
            // point exists further down (lo=0.005) or whether Z/H can't reach 10% abandonment at
            // any meaningful intensity given their current car allocation.
            var presets = LoadPresets(new[] { "Z", "H" });
            if (presets == null) return;

            foreach (var name in new[] { "Z", "H" })
            {
                var asset = presets[name];
                float baseIntensity = CalibrateIntensity(asset, TrafficPattern.UpPeak, seed: 1,
                    targetAbandonRate: 0.10f, lo: 0.005f, hi: 4f, iterations: 12,
                    totalSeconds: 1200f, warmupSeconds: 120f);

                // probe the actual abandonRate AT the returned calibration point (quiet run) so we
                // can see whether it's a genuine crossing or another pinned floor.
                var (probe, _) = RunWithPreset("LOOK", ElevatorHeuristics.CollectiveLook, "diag",
                    asset, TrafficPattern.UpPeak, baseIntensity, seed: 1,
                    totalSeconds: 1200f, warmupSeconds: 120f, bucketSeconds: 1200f, quiet: true);

                Debug.Log($"[Eval] {name}/UpPeak: calibrated intensity={baseIntensity:F4} → " +
                          $"abandonRate={probe.abandonRate:P1} (target 10%) " +
                          $"[lo floor was 0.005; pinned-at-floor value would be ≈0.00654]");
            }
        }

        /// <summary>
        /// Finds, via bisection against LOOK, the intensity at which abandonment rate ≈
        /// targetAbandonRate for this preset+pattern — so different scale-ladder rungs are
        /// compared at a comparable RELATIVE load instead of an identical raw intensity value
        /// (which, even after PassengerArrivals' hub-floor scaling fix, may not land every rung
        /// at the same saturation point, since capacity also depends on floor count, car count,
        /// and zoning). Runs quiet (no CSV writes) — only the summary row for the real sweep at
        /// each multiple of the calibrated base is persisted.
        /// </summary>
        static float CalibrateIntensity(BuildingConfig presetAsset, TrafficPattern pattern, int seed,
            float targetAbandonRate, float lo, float hi, int iterations,
            float totalSeconds, float warmupSeconds)
        {
            float mid = lo;
            for (int iter = 0; iter < iterations; iter++)
            {
                mid = (lo + hi) / 2f;
                var (episode, _) = RunWithPreset("LOOK", ElevatorHeuristics.CollectiveLook, "calib",
                    presetAsset, pattern, mid, seed, totalSeconds, warmupSeconds,
                    bucketSeconds: totalSeconds, quiet: true);
                if (episode.abandonRate > targetAbandonRate) hi = mid; else lo = mid;
            }
            return mid;
        }

        static void RunSweepCalibrated(string[] presetNames, TrafficPattern[] patterns, float[] multiples,
            int[] seeds, float targetAbandonRate, int calibIterations, float calibLo, float calibHi,
            float totalSeconds, float warmupSeconds, float bucketSeconds, string sweepName)
        {
            var presets = LoadPresets(presetNames);
            if (presets == null) return;

            var summaryRows = new List<string>();
            int cell = 0, totalEpisodes = 0;
            int calibSeed = seeds.Length > 0 ? seeds[0] : 1;

            foreach (var presetName in presetNames)
            {
                var presetAsset = presets[presetName];
                foreach (var pattern in patterns)
                {
                    float baseIntensity = CalibrateIntensity(presetAsset, pattern, calibSeed,
                        targetAbandonRate, calibLo, calibHi, calibIterations, totalSeconds, warmupSeconds);
                    Debug.Log($"[Eval] Calibrated {presetName}/{pattern}: base intensity = " +
                              $"{baseIntensity:F3} (target abandonRate={targetAbandonRate:P0}, LOOK, seed={calibSeed})");

                    foreach (var multiple in multiples)
                    {
                        float intensity = baseIntensity * multiple;
                        foreach (var (policyName, factory) in Policies)
                        {
                            cell++;
                            var cellEpisodes = new List<EpisodeStats>(seeds.Length);
                            foreach (var seed in seeds)
                            {
                                var dispatch = factory(presetAsset.numElevators);
                                var (episode, _) = RunWithPreset(policyName, dispatch, presetName,
                                    presetAsset, pattern, intensity, seed,
                                    totalSeconds, warmupSeconds, bucketSeconds);
                                cellEpisodes.Add(episode);
                                totalEpisodes++;
                            }
                            summaryRows.Add(SummaryRow(presetName, policyName, pattern, intensity,
                                baseIntensity, multiple, cellEpisodes));
                        }
                    }
                }
            }

            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string sweepDir = Path.Combine(projectRoot, "Runs", $"{DateTime.Now:yyyyMMdd-HHmmss}-{sweepName}");
            StatsCsv.Write(Path.Combine(sweepDir, "summary.csv"), SummaryHeader, summaryRows);

            Debug.Log($"[Eval] Calibrated sweep '{sweepName}' complete: {cell} cells, {totalEpisodes} " +
                      $"episodes → {Path.Combine(sweepDir, "summary.csv")}");
        }

        const string SummaryHeader =
            "preset,policy,pattern,intensity,calibratedBase,intensityMultiplier,seeds,delivered_mean," +
            "waitMean_mean,waitP95_mean,waitMax_mean,abandoned_mean,rejected_mean,abandonRate_mean," +
            "rwTotal_mean,utilFleetMean_mean";

        static string SummaryRow(string preset, string policy, TrafficPattern pattern, float intensity,
            float calibratedBase, float intensityMultiplier, List<EpisodeStats> episodes)
        {
            int n = episodes.Count;
            float delivered = 0, waitMean = 0, waitP95 = 0, waitMax = 0, abandoned = 0, rejected = 0,
                  abandonRate = 0, rwTotal = 0, util = 0;
            foreach (var e in episodes)
            {
                delivered += e.delivered; waitMean += e.waitMean; waitP95 += e.waitP95;
                waitMax += e.waitMax; abandoned += e.abandoned; rejected += e.rejected;
                abandonRate += e.abandonRate; rwTotal += e.rwTotal; util += e.utilFleetMean;
            }
            return $"{preset},{policy},{pattern},{intensity},{calibratedBase:F3},{intensityMultiplier:F2},{n}," +
                   $"{delivered / n:F2},{waitMean / n:F3},{waitP95 / n:F3},{waitMax / n:F3}," +
                   $"{abandoned / n:F2},{rejected / n:F2},{abandonRate / n:F4},{rwTotal / n:F1},{util / n:F4}";
        }

        /// <summary>Runs one episode against a preset BuildingConfig asset (cloned, never dirtied).</summary>
        public static (EpisodeStats episode, string runDir) RunWithPreset(string policy, Dispatcher dispatch,
            string presetName, BuildingConfig presetAsset,
            TrafficPattern pattern, float intensity, int seed,
            float totalSeconds, float warmupSeconds, float bucketSeconds, bool quiet = false)
        {
            var cfg = ScriptableObject.Instantiate(presetAsset); // clone: preserves floorRange etc.
            var reward  = ScriptableObject.CreateInstance<RewardConfig>();
            var obs     = ScriptableObject.CreateInstance<ObservationConfig>();
            var traffic = ScriptableObject.CreateInstance<TrafficConfig>();
            traffic.useDayCycle = false; traffic.defaultPattern = pattern; traffic.intensity = intensity;

            return RunCore(policy, dispatch, presetName, cfg, reward, obs, traffic, seed,
                totalSeconds, warmupSeconds, bucketSeconds, quiet);
        }

        /// <summary>Runs one episode from inline topology (ad hoc smoke tests; no preset asset).</summary>
        public static (EpisodeStats episode, string runDir) RunSingle(string policy, Dispatcher dispatch,
            string preset, int floors, int cars, int capacity,
            TrafficPattern pattern, float intensity, int seed,
            float totalSeconds, float warmupSeconds, float bucketSeconds, bool quiet = false)
        {
            var cfg = ScriptableObject.CreateInstance<BuildingConfig>();
            cfg.numFloors = floors; cfg.numElevators = cars; cfg.capacity = capacity;
            cfg.randomizeActive = false; cfg.minActiveElevators = 1; cfg.serviceChangeProbability = 0f;

            var reward  = ScriptableObject.CreateInstance<RewardConfig>();
            var obs     = ScriptableObject.CreateInstance<ObservationConfig>();
            var traffic = ScriptableObject.CreateInstance<TrafficConfig>();
            traffic.useDayCycle = false; traffic.defaultPattern = pattern; traffic.intensity = intensity;

            return RunCore(policy, dispatch, preset, cfg, reward, obs, traffic, seed,
                totalSeconds, warmupSeconds, bucketSeconds, quiet);
        }

        static (EpisodeStats episode, string runDir) RunCore(string policy, Dispatcher dispatch,
            string presetName, BuildingConfig cfg, RewardConfig reward, ObservationConfig obs,
            TrafficConfig traffic, int seed, float totalSeconds, float warmupSeconds, float bucketSeconds,
            bool quiet = false)
        {
            var b = new Building(cfg, reward, obs, traffic, seed);
            b.Reset();

            var col = new StatsCollector(b, warmupSeconds, bucketSeconds);

            const float dt = 0.05f;
            float decInterval = cfg.decisionInterval;
            float clock = 0f;
            int guard = 0, guardMax = (int)(totalSeconds / dt) + 1000;

            while (b.simTime < totalSeconds && guard++ < guardMax)
            {
                b.Tick(dt);
                clock += dt;
                if (clock >= decInterval)
                {
                    clock -= decInterval;
                    var act = dispatch(b);
                    for (int i = 0; i < cfg.numElevators; i++) b.ApplyAction(i, act[i]);
                    b.CollectReward();     // accumulates reward decomposition
                    col.Sample();          // occupancy + window rolling
                }
            }

            var id = new RunId
            {
                runId = $"{DateTime.Now:yyyyMMdd-HHmmss}-{policy}-{presetName}-{traffic.defaultPattern}-i{traffic.intensity:0.0}-s{seed}",
                gitSha = "", policy = policy, modelPath = "",
                buildingPreset = presetName, configHash = "", trafficPreset = "inline",
                pattern = traffic.defaultPattern.ToString(), intensity = traffic.intensity, seed = seed,
            };

            var episode = col.Finish(id);
            var floorsRows = col.BuildFloorStats(id);
            var windowRows = col.BuildWindowStats(id);
            var waitHistLines = quiet ? null : WaitHistLines(col.WaitHist);
            col.Dispose();

            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string runDir = Path.Combine(projectRoot, "Runs", id.runId);

            if (!quiet)
            {
                StatsCsv.Write(Path.Combine(runDir, "episode.csv"),
                    EpisodeStats.Header, new List<string> { episode.ToCsv() });

                var floorLines = new List<string>(floorsRows.Count);
                foreach (var fs in floorsRows) floorLines.Add(fs.ToCsv());
                StatsCsv.Write(Path.Combine(runDir, "floor_stats.csv"), FloorStats.Header, floorLines);

                var winLines = new List<string>(windowRows.Count);
                foreach (var ws in windowRows) winLines.Add(ws.ToCsv());
                StatsCsv.Write(Path.Combine(runDir, "window_stats.csv"), WindowStats.Header, winLines);

                // delivered-wait distribution (bins) → ECDF / histogram charts
                StatsCsv.Write(Path.Combine(runDir, "wait_hist.csv"), "binStart,binEnd,count", waitHistLines);
            }

            return (episode, runDir);
        }

        static List<string> WaitHistLines(WaitHistogram h)
        {
            var lines = new List<string>(h.BinCount);
            for (int i = 0; i < h.BinCount; i++)
                lines.Add($"{(i * h.BinWidth):F3},{((i + 1) * h.BinWidth):F3},{h.BinAt(i)}");
            return lines;
        }
    }
}
