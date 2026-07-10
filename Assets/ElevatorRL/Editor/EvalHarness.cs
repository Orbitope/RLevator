using System;
using System.Collections.Generic;
using System.IO;
using ElevatorRL.Stats;
using UnityEngine;
using UnityEditor;

namespace ElevatorRL.Editor
{
    /// <summary>
    /// Headless evaluation runner (seed of the M1 sweep). Runs a deterministic episode under a
    /// chosen dispatcher, drives a StatsCollector, and writes the three canonical CSV tables.
    /// Also the correctness smoke test for the M0 instrumentation.
    /// </summary>
    public static class EvalHarness
    {
        /// <summary>A dispatch policy: given the live Building, return one action per car.</summary>
        public delegate int[] Dispatcher(Building b);

        // NOTE: intensity=1.0 saturates this 8-floor/3-car preset under UpPeak (~41% rejection —
        // floor-0 arrival rate already exceeds fleet throughput capacity there). Use ~0.5 for a
        // representative, non-saturated comparison; sweep intensity deliberately for the
        // saturation-curve experiments (EXPERIMENT_PLAN.md E1/E3).
        const float SmokeIntensity = 0.5f;

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

        /// <summary>Runs one episode headlessly under the given dispatcher, writes CSVs.</summary>
        public static (EpisodeStats episode, string runDir) RunSingle(string policy, Dispatcher dispatch,
            string preset, int floors, int cars, int capacity,
            TrafficPattern pattern, float intensity, int seed,
            float totalSeconds, float warmupSeconds, float bucketSeconds)
        {
            // self-contained configs (don't touch saved assets)
            var cfg = ScriptableObject.CreateInstance<BuildingConfig>();
            cfg.numFloors = floors; cfg.numElevators = cars; cfg.capacity = capacity;
            cfg.randomizeActive = false; cfg.minActiveElevators = 1; cfg.serviceChangeProbability = 0f;

            var reward  = ScriptableObject.CreateInstance<RewardConfig>();
            var obs     = ScriptableObject.CreateInstance<ObservationConfig>();
            var traffic = ScriptableObject.CreateInstance<TrafficConfig>();
            traffic.useDayCycle = false; traffic.defaultPattern = pattern; traffic.intensity = intensity;

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
                runId = $"{DateTime.Now:yyyyMMdd-HHmmss}-{policy}-{preset}",
                gitSha = "", policy = policy, modelPath = "",
                buildingPreset = preset, configHash = "", trafficPreset = "inline",
                pattern = pattern.ToString(), intensity = intensity, seed = seed,
            };

            var episode = col.Finish(id);
            var floorsRows = col.BuildFloorStats(id);
            var windowRows = col.BuildWindowStats(id);
            col.Dispose();

            // write CSVs to <project>/Runs/<runId>/
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string runDir = Path.Combine(projectRoot, "Runs", id.runId);

            StatsCsv.Write(Path.Combine(runDir, "episode.csv"),
                EpisodeStats.Header, new List<string> { episode.ToCsv() });

            var floorLines = new List<string>(floorsRows.Count);
            foreach (var fs in floorsRows) floorLines.Add(fs.ToCsv());
            StatsCsv.Write(Path.Combine(runDir, "floor_stats.csv"), FloorStats.Header, floorLines);

            var winLines = new List<string>(windowRows.Count);
            foreach (var ws in windowRows) winLines.Add(ws.ToCsv());
            StatsCsv.Write(Path.Combine(runDir, "window_stats.csv"), WindowStats.Header, winLines);

            return (episode, runDir);
        }
    }
}
