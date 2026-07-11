using System.Collections.Generic;
using System.Globalization;
using System.IO;
using ElevatorRL.Stats;

namespace ElevatorRL.Reports
{
    public struct WaitHistBin { public float start, end; public int count; }

    /// <summary>One row of an E1 sweep summary.csv (EvalHarness.RunSweepCalibrated output).</summary>
    public struct SummaryRow
    {
        public string preset, policy, pattern;
        public float intensity, calibratedBase, intensityMultiplier;
        public int seeds;
        public float deliveredMean, waitMeanMean, waitP95Mean, waitMaxMean,
            abandonedMean, rejectedMean, abandonRateMean, rwTotalMean, utilFleetMeanMean;
    }

    /// <summary>
    /// Reads the canonical CSV tables (EvalHarness's output — see StatsRecords.cs's Header/ToCsv
    /// for the exact column order every parser below mirrors) back into typed rows for the report
    /// scene generators. Fields never contain embedded commas, so a naive Split(',') is safe — no
    /// quoting/escaping to handle.
    /// </summary>
    public static class ReportData
    {
        static float F(string s) => float.Parse(s, CultureInfo.InvariantCulture);
        static int I(string s) => int.Parse(s, CultureInfo.InvariantCulture);

        static string[][] Rows(string path)
        {
            var lines = File.ReadAllLines(path);
            var rows = new List<string[]>(lines.Length - 1);
            for (int i = 1; i < lines.Length; i++) // skip header
            {
                if (string.IsNullOrWhiteSpace(lines[i])) continue;
                rows.Add(lines[i].Split(','));
            }
            return rows.ToArray();
        }

        public static EpisodeStats ReadEpisode(string path)
        {
            var r = Rows(path)[0];
            return new EpisodeStats
            {
                id = new RunId
                {
                    runId = r[0], gitSha = r[1], policy = r[2], modelPath = r[3],
                    buildingPreset = r[4], configHash = r[5], trafficPreset = r[6],
                    pattern = r[7], intensity = F(r[8]), seed = I(r[9]),
                    simSeconds = F(r[10]), warmupSeconds = F(r[11]),
                },
                delivered = I(r[12]), deliveredPerHour = F(r[13]), deliveredPerDecision = F(r[14]),
                waitMean = F(r[15]), waitP50 = F(r[16]), waitP95 = F(r[17]), waitMax = F(r[18]),
                rideMean = F(r[19]), rideP95 = F(r[20]),
                abandoned = I(r[21]), rejected = I(r[22]), abandonRate = F(r[23]), rejectRate = F(r[24]),
                waitStd = F(r[25]), waitTailRatio = F(r[26]),
                rwTotal = F(r[27]), rwDelivered = F(r[28]), rwToward = F(r[29]), rwAway = F(r[30]),
                rwRejected = F(r[31]), rwAbandoned = F(r[32]), rwInElevator = F(r[33]), rwInQueue = F(r[34]),
                utilFleetMean = F(r[35]), idleFraction = F(r[36]), inServiceMean = F(r[37]),
            };
        }

        public static List<FloorStats> ReadFloors(string path)
        {
            var list = new List<FloorStats>();
            foreach (var r in Rows(path))
                list.Add(new FloorStats
                {
                    id = new RunId { runId = r[0], policy = r[1], buildingPreset = r[2], pattern = r[3], intensity = F(r[4]), seed = I(r[5]) },
                    floor = I(r[6]), carsServing = I(r[7]), origins = I(r[8]), destinations = I(r[9]),
                    delivered = I(r[10]), abandoned = I(r[11]), rejected = I(r[12]),
                    waitMean = F(r[13]), waitP95 = F(r[14]), waitMax = F(r[15]),
                    queueLenMean = F(r[16]), queueLenMax = F(r[17]),
                });
            return list;
        }

        public static List<WindowStats> ReadWindows(string path)
        {
            var list = new List<WindowStats>();
            foreach (var r in Rows(path))
                list.Add(new WindowStats
                {
                    id = new RunId { runId = r[0], policy = r[1], buildingPreset = r[2], pattern = r[3], intensity = F(r[4]), seed = I(r[5]) },
                    bucketStart = F(r[6]), activePattern = r[7],
                    delivered = I(r[8]), deliveredRate = F(r[9]), waitMean = F(r[10]), waitP95 = F(r[11]),
                    abandoned = I(r[12]), rejected = I(r[13]), fleetUtilMean = F(r[14]), carsInService = F(r[15]),
                });
            return list;
        }

        public static List<WaitHistBin> ReadWaitHist(string path)
        {
            var list = new List<WaitHistBin>();
            foreach (var r in Rows(path))
                list.Add(new WaitHistBin { start = F(r[0]), end = F(r[1]), count = I(r[2]) });
            return list;
        }

        public static List<SummaryRow> ReadSummary(string path)
        {
            var list = new List<SummaryRow>();
            foreach (var r in Rows(path))
                list.Add(new SummaryRow
                {
                    preset = r[0], policy = r[1], pattern = r[2],
                    intensity = F(r[3]), calibratedBase = F(r[4]), intensityMultiplier = F(r[5]),
                    seeds = I(r[6]), deliveredMean = F(r[7]), waitMeanMean = F(r[8]), waitP95Mean = F(r[9]),
                    waitMaxMean = F(r[10]), abandonedMean = F(r[11]), rejectedMean = F(r[12]),
                    abandonRateMean = F(r[13]), rwTotalMean = F(r[14]), utilFleetMeanMean = F(r[15]),
                });
            return list;
        }
    }
}
