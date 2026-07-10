using System.Globalization;
using System.Text;

namespace ElevatorRL.Stats
{
    /// <summary>CSV formatting helpers (invariant culture, so decimals are portable).</summary>
    internal static class Csv
    {
        public static string F(float v) => v.ToString("0.####", CultureInfo.InvariantCulture);
        public static string F2(float v) => v.ToString("0.##", CultureInfo.InvariantCulture);

        public static string Join(params string[] cells) => string.Join(",", cells);
    }

    /// <summary>
    /// Shared run/scenario identity stamped onto every record so any table stands alone and can
    /// be filtered/joined without the manifest.
    /// </summary>
    public struct RunId
    {
        public string runId, gitSha, policy, modelPath, buildingPreset, configHash, trafficPreset;
        public string pattern;
        public float intensity;
        public int seed;
        public float simSeconds, warmupSeconds;
    }

    /// <summary>One row per evaluated episode — the aggregate (§1.1).</summary>
    public sealed class EpisodeStats
    {
        public RunId id;

        // throughput
        public int delivered;
        public float deliveredPerHour, deliveredPerDecision;

        // wait (sim seconds)
        public float waitMean, waitP50, waitP95, waitMax, rideMean, rideP95;

        // failure
        public int abandoned, rejected;
        public float abandonRate, rejectRate;

        // fairness
        public float waitStd, waitTailRatio;

        // reward decomposition
        public float rwTotal, rwDelivered, rwToward, rwAway, rwRejected, rwAbandoned, rwInElevator, rwInQueue;

        // fleet
        public float utilFleetMean, idleFraction, inServiceMean;

        public const string Header =
            "runId,gitSha,policy,modelPath,buildingPreset,configHash,trafficPreset,pattern,intensity,seed,simSeconds,warmupSeconds," +
            "delivered,deliveredPerHour,deliveredPerDecision," +
            "waitMean,waitP50,waitP95,waitMax,rideMean,rideP95," +
            "abandoned,rejected,abandonRate,rejectRate," +
            "waitStd,waitTailRatio," +
            "rwTotal,rwDelivered,rwToward,rwAway,rwRejected,rwAbandoned,rwInElevator,rwInQueue," +
            "utilFleetMean,idleFraction,inServiceMean";

        public string ToCsv() => Csv.Join(
            id.runId, id.gitSha, id.policy, id.modelPath, id.buildingPreset, id.configHash, id.trafficPreset,
            id.pattern, Csv.F(id.intensity), id.seed.ToString(), Csv.F(id.simSeconds), Csv.F(id.warmupSeconds),
            delivered.ToString(), Csv.F(deliveredPerHour), Csv.F(deliveredPerDecision),
            Csv.F(waitMean), Csv.F(waitP50), Csv.F(waitP95), Csv.F(waitMax), Csv.F(rideMean), Csv.F(rideP95),
            abandoned.ToString(), rejected.ToString(), Csv.F(abandonRate), Csv.F(rejectRate),
            Csv.F(waitStd), Csv.F(waitTailRatio),
            Csv.F(rwTotal), Csv.F(rwDelivered), Csv.F(rwToward), Csv.F(rwAway), Csv.F(rwRejected),
            Csv.F(rwAbandoned), Csv.F(rwInElevator), Csv.F(rwInQueue),
            Csv.F(utilFleetMean), Csv.F(idleFraction), Csv.F(inServiceMean));
    }

    /// <summary>One row per (episode × floor) — the per-floor breakdown (§1.2).</summary>
    public sealed class FloorStats
    {
        public RunId id;
        public int floor, carsServing, origins, destinations, delivered, abandoned, rejected;
        public float waitMean, waitP95, waitMax, queueLenMean, queueLenMax;

        public const string Header =
            "runId,policy,buildingPreset,pattern,intensity,seed," +
            "floor,carsServing,origins,destinations,delivered,abandoned,rejected," +
            "waitMean,waitP95,waitMax,queueLenMean,queueLenMax";

        public string ToCsv() => Csv.Join(
            id.runId, id.policy, id.buildingPreset, id.pattern, Csv.F(id.intensity), id.seed.ToString(),
            floor.ToString(), carsServing.ToString(), origins.ToString(), destinations.ToString(),
            delivered.ToString(), abandoned.ToString(), rejected.ToString(),
            Csv.F(waitMean), Csv.F(waitP95), Csv.F(waitMax), Csv.F(queueLenMean), Csv.F(queueLenMax));
    }

    /// <summary>One row per (episode × time-bucket) — the temporal breakdown (§1.3).</summary>
    public sealed class WindowStats
    {
        public RunId id;
        public float bucketStart;
        public string activePattern;
        public int delivered, abandoned, rejected;
        public float deliveredRate, waitMean, waitP95, fleetUtilMean;
        public float carsInService;

        public const string Header =
            "runId,policy,buildingPreset,pattern,intensity,seed," +
            "bucketStart,activePattern,delivered,deliveredRate,waitMean,waitP95,abandoned,rejected,fleetUtilMean,carsInService";

        public string ToCsv() => Csv.Join(
            id.runId, id.policy, id.buildingPreset, id.pattern, Csv.F(id.intensity), id.seed.ToString(),
            Csv.F(bucketStart), activePattern, delivered.ToString(), Csv.F(deliveredRate),
            Csv.F(waitMean), Csv.F(waitP95), abandoned.ToString(), rejected.ToString(),
            Csv.F(fleetUtilMean), Csv.F2(carsInService));
    }
}
