using System;
using System.Collections.ObjectModel;
using System.Reflection;
using Unity.InferenceEngine;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace ElevatorRL.Editor
{
    /// <summary>
    /// Eval-side counterpart to PpoDispatcher for the E6 Architecture A (multi-agent /
    /// parameter-sharing) model. That model is ONE shared policy over a single car: input obs_0 is
    /// a per-car observation (Building.WriteCarObservation), output is one size-6 discrete action.
    /// So this dispatcher loops over cars, runs the shared policy once per eligible car with that
    /// car's own observation + 6-mask, and assembles the fleet's int[] actions — exactly how
    /// BuildingManager drives the N car agents at train time, just sequentially here. Plugs into
    /// the same EvalHarness RunScaleLadderSweep loop as PpoDispatcher for an apples-to-apples
    /// comparison against LOOK/ETA/flat-MLP on the identical Building/StatsCollector.
    ///
    /// Only eligible cars (in service + at a floor) are run, matching training: BuildingManager
    /// only calls RequestDecision() on eligible cars, so the policy never saw ineligible-car
    /// observations. Ineligible cars get NOOP (0), which Building.ApplyAction ignores anyway.
    /// </summary>
    public sealed class MultiAgentPpoDispatcher : IDisposable
    {
        readonly Worker _worker;
        readonly int _carObsSize;

        static readonly MethodInfo GetObservationsMethod =
            typeof(VectorSensor).GetMethod("GetObservations", BindingFlags.NonPublic | BindingFlags.Instance);

        // Diagnostic: count how often each of the 6 actions is chosen across a run. A degenerate
        // (all-NOOP) policy shows up as everything in [0]; a working dispatcher on a functional
        // policy shows board/move actions too. Lets us tell "policy never learned" apart from
        // "dispatcher bug" when delivered==0. Read via ActionHistogram after a sweep.
        public readonly long[] ActionHistogram = new long[6];

        public MultiAgentPpoDispatcher(ModelAsset modelAsset, int carObsSize)
        {
            var model = ModelLoader.Load(modelAsset);
            _worker = new Worker(model, BackendType.CPU);
            _carObsSize = carObsSize;
        }

        public int[] Dispatch(Building b)
        {
            int E = b.cfg.numElevators;
            var result = new int[E]; // default 0 = NOOP for ineligible cars (ignored by ApplyAction)

            for (int i = 0; i < E; i++)
            {
                var c = b.cars[i];
                if (!c.inService || !c.AtFloor) continue;

                var sensor = new VectorSensor(_carObsSize);
                b.WriteCarObservation(sensor, i);
                var obsList = (ReadOnlyCollection<float>)GetObservationsMethod.Invoke(sensor, null);
                var obsArr = new float[_carObsSize];
                obsList.CopyTo(obsArr, 0);

                // Same per-car mask as ElevatorCarAgent.WriteDiscreteActionMask (NOOP always legal).
                var mask = new float[6];
                for (int k = 0; k < 6; k++) mask[k] = 1f;
                if (c.Floor >= c.maxFloor) mask[1] = 0f;
                if (c.Floor <= c.minFloor) mask[2] = 0f;
                if (b.upQ[c.Floor].Count == 0) mask[3] = 0f;
                if (b.downQ[c.Floor].Count == 0) mask[4] = 0f;
                if (!c.WantsFloor(c.Floor)) mask[5] = 0f;

                using var obsTensor = new Tensor<float>(new TensorShape(1, _carObsSize), obsArr);
                using var maskTensor = new Tensor<float>(new TensorShape(1, 6), mask);

                _worker.SetInput("obs_0", obsTensor);
                _worker.SetInput("action_masks", maskTensor);
                _worker.Schedule();

                using var outTensor = _worker.PeekOutput("deterministic_discrete_actions") as Tensor<int>;
                int a = outTensor.DownloadToArray()[0];
                if (a >= 0 && a < 6) ActionHistogram[a]++;
                result[i] = a;
            }

            return result;
        }

        public void Dispose() => _worker?.Dispose();
    }
}
