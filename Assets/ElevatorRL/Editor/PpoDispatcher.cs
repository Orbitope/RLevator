using System;
using System.Collections.ObjectModel;
using System.Reflection;
using Unity.InferenceEngine;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace ElevatorRL.Editor
{
    /// <summary>
    /// Runs a trained ML-Agents ONNX policy (Primitive/AS0 action space only) as an
    /// EvalHarness Dispatcher, so it can be driven through the exact same Building/
    /// StatsCollector loop used for the LOOK/ETA baselines (RunSingle/RunCore) — same
    /// seed, same traffic, same everything — for a fair E2 comparison. Bypasses
    /// ML-Agents' own Agent/Academy runtime entirely: no scene, no headless build.
    ///
    /// Observation is built via the real Building.WriteObservation (through a VectorSensor,
    /// identical to what training saw — bit-for-bit, not a reimplementation) and the action
    /// mask duplicates ElevatorControllerAgent.WriteDiscreteActionMask's AS0 branch exactly.
    /// Inference reads the "deterministic_discrete_actions" output (greedy, no sampling
    /// noise), matching the deterministic nature of the heuristic baselines it's compared
    /// against. Tensor/input names ("obs_0", "action_masks") and the mask convention
    /// (1=allowed, 0=disabled) are ML-Agents' own stable inference contract — see
    /// com.unity.ml-agents Runtime/Inference/TensorNames.cs and GeneratorImpl.cs.
    /// </summary>
    public sealed class PpoDispatcher : IDisposable
    {
        readonly Worker _worker;
        readonly int _obsSize;

        // VectorSensor's internal float buffer has no public accessor; Building.WriteObservation
        // is the one production code path that fills it and is deeply tied to VectorSensor's API
        // (AddObservation/AddOneHotObservation), so reflecting into this stable, shipped package
        // method is far less risky than duplicating ~80 lines of observation-writing logic by hand.
        static readonly MethodInfo GetObservationsMethod =
            typeof(VectorSensor).GetMethod("GetObservations", BindingFlags.NonPublic | BindingFlags.Instance);

        public PpoDispatcher(ModelAsset modelAsset, int obsSize)
        {
            var model = ModelLoader.Load(modelAsset);
            _worker = new Worker(model, BackendType.CPU);
            _obsSize = obsSize;
        }

        public int[] Dispatch(Building b)
        {
            int E = b.cfg.numElevators;

            var sensor = new VectorSensor(_obsSize);
            b.WriteObservation(sensor);
            var obsList = (ReadOnlyCollection<float>)GetObservationsMethod.Invoke(sensor, null);
            var obsArr = new float[_obsSize];
            obsList.CopyTo(obsArr, 0);

            var mask = new float[E * 6];
            for (int i = 0; i < mask.Length; i++) mask[i] = 1f;
            for (int i = 0; i < E; i++)
            {
                var c = b.cars[i];
                int baseIdx = i * 6;
                if (!c.inService || !c.AtFloor)
                {
                    for (int a = 1; a <= 5; a++) mask[baseIdx + a] = 0f;
                    continue;
                }
                if (c.Floor >= c.maxFloor) mask[baseIdx + 1] = 0f;
                if (c.Floor <= c.minFloor) mask[baseIdx + 2] = 0f;
                if (b.upQ[c.Floor].Count == 0) mask[baseIdx + 3] = 0f;
                if (b.downQ[c.Floor].Count == 0) mask[baseIdx + 4] = 0f;
                if (!c.WantsFloor(c.Floor)) mask[baseIdx + 5] = 0f;
            }

            using var obsTensor = new Tensor<float>(new TensorShape(1, _obsSize), obsArr);
            using var maskTensor = new Tensor<float>(new TensorShape(1, E * 6), mask);

            _worker.SetInput("obs_0", obsTensor);
            _worker.SetInput("action_masks", maskTensor);
            _worker.Schedule();

            using var outTensor = _worker.PeekOutput("deterministic_discrete_actions") as Tensor<int>;
            var flat = outTensor.DownloadToArray();

            var result = new int[E];
            Array.Copy(flat, result, E);
            return result;
        }

        public void Dispose() => _worker?.Dispose();
    }
}
