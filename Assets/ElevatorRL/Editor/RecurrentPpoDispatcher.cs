using System;
using System.Collections.ObjectModel;
using System.Reflection;
using Unity.InferenceEngine;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace ElevatorRL.Editor
{
    /// <summary>
    /// Eval-side dispatcher for the E13a recurrent (LSTM) model. Identical flat observation + joint
    /// MultiDiscrete action + mask as <see cref="PpoDispatcher"/>, but a recurrent policy is NOT
    /// Markovian at inference: it expects its own hidden state fed back each step. ML-Agents exports
    /// this as a `recurrent_in` input and `recurrent_out` output, both shape [batch, memorySize]
    /// (memorySize = the yaml's network_settings.memory.memory_size, 128 for E13a; confirmed the
    /// tensor names from the installed package's Inference/TensorNames.cs). We carry recurrent_out ->
    /// next step's recurrent_in, and zero the memory at each episode boundary (detected by simTime
    /// resetting) so the memory does NOT leak across the 5 eval seeds that share one dispatcher.
    ///
    /// A stateless PpoDispatcher would feed zeroed memory every step and produce a degenerate policy
    /// (same failure class as the E5 obs-config bug) -- hence this variant.
    /// CONFIRM via onnx.load on the first trained model that memorySize matches recurrent_in's width.
    /// </summary>
    public sealed class RecurrentPpoDispatcher : IDisposable
    {
        readonly Worker _worker;
        readonly int _obsSize;
        readonly int _memorySize;
        readonly float[] _memory;
        float _lastSimTime = float.MaxValue;

        static readonly MethodInfo GetObservationsMethod =
            typeof(VectorSensor).GetMethod("GetObservations", BindingFlags.NonPublic | BindingFlags.Instance);

        public RecurrentPpoDispatcher(ModelAsset modelAsset, int obsSize, int memorySize)
        {
            var model = ModelLoader.Load(modelAsset);
            _worker = new Worker(model, BackendType.CPU);
            _obsSize = obsSize;
            _memorySize = memorySize;
            _memory = new float[memorySize];
        }

        public int[] Dispatch(Building b)
        {
            int E = b.cfg.numElevators;

            // New episode (fresh RunSingle / next seed) -> reset hidden state so memory doesn't leak.
            if (b.simTime < _lastSimTime) Array.Clear(_memory, 0, _memory.Length);
            _lastSimTime = b.simTime;

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
            using var memTensor = new Tensor<float>(new TensorShape(1, _memorySize), _memory);

            _worker.SetInput("obs_0", obsTensor);
            _worker.SetInput("action_masks", maskTensor);
            _worker.SetInput("recurrent_in", memTensor);
            _worker.Schedule();

            using var outTensor = _worker.PeekOutput("deterministic_discrete_actions") as Tensor<int>;
            var flat = outTensor.DownloadToArray();

            // Carry the new hidden state forward.
            using var recOut = _worker.PeekOutput("recurrent_out") as Tensor<float>;
            var mem = recOut.DownloadToArray();
            Array.Copy(mem, _memory, Math.Min(mem.Length, _memory.Length));

            var result = new int[E];
            Array.Copy(flat, result, E);
            return result;
        }

        public void Dispose() => _worker?.Dispose();
    }
}
