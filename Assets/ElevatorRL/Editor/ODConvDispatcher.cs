using System;
using System.Collections.ObjectModel;
using System.Reflection;
using Unity.InferenceEngine;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace ElevatorRL.Editor
{
    /// <summary>
    /// Eval-side dispatcher for the E13d origin×destination conv model. Two observation inputs:
    /// the (2 x F x F) OD grid and the flat 254 vector. ML-Agents sorts sensors by name
    /// (SensorUtils.SortSensors) and "FloorOD" &lt; "VectorSensor_size&lt;N&gt;", so
    /// **obs_0 = OD grid, obs_1 = flat vector** — same index-swap pattern as ConvDispatcher /
    /// AttentionDispatcher.
    ///
    /// Tensor layout is NCHW (batch, 2, F, F): PROVEN, not guessed — ObservationWriter's flat index is
    /// TensorExtensions.Index(n,c,h,w) = n*H*W*C + c*H*W + h*W + w, and a 3-element visual spec is
    /// built as TensorShape(batch, C, H, W). Building.FillFloorOD writes exactly that order
    /// (c*F*F + origin*F + dest), so the buffer feeds straight in. Still worth a one-line
    /// scripts/inspect_onnx.py check on the first trained model to confirm obs_0 is (batch,2,F,F).
    /// </summary>
    public sealed class ODConvDispatcher : IDisposable
    {
        readonly Worker _worker;
        readonly int _obsSize;
        readonly int _floors;
        readonly float[] _od;

        static readonly MethodInfo GetObservationsMethod =
            typeof(VectorSensor).GetMethod("GetObservations", BindingFlags.NonPublic | BindingFlags.Instance);

        public ODConvDispatcher(ModelAsset modelAsset, int obsSize, int floors)
        {
            var model = ModelLoader.Load(modelAsset);
            _worker = new Worker(model, BackendType.CPU);
            _obsSize = obsSize;
            _floors = floors;
            _od = new float[Building.FloorODChannels * floors * floors];
        }

        public int[] Dispatch(Building b)
        {
            int E = b.cfg.numElevators;

            var sensor = new VectorSensor(_obsSize);
            b.WriteObservation(sensor);
            var obsList = (ReadOnlyCollection<float>)GetObservationsMethod.Invoke(sensor, null);
            var flatArr = new float[_obsSize];
            obsList.CopyTo(flatArr, 0);

            b.FillFloorOD(_od); // same source of truth the training sensor uses

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

            using var odTensor = new Tensor<float>(
                new TensorShape(1, Building.FloorODChannels, _floors, _floors), _od);
            using var flatTensor = new Tensor<float>(new TensorShape(1, _obsSize), flatArr);
            using var maskTensor = new Tensor<float>(new TensorShape(1, E * 6), mask);

            _worker.SetInput("obs_0", odTensor);
            _worker.SetInput("obs_1", flatTensor);
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
