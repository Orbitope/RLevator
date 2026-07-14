using System;
using System.Collections.ObjectModel;
using System.Reflection;
using Unity.InferenceEngine;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace ElevatorRL.Editor
{
    /// <summary>
    /// Eval-side counterpart to PpoDispatcher for the E6 Architecture B (BufferSensor attention)
    /// model. Confirmed the exported ONNX's actual input contract by inspecting the graph directly
    /// (onnx.load on ElevatorAttention_M_e6b_10m.onnx): "obs_0" is the BufferSensor entity tensor
    /// (batch, maxNumObservables, carEntitySize) and "obs_1" is the global VectorSensor (batch,
    /// globalObsSize) — BufferSensorComponent registers before the Agent's own VectorSensor, so the
    /// indices are swapped from the naive CollectObservations call order. No separate entity mask
    /// input; ML-Agents' attention handles unused/padded entity slots internally.
    /// </summary>
    public sealed class AttentionDispatcher : IDisposable
    {
        readonly Worker _worker;
        readonly int _globalObsSize;
        readonly int _carEntitySize;
        readonly int _maxNumObservables;

        static readonly MethodInfo GetObservationsMethod =
            typeof(VectorSensor).GetMethod("GetObservations", BindingFlags.NonPublic | BindingFlags.Instance);

        public AttentionDispatcher(ModelAsset modelAsset, int globalObsSize, int carEntitySize, int maxNumObservables)
        {
            var model = ModelLoader.Load(modelAsset);
            _worker = new Worker(model, BackendType.CPU);
            _globalObsSize = globalObsSize;
            _carEntitySize = carEntitySize;
            _maxNumObservables = maxNumObservables;
        }

        public int[] Dispatch(Building b)
        {
            int E = b.cfg.numElevators;

            var sensor = new VectorSensor(_globalObsSize);
            b.WriteGlobalObservation(sensor);
            var obsList = (ReadOnlyCollection<float>)GetObservationsMethod.Invoke(sensor, null);
            var globalArr = new float[_globalObsSize];
            obsList.CopyTo(globalArr, 0);

            // Entity buffer: one row per in-service car, zero-padded to maxNumObservables (matches
            // BufferSensor.AppendObservation order in ElevatorAttentionAgent.CollectObservations).
            var entityArr = new float[_maxNumObservables * _carEntitySize];
            var carBuf = new float[_carEntitySize];
            int row = 0;
            for (int i = 0; i < E && row < _maxNumObservables; i++)
            {
                if (!b.cars[i].inService) continue;
                Array.Clear(carBuf, 0, carBuf.Length);
                b.WriteCarEntity(carBuf, i);
                Array.Copy(carBuf, 0, entityArr, row * _carEntitySize, _carEntitySize);
                row++;
            }

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

            using var entityTensor = new Tensor<float>(new TensorShape(1, _maxNumObservables, _carEntitySize), entityArr);
            using var globalTensor = new Tensor<float>(new TensorShape(1, _globalObsSize), globalArr);
            using var maskTensor = new Tensor<float>(new TensorShape(1, E * 6), mask);

            _worker.SetInput("obs_0", entityTensor);
            _worker.SetInput("obs_1", globalTensor);
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
