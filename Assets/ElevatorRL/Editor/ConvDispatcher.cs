using System;
using System.Collections.ObjectModel;
using System.Reflection;
using Unity.InferenceEngine;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace ElevatorRL.Editor
{
    /// <summary>
    /// Eval-side dispatcher for the E13b floor-axis-conv model (EXPERIMENT_PLAN.md E13). That model
    /// has TWO observation inputs: the flat VectorSensor (baseline obs) plus the FloorGridSensor's
    /// (1 x F x 8) visual grid. ML-Agents sorts sensors alphabetically by name
    /// (SensorUtils.SortSensors), and "FloorGrid" &lt; "VectorSensor_size&lt;N&gt;", so the grid
    /// registers FIRST: **obs_0 = visual grid, obs_1 = flat vector** (same index-swap the E6
    /// AttentionDispatcher documented for its custom sensor).
    ///
    /// TWO THINGS MUST BE CONFIRMED VIA onnx.load ON THE FIRST TRAINED CONV MODEL before trusting any
    /// eval numbers (exactly how AttentionDispatcher was finalized):
    ///   1. Input names / order — verify obs_0 is the (1,?,?,?) visual tensor and obs_1 the (1,254)
    ///      vector (a mismatch = silent garbage, NOT a crash).
    ///   2. Visual tensor LAYOUT — ML-Agents may export the grid as NHWC (1,F,8,1) or NCHW (1,1,F,8);
    ///      _visualShape below is the current best guess and must be checked against the ONNX graph.
    /// Until both are confirmed, treat this dispatcher's output as UNVALIDATED.
    /// </summary>
    public sealed class ConvDispatcher : IDisposable
    {
        readonly Worker _worker;
        readonly int _obsSize;
        readonly int _floors;
        readonly int _features;

        static readonly MethodInfo GetObservationsMethod =
            typeof(VectorSensor).GetMethod("GetObservations", BindingFlags.NonPublic | BindingFlags.Instance);

        public ConvDispatcher(ModelAsset modelAsset, int obsSize, int floors, int features)
        {
            var model = ModelLoader.Load(modelAsset);
            _worker = new Worker(model, BackendType.CPU);
            _obsSize = obsSize;
            _floors = floors;
            _features = features;
        }

        public int[] Dispatch(Building b)
        {
            int E = b.cfg.numElevators;

            // Flat vector obs (obs_1) — same as PpoDispatcher.
            var sensor = new VectorSensor(_obsSize);
            b.WriteObservation(sensor);
            var obsList = (ReadOnlyCollection<float>)GetObservationsMethod.Invoke(sensor, null);
            var flatArr = new float[_obsSize];
            obsList.CopyTo(flatArr, 0);

            // Visual grid obs (obs_0). Building.FillFloorGrid is the SAME source of truth the training
            // sensor uses (FloorGridSensor -> WriteFloorGrid -> FillFloorGrid), laid out h-major
            // (buf[floor*features + feature]); the Tensor shape below decides how Sentis interprets it
            // (SEE CLASS DOC — layout unconfirmed).
            var gridArr = new float[_floors * _features];
            b.FillFloorGrid(gridArr);

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

            // CONFIRMED via scripts/inspect_onnx.py on the smoke model: obs_0 = (batch, 1, F, 8),
            // i.e. NCHW (channels=1, height=floors, width=features). FillFloorGrid already writes
            // h-major/w-minor (buf[f*8 + c]), which is exactly the flat order NCHW (1,1,F,8) expects.
            using var gridTensor = new Tensor<float>(new TensorShape(1, 1, _floors, _features), gridArr);
            using var flatTensor = new Tensor<float>(new TensorShape(1, _obsSize), flatArr);
            using var maskTensor = new Tensor<float>(new TensorShape(1, E * 6), mask);

            _worker.SetInput("obs_0", gridTensor);
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
