using System.Collections.Generic;
using UnityEngine;

namespace ElevatorRL
{
    public enum CarState { Idle, Moving, DoorsOpening, Dwelling, DoorsClosing }

    /// <summary>
    /// One elevator car. Holds a CONTINUOUS position (in floor units) plus a door
    /// state machine. The car carries no Unity types so it can run headless at high
    /// time-scale. Building advances it; the agent only issues commands.
    /// </summary>
    public sealed class Elevator
    {
        public int id;
        public int minFloor, maxFloor, capacity;

        public float position;          // continuous, in floor units
        public int target;              // commanded floor (for Moving)
        public int dir = 1;             // last serviced direction (+1 up / -1 down)
        public CarState state = CarState.Idle;
        public float timer;             // door / dwell countdown (seconds)
        public int pending;             // action that opened the doors (3/4/5); 0 otherwise

        public bool inService = true;   // false => out of service / not present in this episode

        public readonly List<Passenger> riders = new List<Passenger>();

        public int Floor => Mathf.RoundToInt(position);
        public bool AtFloor => state == CarState.Idle;
        public int Free => capacity - riders.Count;
        public float Load => capacity > 0 ? riders.Count / (float)capacity : 0f;

        public Elevator(int id, int minFloor, int maxFloor, int capacity, int startFloor)
        {
            this.id = id;
            this.minFloor = minFloor;
            this.maxFloor = maxFloor;
            this.capacity = capacity;
            position = startFloor;
            target = startFloor;
        }

        public bool WantsFloor(int f)
        {
            for (int i = 0; i < riders.Count; i++)
                if (riders[i].dest == f) return true;
            return false;
        }

        public void HardReset(int startFloor)
        {
            position = startFloor;
            target = startFloor;
            dir = 1;
            state = CarState.Idle;
            timer = 0f;
            pending = 0;
            inService = true;
            riders.Clear();
        }
    }
}
