namespace ElevatorRL
{
    /// <summary>
    /// State-only record for one rider. Passengers are never agents; they are part
    /// of the environment state and the reward signal. Times are in SIM SECONDS.
    /// </summary>
    public sealed class Passenger
    {
        public int id;
        public int origin;      // floor the passenger appeared on (re-set if re-queued)
        public int dest;        // destination floor (never == origin)
        public float arrivalTime;   // simTime when spawned
        public float waitTime;      // accrues ONLY while standing in a hall queue
        public float age;           // accrues everywhere (queue + in-car)

        /// <summary>+1 if travelling up, -1 if travelling down.</summary>
        public int Dir => dest > origin ? 1 : -1;

        public Passenger(int id, int origin, int dest, float now)
        {
            this.id = id;
            this.origin = origin;
            this.dest = dest;
            arrivalTime = now;
            waitTime = 0f;
            age = 0f;
        }
    }
}
