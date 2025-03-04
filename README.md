# Prosit: PROcess SImulation Tool

A Python Package for building and discovering Rule-Aware Business Process Simulation from event log data.


### How to use:

<ol>
    <li>
        <strong>Clone this repository.</strong>
    </li>
    <li>
        <strong>Create environment:</strong>
        <pre><code>$ conda env create -f environment.yml</code></pre>
    </li>
</ol>


```python
import prosit.simulator as simulator
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

# read input Event log and Petri net
log = xes_importer.apply('data/logs/purchasing.xes')
net, initial_marking, final_marking = pm4py.read_pnml("data/models/purchasing.pnml")

# initialize simulation parameters
parameters = simulator.SimulatorParameters(net, initial_marking, final_marking)
# discover from event log
parameters.discover_from_eventlog(log)

# initialize simulation engine
simulator_eng = simulator.SimulatorEngine(parameters)
# simulate event log
sim_log = simulator_eng.apply(n_sim_traces, start_ts_simulation=log[0][0]['start:timestamp'])
```
