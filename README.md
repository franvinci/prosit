# Prosit: PROcess SImulation Tool

A Python Package for building and discovering Rule-Aware Business Process Simulation from event log data.


## 🛠️ How to Use

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
parameters = simulator.SimulatorParameters(net, initial_marking, final_marking, max_depth_tree=5)
# discover from event log
parameters.discover_from_eventlog(log)

# initialize simulation engine
simulator_eng = simulator.SimulatorEngine(parameters)
# simulate event log
sim_log = simulator_eng.apply(n_traces=len(log), t_start=log[0][0]['start:timestamp'])
```


## 📊 Evaluation Results

- **3GD**: 3-Gram Distance  
- **CTD**: Cycle Time Distribution Distance 
- **CAR**: Case Arrival Rate Distance 

- ⬜ = White-Box, ⬛ = Black-Box
- 🟩 = Best result, 🟦 = Best White-Box, 🟥 = Worst result

<table>
  <thead>
    <tr>
      <th>Case Study</th>
      <th>Metric</th>
      <th>⬜ Simod</th>
      <th>⬛ DSim</th>
      <th>⬛ RIMS</th>
      <th>⬜ ProSiT</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="6"></td></tr>
    <!-- P2P -->
    <tr><td rowspan="3">P2P</td><td>3GD</td><td>0.577218</td><td>🟥 0.596506</td><td>0.593391</td><td>🟩🟦 0.351867</td></tr>
    <tr><td>CTD</td><td>453.621311</td><td>🟥 584.295082</td><td>578.331066</td><td>🟩🟦 432.121311</td></tr>
    <tr><td>CAR</td><td>731.434426</td><td>770.538441</td><td>🟥 773.016897</td><td>🟩🟦 669.173770</td></tr>
    <tr><td colspan="6"></td></tr>
    <!-- ACR -->
    <tr><td rowspan="3">ACR</td><td>3GD</td><td>🟥 0.534745</td><td>0.249688</td><td>0.231299</td><td>🟩🟦 0.229476</td></tr>
    <tr><td>CTD</td><td>🟥 716.051309</td><td>70.190285</td><td>🟩 46.282942</td><td>🟦 190.077487</td></tr>
    <tr><td>CAR</td><td>🟥 250.330890</td><td>236.814151</td><td>233.661219</td><td>🟩🟦 180.235602</td></tr>
    <tr><td colspan="6"></td></tr>
    <!-- CVS -->
    <tr><td rowspan="3">CVS</td><td>3GD</td><td>🟥 0.581966</td><td>0.341521</td><td>0.384304</td><td>🟩🟦 0.124729</td></tr>
    <tr><td>CTD</td><td>🟥 269.929800</td><td>🟩 52.429900</td><td>99.247400</td><td>🟦 59.904300</td></tr>
    <tr><td>CAR</td><td> 🟩 5.394300</td><td>🟥 20.366700</td><td>20.262100</td><td>17.777000</td></tr>
    <tr><td colspan="6"></td></tr>
    <!-- BPIC12W -->
    <tr><td rowspan="3">BPIC12W</td><td>3GD</td><td>🟥 0.794951</td><td>0.662948</td><td>0.631976</td><td>🟩🟦 0.350644</td></tr>
    <tr><td>CTD</td><td>🟥 191.924472</td><td>153.372272</td><td>92.282846</td><td>🟩🟦 32.075031</td></tr>
    <tr><td>CAR</td><td>🟩 28.736770</td><td>🟥 97.492991</td><td>92.738075</td><td>79.711304</td></tr>
    <tr><td colspan="6"></td></tr>
    <!-- BPIC17W -->
    <tr><td rowspan="3">BPIC17W</td><td>3GD</td><td>🟥 0.814347</td><td>0.584121</td><td>0.581658</td><td>🟩🟦 0.319168</td></tr>
    <tr><td>CTD</td><td>🟥 148.790470</td><td>114.314488</td><td>🟩 36.529998</td><td>🟦 54.971774</td></tr>
    <tr><td>CAR</td><td>🟥 145.961497</td><td>🟩 24.326364</td><td>82.289713</td><td>🟦 53.717958</td></tr>
    <tr><td colspan="6"></td></tr>
  </tbody>
</table>

**For more detailed analysis, please check the [Analysis Notebook](analysis.ipynb).**


## 🔬 Experiments Replicability Instructions:

### To see results:
<ol>
    <li>
        <strong>Clone this repository.</strong>
    </li>
    <li>
        <strong>Create environment:</strong>
        <pre><code>$ conda env create -f environment.yml</code></pre>
    </li>
    <li>
        <strong>Unzip output folder:</strong>
        <pre><code>$ unzip output.zip</code></pre>
    </li>
    <li>
        <strong>Unzip SOTA folder:</strong>
        <pre><code>$ unzip sota_results.zip</code></pre>
    </li>
</ol>

Run the [Analysis Notebook](analysis.ipynb) for exploring the results.


### To re-launch experiments:
<ol>
    <li>
        <strong>Clone this repository.</strong>
    </li>
    <li>
        <strong>Create environment:</strong>
        <pre><code>$ conda env create -f environment.yml</code></pre>
    </li>
    <li>
        <strong>Unzip data folder:</strong>
        <pre><code>$ unzip data.zip</code></pre>
    </li>
    <li>
        <strong>Run experiments:</strong>
        <pre><code>$ python main.py</code></pre>
    </li>
    <li>
        <strong>(Optionally) Compute rules metrics:</strong>
        <pre><code>$ python conmpute_rules_metrics.py</code></pre>
    </li>
    <li>
        <strong>(Optionally) (Re-)Evaluate SOTA:</strong>
        <pre><code>$ python evaluate_sota.py</code></pre>
    </li>
</ol>